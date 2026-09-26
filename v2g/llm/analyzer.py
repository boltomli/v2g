"""Analyze video (frames or full video) and produce a structured game design document.

The design document is intended to capture enough detail for a near 1:1 recreation
of the source content as a Godot game — characters, scenes, narrative beats,
visual style, spatial layout, and gameplay mechanics.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ValidationError

from v2g import cache, runlog
from v2g.config import settings
from v2g.llm import jsonfix
from v2g.llm.client import ChatResult, chat
from v2g.llm.errors import LLMOutputError
from v2g.video.dialogue import TranscriptLine, format_transcript
from v2g.video.extractor import _get_duration

# Import OpenAI exceptions for chunked analysis error handling
try:
    from openai import APIError as _OpenAIError
except ImportError:

    class _OpenAIError(Exception):
        pass


log = logging.getLogger(__name__)

# ── JSON schema (shared by both prompts) ────────────────────────────────────

_JSON_SCHEMA = """\
{
  "title": "string — game title derived from the content",
  "genre": "string — primary genre (platformer, adventure, puzzle, action, RPG, etc.)",
  "summary": "string — 2-3 sentence pitch capturing the essence of the source",
  "narrative": "string — full story arc: setup, conflict, climax, resolution. Include character motivations and key plot beats.",
  "mechanics": ["string — core gameplay mechanics (movement, combat, dialogue, puzzle, stealth, etc.)"],
  "controls": ["string — control descriptions with context (e.g. 'A/D or arrows: walk left/right', 'Space: jump / advance dialogue')"],
  "style": "string — art/visual style: color palette, lighting mood, era/setting, rendering approach. Specific enough for an artist.",
  "physics": "string — movement rules: gravity, speed, momentum, collision behavior, special movement (fly, swim, dash, teleport)",
  "progression": "string — how difficulty/content evolves: unlock order, enemy scaling, area progression, resource economy",
  "atmosphere": "string — mood, tone, emotional arc. Sound design cues (music genre, ambient sounds, silence moments).",

  "characters": [
    {
      "name": "string — character name or identifier (ONLY someone actually visible on screen; a name merely mentioned in the transcript is not a character — list every alias of one visible person here, separated by ' / ')",
      "face_id": "string — identity anchor: a stable ID like 'char_01' that stays the same even when the name or costume changes across scenes",
      "role": "string — protagonist / antagonist / NPC / companion / boss / minion / merchant / narrator",
      "visual": "string — PRIMARY appearance (most common outfit): body shape, face, hair, build, distinguishing features",
      "personality": "string — temperament, speech patterns, motivations, quirks",
      "behavior": "string — AI behavior: patrol patterns, attack patterns, dialogue triggers, reaction to player",
      "abilities": ["string — special abilities, attacks, or interactions (can be empty array)"],
      "relationships": "string — how they relate to other characters and the player",
      "personas": [
        {
          "outfit": "string — costume name (e.g. 'armored battle gear', 'casual clothes', 'disguise')",
          "visual": "string — full appearance in this outfit (face stays the same, describe what changes)",
          "context": "string — when/where this look appears (e.g. 'act 1 forest scene', 'final battle')"
        }
      ]
    }
  ],

  "objects": [
    {
      "name": "string — object/element name",
      "role": "string — player / enemy / obstacle / collectible / environment / trigger / UI / vehicle / weapon / decoration",
      "visual": "string — detailed visual description: shape, size, color, texture, animation states",
      "behavior": "string — how it moves, reacts, interacts. Include timing, conditions, state changes.",
      "spatial": "string — where it appears in the scene: position relative to other elements, layer (foreground/mid/background)"
    }
  ],

  "scenes": [
    {
      "name": "string — scene/level/location name",
      "description": "string — what happens in this scene: events, dialogue, transitions, player actions",
      "layout": "string — spatial layout: ground, platforms, walls, doors, exits. Use compass/directional language.",
      "goals": ["string — what the player must accomplish to progress"],
      "hazards": ["string — dangers, enemies, traps, time limits, environmental threats"],
      "triggers": ["string — events that activate: cutscenes, spawns, door opens, dialogue, phase change"],
      "visual_theme": "string — colors, lighting, textures specific to this scene"
    }
  ],

  "scene_transitions": [
    {
      "from": "string — source scene name",
      "to": "string — destination scene name",
      "trigger": "string — what causes the transition (reach exit, defeat boss, cutscene ends, item collected)",
      "effect": "string — visual transition effect (fade, wipe, instant, camera pan)"
    }
  ],

  "dialogue_samples": [
    {
      "speaker": "string — speaker name; may be an off-screen voice with NO character entry (empty for narration entries)",
      "line": "string — VERBATIM source-language line copied from the transcript (or dialogue you can actually perceive); EMPTY for narration",
      "line_zh": "string — REQUIRED. Simplified Chinese: translation of `line` for dialogue, original Chinese narration otherwise",
      "context": "string — when/why this line is said",
      "choices": [
        {"line": "string — source-language option text ONLY if it appears verbatim in the transcript, else empty",
         "line_zh": "string — REQUIRED. Simplified Chinese option text",
         "score": 1}
      ]
    }
  ]
}
"""

# ── Shared rule blocks (both prompts) ──────────────────────────────────────

_LANGUAGE_RULES = """\
LANGUAGE AND SUBTITLE RULES (hard requirements):
- TARGET LANGUAGE: Simplified Chinese. Every dialogue entry MUST carry `line_zh`.
- SOURCE LANGUAGE: any non-Chinese text you put in `line` MUST be transcribed
  VERBATIM from the source material — the provided transcript, or dialogue you
  can actually perceive in the video. NEVER invent, paraphrase, or translate
  into a foreign language yourself.
- No transcript provided (or you cannot perceive speech): leave `line` EMPTY.
  An empty `line` marks a narration entry — write it in `line_zh` only.
- Narration and stage directions not spoken in the video are Chinese-only
  (`line` empty, `line_zh` carries the text).
- Choice options are player-authored UI text: Chinese-only (`line` empty)
  unless the exact wording appears in the transcript.
"""

_VN_RULES = """\
GAME FORM (hard requirements):
- The output is a VISUAL NOVEL: a dialogue-driven story game with choices.
  No platformer physics, no free movement, no combat.
- `dialogue_samples` IS the full playable story script in playing order: every
  usable source line from the transcript, plus Chinese narration beats bridging
  the scenes, ending with a closing beat.
- Attach `choices` to entries where the player must decide; each option's
  `score` feeds reputation/alliance tracking.
- `controls` describes only the VN scheme: Space/Enter/click advances dialogue,
  mouse picks choices. Never describe move/jump/combat controls.
- `mechanics`, `scenes`, and `progression` adapt the source content into
  choice-driven narrative beats, not platformer levels to traverse.
"""

_CAST_RULES = """\
CHARACTER GROUNDING (the pictures decide who exists, not the transcript):
- `characters` holds ONLY people who APPEAR ON SCREEN in the frames/video. The
  transcript is dialogue text, not a cast list: someone who is mentioned,
  asked after, or spoken of but never shown is NOT a character — no entry, and
  never an invented `visual` for them, however important to the story.
- One visible person = ONE entry however many names or titles the dialogue
  uses for them: put every alias in `name` (e.g. "John / The Stranger") instead
  of fanning one face out into several characters.
- Speakers are not automatically characters: an off-screen voice, a narrator or
  a mentioned party may be a `dialogue_samples.speaker` with no portrait of
  their own.
"""

# ── System prompts ──────────────────────────────────────────────────────────

_SYSTEM_FRAMES = f"""\
You are a world-class game designer and adaptation specialist. The user will provide
key frames extracted from a video. Your task is to analyze these frames and produce
a game design document that enables a near 1:1 recreation of the source content
as a playable game.

ANALYSIS APPROACH:
1. First, identify the source material: what show/movie/game/story is this?
2. Catalog every character visible — their appearance, role, personality, behavior
3. Map every distinct scene/environment — layout, objects, atmosphere
4. Identify the narrative arc — what story beats occur in sequence
5. Note every interactive element — what could a player interact with
6. Capture the visual style precisely — palette, lighting, era, mood
7. Infer gameplay mechanics from observed actions (fighting, running, talking, exploring)

FAITHFULNESS RULES:
- Characters MUST be described in enough detail to recreate them visually
- Scenes MUST include spatial layout so a level designer could build them
- The narrative MUST preserve the original story beats and character motivations
- The style MUST match the source — not a generic approximation
- Every distinct visual element in the frames should appear in the design

CHARACTER IDENTITY RULES:
- **face_id**: Assign a stable ID (e.g. "char_01") to each character. This ID must stay
  the same even if you don't know their name — use body type, hair, skin tone, and
  distinguishing features as anchors.
- **visual**: Describe the PRIMARY (most common) appearance. Focus on FACE and BODY
  features that never change (face shape, hair color, skin tone, build, scars, age).
- **personas**: If the same person appears in different costumes/outfits in different
  scenes, list each as a separate persona. The face/body description should be
  consistent — only clothing and gear change.
- **Do NOT create separate characters for the same person in different clothes.**
  A warrior in armor and the same man in casual clothes are ONE character with TWO personas.
If you cannot tell whether two people are the same, use face_id to mark them as
"possibly same" — the merge system will reconcile.

{_CAST_RULES}
{_LANGUAGE_RULES}
{_VN_RULES}
{_JSON_SCHEMA}

Return ONLY the JSON, no markdown fences, no commentary.
"""

_SYSTEM_VIDEO = f"""\
You are a world-class game designer, film analyst, and adaptation specialist.
The user will provide a full video. Your task is to analyze it thoroughly and produce
a game design document that enables a near 1:1 recreation of the source content.

WATCH THE VIDEO MULTIPLE TIMES:
1. First pass: identify the source material, overall narrative, cast of characters
2. Second pass: catalog every scene transition, character action, environmental detail
3. Third pass: note timing, rhythm, spatial relationships, camera angles, sound cues

ANALYSIS PRIORITIES:
- **Characters**: Every distinct character — appearance (in detail), personality, abilities,
  relationships, behavior patterns, dialogue style
- **Scenes/Environments**: Every location — layout, interactive objects, visual theme,
  hazards, transitions between areas
- **Narrative**: Full story arc — setup, conflict, escalation, climax, resolution.
  Include character motivations and key dialogue
- **Mechanics**: All observed gameplay-relevant actions — movement types, combat,
  puzzle-solving, dialogue choices, item usage, special abilities
- **Visual Style**: Precise description — color palette, lighting, era/setting,
  rendering style, UI elements, text overlays
- **Physics**: Movement rules — gravity, speed, momentum, collision, special movement
- **Progression**: How challenge/content evolves — enemy scaling, area unlocking,
  resource economy, difficulty curve
- **Atmosphere**: Mood, tone, emotional arc, sound design cues

FAITHFULNESS RULES:
- Characters MUST be described so an artist could recreate them
- Scenes MUST have enough spatial detail for a level designer
- The narrative MUST preserve original story beats and character voices
- The style MUST match the source exactly — not a genre approximation
- Include representative dialogue lines for each speaking character

CHARACTER IDENTITY RULES:
- **face_id**: Assign a stable ID (e.g. "char_01") to each character. This ID must stay
  the same even if you don't know their name — use face shape, hair, skin tone, build,
  scars, and age as anchors.
- **visual**: Describe the PRIMARY (most common) appearance. Focus on FACE and BODY
  features that never change across costumes.
- **personas**: If the same person changes costume across scenes, list EACH outfit as a
  separate persona with its own visual description and the scene context where it appears.
  Face/body features must remain consistent — only clothing/gear changes.
- **Do NOT create separate characters for the same person in different clothes.**
  A man in a suit and the same man in gym clothes are ONE character with TWO personas.
- Same character referred to by different names in different scenes → merge under one entry,
  note all names in the "name" field (e.g. "John / The Stranger").

{_CAST_RULES}
{_LANGUAGE_RULES}
{_VN_RULES}
{_JSON_SCHEMA}

Return ONLY the JSON, no markdown fences, no commentary.
"""


# ── Pydantic models ────────────────────────────────────────────────────────


class Persona(BaseModel):
    """A costume/outfit variant of a character. Same person, different look."""

    outfit: str = ""
    visual: str = ""
    context: str = ""  # when/where this look appears


class Character(BaseModel):
    name: str
    role: str
    visual: str  # primary appearance (default outfit)
    personality: str = ""
    behavior: str = ""
    abilities: list[str] = []
    relationships: str = ""
    personas: list[Persona] = []  # alternate costumes/outfits
    face_id: str = ""  # identity anchor for cross-segment dedup


class GameObject(BaseModel):
    name: str
    role: str
    visual: str
    behavior: str = ""
    spatial: str = ""


class SceneDesign(BaseModel):
    name: str
    description: str
    layout: str = ""
    goals: list[str] = []
    hazards: list[str] = []
    triggers: list[str] = []
    visual_theme: str = ""


class SceneTransition(BaseModel):
    source: str = ""  # "from" is a reserved word in some contexts
    destination: str = ""  # "to"
    trigger: str = ""
    effect: str = ""

    model_config = {"populate_by_name": True}

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        """Handle from/to field mapping."""
        import json as _json

        if isinstance(json_data, (str, bytes)):
            data = _json.loads(json_data)
        else:
            data = json_data
        if isinstance(data, dict):
            if "from" in data and "source" not in data:
                data["source"] = data.pop("from")
            if "to" in data and "destination" not in data:
                data["destination"] = data.pop("to")
        return super().model_validate(data, **kwargs)


class DialogueChoice(BaseModel):
    """One player-facing choice option. `line` only if verbatim from source."""

    line: str = ""
    line_zh: str = ""
    score: int = 1


class DialogueSample(BaseModel):
    """One story-script entry. Empty `line` = narration (Chinese-only)."""

    speaker: str = ""
    line: str = ""  # source language, VERBATIM from video transcript only
    line_zh: str = ""  # Simplified Chinese: translation / original narration
    context: str = ""
    choices: list[DialogueChoice] = []


class GameDesign(BaseModel):
    title: str
    genre: str
    summary: str
    narrative: str = ""
    mechanics: list[str]
    controls: list[str]
    style: str
    physics: str = ""
    progression: str = ""
    atmosphere: str = ""

    characters: list[Character] = []
    objects: list[GameObject]
    scenes: list[SceneDesign] = []
    scene_transitions: list[SceneTransition] = []
    dialogue_samples: list[DialogueSample] = []

    # Legacy aliases for backward compat
    @property
    def levels(self) -> list[SceneDesign]:
        return self.scenes


def _normalize(data: dict) -> dict:
    """Pre-process a decoded candidate: scene_transitions from/to → source/destination,
    legacy `levels` key → `scenes`."""
    if "scene_transitions" in data:
        for t in data["scene_transitions"]:
            if "from" in t and "source" not in t:
                t["source"] = t.pop("from")
            if "to" in t and "destination" not in t:
                t["destination"] = t.pop("to")
    if "levels" in data and "scenes" not in data:
        data["scenes"] = data.pop("levels")
    return data


def _parse(raw: str) -> GameDesign:
    """Clean, repair and parse LLM output into GameDesign.

    Repairs markdown fences, surrounding prose, stray tokens mid-document
    (deleted at the decode-error position), and truncation at max_tokens.
    Repairs yield several candidate dicts — each is normalized and validated
    in turn, so a repair that decodes but mangles a required field is skipped
    in favor of a fuller one. If nothing parses, the raw response is dumped to
    <run>/llm/ and LLMOutputError says where to look.
    """
    import json as _json

    dump = runlog.llm_dump("analysis", raw)  # keep the raw response either way
    where = f" Raw response saved to {dump}." if dump else ""
    last_err: Exception | None = None
    direct: dict | None = None
    for cand in jsonfix.json_candidates(raw):
        try:
            parsed = _json.loads(cand)
        except _json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(parsed, dict):
            direct = parsed
            break
        last_err = ValueError(f"top-level JSON value is {type(parsed).__name__}, not an object")

    # A clean decode is validated alone: schema failures there are the model's,
    # and no text repair can add fields that were never emitted.
    stream: Iterator[dict] = (
        iter([direct]) if direct is not None else jsonfix.repair_candidates(raw)
    )
    first_val_err: ValidationError | None = None
    for data in stream:
        try:
            return GameDesign.model_validate(_normalize(data))
        except ValidationError as e:
            if first_val_err is None:
                first_val_err = e
    if first_val_err is None:
        raise LLMOutputError(
            f"Cannot parse LLM design JSON: {last_err}.{where}"
            " If the response was truncated, raise max_tokens or shorten the design."
        )
    raise LLMOutputError(
        f"LLM design failed validation: {first_val_err}.{where}"
    ) from first_val_err


def _design_from(res: ChatResult) -> GameDesign:
    """Parse one transport result — content filtering is not a JSON problem."""
    if res.finish == "content_filter":
        raise LLMOutputError(
            "LLM response was content-filtered (finish=content_filter) — unusable "
            "and never cached; check the input frames against the provider's policy"
        )
    return _parse(res.text)


def _request_design(system: str, parts: list[str | Path], **chat_kw) -> GameDesign:
    """Parse layer above transport, with a bounded refresh policy.

    A fresh response that fails to parse is never re-requested — that would
    just re-pay for the same input. Only a *cached* response that proves bad
    is invalidated and refetched, at most once.
    """
    res = chat(system, parts, **chat_kw)
    try:
        return _design_from(res)
    except LLMOutputError:
        if not res.cached:
            raise
        cache.invalidate(res.key)
        log.warning("Discarded invalid cached LLM response; refetching once (key=%.12s)", res.key)
        return _design_from(chat(system, parts, refresh=True, **chat_kw))


def analysis_key(
    source_video: Path,
    transcript: list[TranscriptLine],
    *,
    detailed: bool,
) -> str:
    """Coarse key for a whole analysis result (run-local checkpoint).

    Covers everything that shapes the **stage 1** design: source file identity,
    mode (system prompt + token budget), extraction settings, model and the
    transcript. Deliberately theme-free — the theme enters in stage 2, so every
    themed variant of one video reuses the same analysis checkpoint.
    """
    import hashlib
    import json as _json

    stat = source_video.stat()
    material = _json.dumps(
        {
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime_ns,
            "model": settings.llm_model,
            "system": _SYSTEM_VIDEO if detailed else _SYSTEM_FRAMES,
            "max_tokens": 16384 if detailed else 8192,
            "detailed": detailed,
            "transcript": format_transcript(transcript),
            "extract": {
                "frame_interval": settings.frame_interval,
                "scene_threshold": settings.scene_threshold,
                "frame_budget": settings.frame_budget,
                "max_duration": settings.max_duration,
                "video_max_mb": settings.video_max_mb,
                "chunk_duration": settings.chunk_duration,
            },
        },
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def save_checkpoint(run_dir: Path, design: GameDesign, key: str) -> None:
    """Persist the analysis layer's output so a rerun of this run skips the LLM."""
    (run_dir / "design.json").write_text(design.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "design.key").write_text(key, encoding="utf-8")


def load_checkpoint(run_dir: Path, key: str) -> GameDesign | None:
    """Checkpointed design when inputs are unchanged, else None (stale/absent)."""
    try:
        if (run_dir / "design.key").read_text(encoding="utf-8").strip() != key:
            log.info("Analysis checkpoint stale (inputs changed) — will re-run the LLM")
            return None
        return GameDesign.model_validate_json((run_dir / "design.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing/corrupt files; ValidationError ⊂ ValueError
        return None


def _transcript_note(transcript: str | None) -> str:
    """Wrap the extracted source transcript for injection into the user message."""
    if transcript:
        return (
            "\n=== TRANSCRIPT EXTRACTED FROM THE SOURCE VIDEO "
            "(authoritative for every `line` — DIALOGUE ONLY, not a cast list: "
            "being named here does not make someone a character) ===\n"
            f"{transcript}\n=== END TRANSCRIPT ===\n"
        )
    return (
        "\n(No transcript is available for this video and you cannot rely on "
        "speech: leave every `line` EMPTY — narration goes into `line_zh` only.)\n"
    )


def _theme_block(instruct: str) -> str:
    """The THEME INSTRUCTION block appended to every analysis message.

    Also hashed into ``analysis_key`` so editing this template invalidates
    run-local checkpoints produced under the old wording.
    """
    return (
        f"=== THEME INSTRUCTION ===\n"
        f"{instruct}\n\n"
        f"This is a FULL REDESIGN, not a light reskin — the game must no longer look "
        f"like the source video. Keep ONLY the story structure, beat order and character "
        f"motivations; rewrite every visual field to fit the theme above:\n"
        f"- characters: new names, faces, outfits, hairstyles, palettes and signature "
        f"poses/actions — no character keeps the source's look\n"
        f"- objects: re-invented shape, material and colors\n"
        f"- scenes: new layout, palette, lighting and atmosphere\n"
        f"- style and atmosphere: rewritten around the theme\n"
        f"The FAITHFULNESS RULES in the system prompt describe the SOURCE analysis only; "
        f"wherever the theme and the source disagree, the theme wins.\n"
        f"========================="
    )


_NO_THEME = (
    "No theme was given. Keep the world and the story recognisable, but re-skin the "
    "presentation so the result does not look like the source video — a different "
    "palette, rendering style and lighting is enough."
)

_REWRITE_SYSTEM = (
    "You re-skin a game design for a new style without touching its story. "
    "Answer only with the complete GameDesign JSON document."
)


def rewrite_design(design: GameDesign, instruct: str | None) -> GameDesign:
    """Stage 2 (text half): re-skin the design's presentation — theme or not.

    Runs after analysis **and** asset extraction, so nothing theme-shaped ever
    reaches stage 1: one source video → one faithful analysis → one extraction,
    and each theme is just another re-skin of that.

    Identity is frozen, not merely requested: character / object / scene names
    and ``face_id`` are what stage 1's asset keys, speaker→portrait mapping and
    scene keys are built from, and ``dialogue_samples`` is transcribed from the
    source video — a rewrite that renames or drops any of them would orphan the
    sprites just extracted, so those are restored/validated here.

    A failed or identity-breaking re-skin returns the design unchanged: it can
    never fail a run.
    """
    user = (
        "Re-skin this GameDesign for the style below.\n"
        "REWRITE — presentation only: title, summary, narrative, style, atmosphere, "
        "character visual/personas, object visual/spatial, scene layout/visual_theme.\n"
        "KEEP UNCHANGED — identity and gameplay: character names and face_id, object "
        "names, scene names, roles, mechanics, controls, physics, progression, scene "
        "goals/hazards/triggers, and every dialogue_samples entry (transcribed from the "
        "source video).\n\n"
        f"{_theme_block(instruct or _NO_THEME)}\n\n"
        "GameDesign JSON:\n" + design.model_dump_json(indent=2)
    )
    try:
        rewritten = _request_design(_REWRITE_SYSTEM, [user], max_tokens=16384, temperature=0.3)
    except Exception as e:  # noqa: BLE001 — a failed re-skin must never fail the run
        log.warning("Design re-skin failed (%s) — keeping the source-faithful design", e)
        return design
    rewritten.dialogue_samples = design.dialogue_samples  # transcript authority, not the model's
    for label, old, new in (
        ("character", {c.name for c in design.characters}, {c.name for c in rewritten.characters}),
        ("object", {o.name for o in design.objects}, {o.name for o in rewritten.objects}),
        ("scene", {s.name for s in design.scenes}, {s.name for s in rewritten.scenes}),
    ):
        missing = sorted(old - new)
        if missing:
            log.warning(
                "Design re-skin dropped %s id(s) %s — keeping the source-faithful design",
                label,
                ", ".join(missing)[:120],
            )
            return design
    return rewritten


def analyze(
    frames: list[Path],
    *,
    transcript: str | None = None,
) -> GameDesign:
    """Stage 1 (fast mode): analyze keyframes of the ORIGINAL video → GameDesign.

    Always source-faithful: the theme is applied later, in stage 2, so this
    analysis is shared by every themed variant of the same video.
    """
    user_msg = (
        "Here are key frames extracted from a video. "
        "Analyze every frame in detail: identify characters, environments, narrative beats, "
        "visual style, and interactive elements. Generate a comprehensive game design document "
        "that would allow a developer to recreate this content as a playable game.\n"
        + _transcript_note(transcript)
    )
    parts: list[str | Path] = [user_msg]
    parts.extend(frames)
    # Fast-mode budget must stay aligned with analysis_key's max_tokens field.
    return _request_design(_SYSTEM_FRAMES, parts, max_tokens=8192)


def analyze_video(
    video_path: Path,
    *,
    transcript: str | None = None,
) -> GameDesign:
    """Stage 1 (detailed mode): analyze the ORIGINAL video file → GameDesign.

    Sends the video directly to a model with video understanding support.
    Produces the most faithful analysis with full narrative, character detail,
    scene-by-scene breakdown, and dialogue samples. Source-faithful by design —
    any theme is applied afterwards (stage 2), never here.
    """
    user_msg = (
        "Here is a video. Watch it carefully — multiple times if needed. "
        "Identify every character, environment, story beat, and visual element. "
        "Generate a comprehensive game design document that captures the source "
        "content with enough fidelity for a near 1:1 recreation as a playable game.\n"
        + _transcript_note(transcript)
    )
    parts: list[str | Path] = [user_msg, video_path]
    return _request_design(_SYSTEM_VIDEO, parts, max_tokens=16384, temperature=0.3)


def analyze_video_chunked(
    segments: list[Path],
    *,
    chunk_label: str = "",
    transcript: list[TranscriptLine] | None = None,
) -> GameDesign:
    """Analyze a long video in segments and merge results.

    Each segment is analyzed independently, then all results are merged into
    a single GameDesign. This avoids LLM upload-size limits for long videos.

    Args:
        segments: Ordered list of video segment file paths.
        chunk_label: Prefix for progress messages (e.g. "1/3").
        transcript: Full-source transcript; each segment only receives the
            lines whose timestamps fall inside its window.
    """
    if len(segments) == 1:
        return analyze_video(
            segments[0],
            transcript=(format_transcript(transcript) if transcript else None),
        )

    import logging

    log = logging.getLogger(__name__)

    designs: list[GameDesign] = []
    # Segment lengths vary (trailing segment, size-driven splits) — anchor each
    # transcript window to the segments' actual durations, not the chunk grid.
    windows: list[tuple[float, float]] = []
    if transcript:
        cursor = 0.0
        for seg in segments:
            seg_dur = _get_duration(seg)
            windows.append((cursor, cursor + seg_dur))
            cursor += seg_dur
    for i, seg in enumerate(segments):
        label = f"[{i + 1}/{len(segments)}]"
        log.info("Analyzing segment %s", label)
        seg_transcript: str | None = None
        if transcript:
            seg_transcript = (
                format_transcript(
                    transcript,
                    start=windows[i][0],
                    end=windows[i][1],
                )
                or None
            )
        try:
            d = analyze_video(seg, transcript=seg_transcript)
            designs.append(d)
            log.info(
                "  Segment %s: '%s' — %d chars, %d scenes",
                label,
                d.title,
                len(d.characters),
                len(d.scenes),
            )
        except (ValueError, RuntimeError, KeyError, OSError, _OpenAIError) as e:
            log.warning("  Segment %s failed: %s", label, e)

    if not designs:
        raise RuntimeError("All video segments failed analysis")

    return _merge_designs(designs)


def _merge_designs(designs: list[GameDesign]) -> GameDesign:
    """Merge multiple GameDesign objects (from video segments) into one.

    Character merging strategy (face/identity-based):
    1. Match by face_id (exact, case-insensitive) — strongest signal
    2. Match by name similarity (normalized) — catches same-name appearances
    3. Match by visual feature overlap — catches unnamed/recurring extras
    4. Different costumes → merged as personas of the same character
    """
    if len(designs) == 1:
        return designs[0]

    base = designs[0].model_copy()

    # ── Merge characters by identity (face), not just name ───────────────
    merged_chars: list[Character] = list(base.characters)

    for d in designs[1:]:
        for c in d.characters:
            match_idx = _find_character_match(c, merged_chars)
            if match_idx is not None:
                _absorb_character(merged_chars[match_idx], c)
            else:
                merged_chars.append(c)
    base.characters = merged_chars

    # Merge objects by name (keep most detailed)
    obj_map: dict[str, GameObject] = {}
    for o in base.objects:
        obj_map[o.name.lower()] = o
    for d in designs[1:]:
        for o in d.objects:
            key = o.name.lower()
            if key not in obj_map or len(o.visual) + len(o.behavior) > len(
                obj_map[key].visual
            ) + len(obj_map[key].behavior):
                obj_map[key] = o
    base.objects = list(obj_map.values())

    # Concatenate scenes in order
    all_scenes = list(base.scenes)
    seen_scene_names = {s.name.lower() for s in all_scenes}
    for d in designs[1:]:
        for s in d.scenes:
            if s.name.lower() not in seen_scene_names:
                all_scenes.append(s)
                seen_scene_names.add(s.name.lower())
    base.scenes = all_scenes

    # Merge transitions
    seen_transitions = {(t.source, t.destination) for t in base.scene_transitions}
    for d in designs[1:]:
        for t in d.scene_transitions:
            if (t.source, t.destination) not in seen_transitions:
                base.scene_transitions.append(t)
                seen_transitions.add((t.source, t.destination))

    # Union mechanics and controls
    mech_set = set(base.mechanics)
    ctrl_set = set(base.controls)
    for d in designs[1:]:
        mech_set.update(d.mechanics)
        ctrl_set.update(d.controls)
    base.mechanics = list(mech_set)
    base.controls = list(ctrl_set)

    # Concatenate narratives
    narratives = [d.narrative for d in designs if d.narrative]
    base.narrative = "\n\n".join(narratives) if narratives else base.narrative

    # Concatenate dialogue, dropping exact repeats across segments
    seen_lines = {(ds.speaker.lower(), ds.line, ds.line_zh) for ds in base.dialogue_samples}
    for d in designs[1:]:
        for ds in d.dialogue_samples:
            key = (ds.speaker.lower(), ds.line, ds.line_zh)
            if key not in seen_lines:
                base.dialogue_samples.append(ds)
                seen_lines.add(key)

    # Take longest (most detailed) for these text fields
    for field in ("physics", "progression", "atmosphere"):
        values = [getattr(d, field) for d in designs if getattr(d, field)]
        if values:
            setattr(base, field, max(values, key=len))

    return base


# ── Character identity matching ──────────────────────────────────────────────

_FEATURE_KEYWORDS = {
    "hair": [
        "blonde",
        "brown",
        "black",
        "red",
        "white",
        "gray",
        "grey",
        "bald",
        "long",
        "short",
        "curly",
        "straight",
        "ponytail",
        "braid",
    ],
    "build": ["tall", "short", "slim", "muscular", "stocky", "thin", "heavy", "athletic"],
    "skin": ["pale", "dark", "tan", "olive", "fair", "brown"],
    "age": ["young", "old", "elderly", "middle-aged", "teen", "child", "adult"],
    "face": ["beard", "mustache", "scar", "glasses", "freckles", "wrinkles"],
    "distinguishing": ["tattoo", "piercing", "mask", "hood", "cape", "crown", "eye patch"],
}


def _extract_features(text: str) -> set[str]:
    """Extract visual feature keywords from a description string."""
    words = text.lower().split()
    features: set[str] = set()
    for category, keywords in _FEATURE_KEYWORDS.items():
        for kw in keywords:
            if kw in words or kw in text.lower():
                features.add(f"{category}:{kw}")
    return features


def _name_similarity(a: str, b: str) -> float:
    """Compute name similarity (0.0–1.0).

    Handles: exact match, substring, slash-separated aliases, shared words.
    """

    def _norm(s: str) -> list[str]:
        # Split on /, |, or " / " to handle "John / The Stranger"
        parts: list[str] = []
        for segment in s.replace("/", " / ").replace("|", " | ").split(" / "):
            parts.extend(segment.strip().lower().split())
        return [p for p in parts if p and len(p) > 1]

    a_words = _norm(a)
    b_words = _norm(b)
    if not a_words or not b_words:
        return 0.0

    a_set = set(a_words)
    b_set = set(b_words)
    shared = a_set & b_set
    if not shared:
        return 0.0

    # Jaccard-ish: shared / union, but weighted toward the smaller set
    return len(shared) / min(len(a_set), len(b_set))


def _visual_feature_overlap(a: str, b: str) -> float:
    """How much do two visual descriptions share? (0.0–1.0)"""
    fa = _extract_features(a)
    fb = _extract_features(b)
    if not fa or not fb:
        return 0.0
    shared = fa & fb
    return len(shared) / min(len(fa), len(fb))


def _find_character_match(target: Character, candidates: list[Character]) -> int | None:
    """Find the index of the best-matching character in *candidates*.

    Matching priority:
    1. face_id exact match (case-insensitive)
    2. Name similarity ≥ 0.7 (handles "John" vs "John / The Stranger")
    3. Visual feature overlap ≥ 0.5 AND same role
    Returns None if no confident match.
    """
    target_fid = target.face_id.lower().strip()

    for i, c in enumerate(candidates):
        # 1. face_id exact match
        if target_fid and c.face_id and target_fid == c.face_id.lower().strip():
            return i

    for i, c in enumerate(candidates):
        # 2. Name similarity
        sim = _name_similarity(target.name, c.name)
        if sim >= 0.7:
            return i

    for i, c in enumerate(candidates):
        # 3. Visual feature overlap + same role
        if target.role == c.role:
            overlap = _visual_feature_overlap(target.visual, c.visual)
            if overlap >= 0.5:
                return i

    return None


def _absorb_character(existing: Character, incoming: Character) -> None:
    """Merge *incoming* into *existing* as a persona variant.

    - If the visual description differs significantly, adds as a new Persona
    - Upgrades fields if incoming has more detail
    - Merges abilities and relationships
    """
    # Compare visuals — if different enough, incoming is a new persona
    overlap = _visual_feature_overlap(existing.visual, incoming.visual)
    visual_differs = (
        overlap < 0.85 or existing.visual.lower().strip() != incoming.visual.lower().strip()
    )

    if visual_differs:
        # Check this isn't already captured as a persona
        existing_visuals = {existing.visual.lower()} | {p.visual.lower() for p in existing.personas}
        if incoming.visual.lower().strip() not in existing_visuals:
            existing.personas.append(
                Persona(
                    outfit=incoming.visual.split(",")[0][:60] if "," in incoming.visual else "",
                    visual=incoming.visual,
                    context=f"alt appearance from {incoming.name}",
                )
            )

    # Upgrade fields if incoming is more detailed
    if len(incoming.visual) > len(existing.visual):
        existing.visual = incoming.visual
    if len(incoming.behavior) > len(existing.behavior):
        existing.behavior = incoming.behavior
    if len(incoming.personality) > len(existing.personality):
        existing.personality = incoming.personality
    if len(incoming.relationships) > len(existing.relationships):
        existing.relationships = incoming.relationships

    # Merge abilities
    existing_set = set(existing.abilities)
    for a in incoming.abilities:
        if a.lower() not in {x.lower() for x in existing_set}:
            existing.abilities.append(a)
            existing_set.add(a)

    # Keep richer face_id
    if not existing.face_id and incoming.face_id:
        existing.face_id = incoming.face_id

    # Merge slash-separated names
    existing_names = {n.strip().lower() for n in existing.name.replace("/", " / ").split(" / ")}
    for n in incoming.name.replace("/", " / ").split(" / "):
        n = n.strip()
        if n and n.lower() not in existing_names:
            existing.name = f"{existing.name} / {n}"
            existing_names.add(n.lower())
