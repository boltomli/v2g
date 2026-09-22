"""Analyze video (frames or full video) and produce a structured game design document.

The design document is intended to capture enough detail for a near 1:1 recreation
of the source content as a Godot game — characters, scenes, narrative beats,
visual style, spatial layout, and gameplay mechanics.
"""

from pathlib import Path

from pydantic import BaseModel

from v2g.config import settings
from v2g.llm.client import chat
from v2g.video.dialogue import TranscriptLine, format_transcript

# Import OpenAI exceptions for chunked analysis error handling
try:
    from openai import APIError as _OpenAIError
except ImportError:
    class _OpenAIError(Exception):
        pass

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
      "name": "string — character name or identifier",
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
      "speaker": "string — character name (empty for narration entries)",
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
    visual: str                     # primary appearance (default outfit)
    personality: str = ""
    behavior: str = ""
    abilities: list[str] = []
    relationships: str = ""
    personas: list[Persona] = []    # alternate costumes/outfits
    face_id: str = ""               # identity anchor for cross-segment dedup


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
    source: str = ""   # "from" is a reserved word in some contexts
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
    line: str = ""       # source language, VERBATIM from video transcript only
    line_zh: str = ""    # Simplified Chinese: translation / original narration
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


def _parse(raw: str) -> GameDesign:
    """Clean LLM output and parse into GameDesign."""
    import json as _json
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    # Pre-process: handle from/to in scene_transitions
    data = _json.loads(cleaned)
    if "scene_transitions" in data:
        for t in data["scene_transitions"]:
            if "from" in t and "source" not in t:
                t["source"] = t.pop("from")
            if "to" in t and "destination" not in t:
                t["destination"] = t.pop("to")

    # Backward compat: map "levels" → "scenes" if present
    if "levels" in data and "scenes" not in data:
        data["scenes"] = data.pop("levels")

    return GameDesign.model_validate(data)


def _transcript_note(transcript: str | None) -> str:
    """Wrap the extracted source transcript for injection into the user message."""
    if transcript:
        return (
            "\n=== TRANSCRIPT EXTRACTED FROM THE SOURCE VIDEO "
            "(authoritative for every `line`) ===\n"
            f"{transcript}\n=== END TRANSCRIPT ===\n"
        )
    return (
        "\n(No transcript is available for this video and you cannot rely on "
        "speech: leave every `line` EMPTY — narration goes into `line_zh` only.)\n"
    )


def _inject_instruct(user_msg: str, instruct: str | None) -> str:
    """Append style instruction to the user message if provided."""
    if not instruct:
        return user_msg
    return (
        f"{user_msg}\n\n"
        f"=== STYLE INSTRUCTION ===\n"
        f"{instruct}\n\n"
        f"You MUST adapt the entire design to match this style/theme while preserving "
        f"the core structure and story beats of the source material. "
        f"Rename characters, reskin environments, adjust tone and atmosphere accordingly.\n"
        f"========================="
    )


def analyze(
    frames: list[Path],
    *,
    instruct: str | None = None,
    transcript: str | None = None,
) -> GameDesign:
    """Analyze video keyframes → GameDesign (fast mode)."""
    user_msg = (
        "Here are key frames extracted from a video. "
        "Analyze every frame in detail: identify characters, environments, narrative beats, "
        "visual style, and interactive elements. Generate a comprehensive game design document "
        "that would allow a developer to recreate this content as a playable game.\n"
        + _transcript_note(transcript)
    )
    parts: list[str | Path] = [_inject_instruct(user_msg, instruct)]
    parts.extend(frames)
    return _parse(chat(_SYSTEM_FRAMES, parts))


def analyze_video(
    video_path: Path,
    *,
    instruct: str | None = None,
    transcript: str | None = None,
) -> GameDesign:
    """Analyze a full video file → GameDesign (detailed mode).

    Sends the video directly to a model with video understanding support.
    Produces the most faithful analysis with full narrative, character detail,
    scene-by-scene breakdown, and dialogue samples.
    """
    user_msg = (
        "Here is a video. Watch it carefully — multiple times if needed. "
        "Identify every character, environment, story beat, and visual element. "
        "Generate a comprehensive game design document that captures the source "
        "content with enough fidelity for a near 1:1 recreation as a playable game.\n"
        + _transcript_note(transcript)
    )
    parts: list[str | Path] = [_inject_instruct(user_msg, instruct), video_path]
    return _parse(chat(_SYSTEM_VIDEO, parts, max_tokens=16384, temperature=0.3))


def analyze_video_chunked(
    segments: list[Path],
    *,
    instruct: str | None = None,
    chunk_label: str = "",
    transcript: list[TranscriptLine] | None = None,
) -> GameDesign:
    """Analyze a long video in segments and merge results.

    Each segment is analyzed independently, then all results are merged into
    a single GameDesign. This avoids LLM upload-size limits for long videos.

    Args:
        segments: Ordered list of video segment file paths.
        instruct: Optional style instruction.
        chunk_label: Prefix for progress messages (e.g. "1/3").
        transcript: Full-source transcript; each segment only receives the
            lines whose timestamps fall inside its window.
    """
    if len(segments) == 1:
        return analyze_video(segments[0], instruct=instruct, transcript=(
            format_transcript(transcript) if transcript else None
        ))

    import logging
    log = logging.getLogger(__name__)

    designs: list[GameDesign] = []
    for i, seg in enumerate(segments):
        label = f"[{i + 1}/{len(segments)}]"
        log.info("Analyzing segment %s", label)
        seg_transcript: str | None = None
        if transcript:
            seg_transcript = format_transcript(
                transcript,
                start=i * settings.chunk_duration,
                end=(i + 1) * settings.chunk_duration,
            ) or None
        try:
            d = analyze_video(seg, instruct=instruct, transcript=seg_transcript)
            designs.append(d)
            log.info("  Segment %s: '%s' — %d chars, %d scenes",
                     label, d.title, len(d.characters), len(d.scenes))
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
            if key not in obj_map or len(o.visual) + len(o.behavior) > len(obj_map[key].visual) + len(obj_map[key].behavior):
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
    "hair": ["blonde", "brown", "black", "red", "white", "gray", "grey", "bald",
             "long", "short", "curly", "straight", "ponytail", "braid"],
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
    visual_differs = overlap < 0.85 or existing.visual.lower().strip() != incoming.visual.lower().strip()

    if visual_differs:
        # Check this isn't already captured as a persona
        existing_visuals = {existing.visual.lower()} | {p.visual.lower() for p in existing.personas}
        if incoming.visual.lower().strip() not in existing_visuals:
            existing.personas.append(Persona(
                outfit=incoming.visual.split(",")[0][:60] if "," in incoming.visual else "",
                visual=incoming.visual,
                context=f"alt appearance from {incoming.name}",
            ))

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

