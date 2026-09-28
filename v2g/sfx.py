"""Sound effects: fixed VN event sounds plus per-line diegetic cues.

Two layers, one generator: clips are synthesized **locally** — for each cue
the chat model writes a self-contained Python synthesis script
(:mod:`v2g.codegen`, same machinery as the BGM ``llm`` provider), the script
runs isolated and its WAV becomes the clip. No audio endpoint is involved,
so nothing can be read aloud: the TTS route (whatever it is asked to say,
it *speaks*) proved that a description-only prompt still yields a voice.

Backend switch: ``V2G_SFX_PROVIDER=llm`` enables the code-writing model,
unset (or any unknown value) = SFX off. Which model answers the chat call is
``V2G_LLM_MODEL`` — the same one that writes the music scripts.

The per-line cues are derived from the finished design by ONE cached LLM
call — stage 3 reads stages 1/2, so no schema or analyzer prompt changes.

Contract, like every audio layer here: never fails a run. Chat problems
warn once and disable SFX; a bad script drops only that clip.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from v2g import cache, codegen, runlog
from v2g.audio_api import AudioAPIError
from v2g.config import settings
from v2g.llm import jsonfix
from v2g.llm.analyzer import GameDesign
from v2g.llm.client import ChatResult, chat

log = logging.getLogger(__name__)

_MAX_CUES = 8  # cap on non-empty content cues per script
_MAX_CUE_CHARS = 40  # bound on the codegen prompt — a cue is a sound, not prose
_SCRIPT_TIMEOUT = 60.0  # seconds — a ≤2 s clip renders in well under this
_EVENT_PROMPTS = {
    "select": "清脆的电子提示音，短促的 UI 点击反馈",
    "transition": "快速掠过的嗖声，场景切换的过渡音",
}

_SFX_SYSTEM = """\
You are the sound designer for a visual novel. You receive a game design JSON;
`dialogue_samples` is the full script in play order.

Each cue is rendered by a Python synthesis script the pipeline writes from
YOUR description — so each cue names what is heard: material, source, motion.
Never a word meant to be spoken, never a sentence.

Return ONLY a JSON array of strings, EXACTLY as long as `dialogue_samples`,
one entry per sample in order. Each entry is either:
- "" (no sound — the default), or
- one sound description of 4–40 Chinese characters,
  e.g. 沉闷的关门声 / 哗哗的大雨 / 急促的脚步声 / 玻璃碎裂声 /
  远处的雷声.

Rules:
- At most 8 non-empty entries overall; prefer silence.
- Describe the sound itself (source + texture), not its cause and not a
  feeling: no sentences, no dialogue, no music directions.
- The sound must follow from the line, its `context` or the atmosphere, and
  never contradict the story.
- Output raw JSON only: no markdown fences, no commentary.
"""


# ── Cue derivation (one cached LLM call) ────────────────────────────────────


def derive_cues(design: GameDesign) -> list[str] | None:
    """One cue string per dialogue sample ("" = silence); None on any failure.

    Length mismatches and malformed answers degrade to a padded/truncated
    list or None — never an exception.
    """
    n = len(design.dialogue_samples)
    try:
        res: ChatResult = chat(_SFX_SYSTEM, [design.model_dump_json()], temperature=0.2)
    except Exception as e:  # noqa: BLE001 — chat layer decides what is fatal
        log.warning("SFX cue derivation failed: %s", e)
        return None
    cues: list[str] | None = None
    for cand in jsonfix.json_candidates(res.text):
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("cues")
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            cues = parsed
            break
    if cues is None:
        log.warning("SFX cue derivation returned no usable JSON array")
        return None
    if len(cues) != n:
        log.warning("SFX cue count %d != dialogue samples %d — normalizing", len(cues), n)
        cues = (cues + [""] * n)[:n]
    non_empty = sum(1 for c in cues if c.strip())
    if non_empty > _MAX_CUES:
        keep = 0
        trimmed: list[str] = []
        for c in cues:
            if c.strip() and keep >= _MAX_CUES:
                trimmed.append("")
            else:
                trimmed.append(c)
                if c.strip():
                    keep += 1
        cues = trimmed
        log.warning("SFX cues capped at %d non-empty entries", _MAX_CUES)
    return cues


# ── Local synthesis (the chat model writes the script) ──────────────────────


def _clip_system() -> str:
    """Spec for the text model: one self-contained script → one sound → WAV."""
    return f"""\
You are an expert Python audio programmer. Write ONE self-contained Python
script that renders ONE sound effect for a visual novel and writes it to disk.

Contract:
- The script receives the OUTPUT WAV PATH as sys.argv[1] and writes 16-bit
  PCM mono audio at 44100 Hz: between 0.5 and 2.0 seconds — the natural
  length of THIS sound (a click is short, an ambience fills the window).
- The sound to render is described in the user message. Render ONLY that
  sound: material, source, motion. Absolutely no speech, no voices, no
  singing, no speech-like syllables — only the described sound, decaying
  naturally into silence (no click at either end).
- Runtime: Python {sys.version_info.major}.{sys.version_info.minor}; {codegen.libs_note()}.
- Deterministic: fixed random seed. No network, no reading of any file, no
  input(), no GUI, no packages beyond what is stated above.
- Self-check before answering: re-read the script for syntax errors, shape
  mismatches (every slice/mask must match its array's length exactly) and
  wrong output paths — it must run without editing.

Output ONLY the raw Python source — no markdown fences, no commentary.
"""


# ── Per-run builder ─────────────────────────────────────────────────────────


class Sfx:
    """Per-run sound-effect builder: event clips plus aligned step cues.

    Clips land in ``assets/sfx/`` and are content-addressed in
    ``.v2g_cache/sfx/`` (gated by ``V2G_MEDIA_CACHE``), so reruns are free;
    the generated scripts stay in ``work/sfx_llm/`` for inspection.
    """

    def __init__(self, run_dir: Path) -> None:
        self.sfx_dir = run_dir / "assets" / "sfx"
        self.work = run_dir / "work" / "sfx_llm"
        self.cues: list[str] = []  # aligned to design.dialogue_samples
        self.enabled = True
        self.generated = 0
        self._by_text: dict[str, str] = {}

    def _synthesize(self, text: str, stem: str) -> bytes | None:
        """One clip: chat writes a script → run it → MP3 bytes, or None.

        A chat failure is endpoint-wide — it disables SFX for the run
        (warned exactly once). A bad script drops only this clip — but a
        bad script that came from the *response cache* would fail every
        rerun, so it is invalidated and refetched once (same policy as
        :func:`v2g.llm.analyzer._request_design`).
        """
        system = _clip_system()
        res: ChatResult | None = None
        for attempt in range(2):
            try:
                res = chat(system, [text], temperature=0.4, refresh=attempt > 0)
            except Exception as e:  # noqa: BLE001 — transport/auth = layer down
                self.enabled = False
                log.warning("Sound effects disabled for this run: %s", e)
                return None
            code = codegen.extract_code(res.text)
            if not code:
                log.warning("Sound effect %r: codegen returned no code", text[:60])
                return None
            try:
                return codegen.render(code, self.work, stem=stem, timeout=_SCRIPT_TIMEOUT)
            except AudioAPIError as e:
                if res.cached and attempt == 0:
                    log.info("Sound effect %r: cached script failed — refetching", text[:60])
                    continue
                log.warning("Sound effect %r failed: %s", text[:60], e)
                return None
        return None

    def _clip(self, text: str, stem: str) -> str | None:
        """Synthesize one clip into ``assets/sfx/<stem>.mp3`` → res path or None.

        The cue travels verbatim as the codegen prompt — a description of
        the sound to render, nothing appended.
        """
        text = text.strip().strip("（）()[]").strip()
        if not text or not self.enabled:
            return None
        if len(text) > _MAX_CUE_CHARS:
            text = text[:_MAX_CUE_CHARS]
        if text in self._by_text:
            return self._by_text[text]
        material = {
            "provider": settings.sfx_provider.strip(),
            "model": settings.llm_model,
            "text": text,
            "format": "mp3",
        }
        try:
            found = cache.audio_find("sfx", material, "mp3")
            if found is not None:
                data = found.read_bytes()
            else:
                data = self._synthesize(text, stem)
                if not data:
                    return None
                cache.audio_put("sfx", material, data, "mp3")
        except Exception as e:  # noqa: BLE001 — one bad clip ships without audio
            log.warning("Sound effect %r failed: %s", text[:60], e)
            return None
        target = self.sfx_dir / f"{stem}.mp3"
        try:
            if not target.is_file() or target.stat().st_size == 0:
                self.sfx_dir.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        except OSError as e:
            log.warning("Sound effect: cannot write %s: %s", target, e)
            return None
        self.generated += 1
        res_path = f"res://assets/sfx/{target.name}"
        self._by_text[text] = res_path
        return res_path

    def event(self, name: str) -> str | None:
        """Path of a fixed event clip (``select`` / ``transition``), or None."""
        prompt = _EVENT_PROMPTS.get(name)
        if prompt is None:
            return None
        return self._clip(prompt, name)

    def step_cue(self, index: int) -> str | None:
        """Path of dialogue sample *index*'s cue, or None (empty/absent/off)."""
        if not self.enabled or not 0 <= index < len(self.cues):
            return None
        cue = self.cues[index].strip()
        if not cue:
            return None
        return self._clip(cue, f"cue_{index}")


def build_sfx(run_dir: Path, design: GameDesign) -> Sfx | None:
    """The run's SFX builder; None when no provider backs it. Never raises."""
    provider = settings.sfx_provider.strip().lower()
    if not provider:
        return None
    if provider != "llm":
        log.log(
            runlog.NOTICE,
            "SFX: unknown provider %r (expected llm) — skipped",
            provider,
        )
        return None
    sfx = Sfx(run_dir)
    cues = derive_cues(design)
    if cues is None:
        log.log(runlog.NOTICE, "SFX: cue derivation failed — event sounds only")
    else:
        sfx.cues = cues
        log.log(
            runlog.NOTICE,
            "SFX: %d content cue(s) (cap %d)",
            sum(1 for c in cues if c.strip()),
            _MAX_CUES,
        )
    return sfx
