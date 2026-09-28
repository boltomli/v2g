"""Sound effects: fixed VN event sounds plus per-line diegetic cues.

Two layers, one endpoint: clips come from the trunk chat-audio endpoint with
``V2G_SFX_MODEL``, **falling back to the voice-over model**
(``V2G_TTS_MODEL``) — both unset = SFX off. Since the serving model is a
TEXT-TO-SPEECH model, cues are Chinese onomatopoeia a voice can perform
(咚咚咚 / 哗啦啦 / 叮——), never descriptions: whatever we send gets read
aloud. Which backend actually serves the model is the user's routing concern
(see :mod:`v2g.audio_api`).

The per-line cues are derived from the finished design by ONE cached LLM
call — stage 3 reads stages 1/2, so no schema or analyzer prompt changes.

Contract, like every audio layer here: never fails a run. Endpoint problems
warn once and disable SFX; a bad clip drops only that clip.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from v2g import audio_api, cache, runlog
from v2g.config import settings
from v2g.llm import jsonfix
from v2g.llm.analyzer import GameDesign
from v2g.llm.client import ChatResult, chat

log = logging.getLogger(__name__)

_MAX_CUES = 8  # cap on non-empty content cues per script
_EVENT_PROMPTS = {
    # Vocal (TTS-read) sound words: the gateway's TTS model performs them.
    "select": "叮——",
    "transition": "嗖——",
}

_SFX_SYSTEM = """\
You are the sound designer for a visual novel. You receive a game design JSON;
`dialogue_samples` is the full script in play order.

The sound effects are spoken by a TEXT-TO-SPEECH model, so cues must be
short Chinese onomatopoeia (拟声词) a voice can perform — never descriptions.

Return ONLY a JSON array of strings, EXACTLY as long as `dialogue_samples`,
one entry per sample in order. Each entry is either:
- "" (no sound — the default), or
- one onomatopoeia phrase of 2–8 characters, e.g. 咚咚咚 / 哗啦啦 /
  轰隆隆 / 叮咚 / 咔嚓 / 汪汪.

Rules:
- At most 8 non-empty entries overall; prefer silence.
- Pure sound words only: no descriptions, no sentences, no music notation,
  no explanation — anything written gets read aloud.
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


# ── Per-run builder ─────────────────────────────────────────────────────────


class Sfx:
    """Per-run sound-effect builder: event clips plus aligned step cues.

    Clips land in ``assets/sfx/`` and are content-addressed in
    ``.v2g_cache/sfx/`` (gated by ``V2G_MEDIA_CACHE``), so reruns are free.
    """

    def __init__(self, sfx_dir: Path) -> None:
        self.sfx_dir = sfx_dir
        self.model = settings.sfx_model.strip() or settings.tts_model.strip()
        self.cues: list[str] = []  # aligned to design.dialogue_samples
        self.enabled = True
        self.generated = 0
        self._by_text: dict[str, str] = {}

    def _clip(self, text: str, stem: str) -> str | None:
        """Synthesize one clip into ``assets/sfx/<stem>.mp3`` → res path or None.

        The text reaches the TTS model verbatim — no prompt decoration and no
        duration hints: the model READS whatever it is given, so anything
        appended would be spoken aloud.
        """
        text = text.strip()
        if not text or not self.enabled or not self.model:
            return None
        if text in self._by_text:
            return self._by_text[text]
        material = {
            "model": self.model,
            "text": text,
            "format": "mp3",
        }
        try:
            found = cache.audio_find("sfx", material, "mp3")
            if found is not None:
                data = found.read_bytes()
            else:
                data = audio_api.chat_audio(self.model, text)
                if not data:
                    log.debug("Sound effect endpoint returned no audio for %r", text[:60])
                    return None
                cache.audio_put("sfx", material, data, "mp3")
        except audio_api.AudioAPIError as e:
            self.enabled = False
            log.warning("Sound effects disabled for this run: %s", e)
            return None
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
    """The run's SFX builder; None when no model backs it. Never raises."""
    if not (settings.sfx_model.strip() or settings.tts_model.strip()):
        return None
    sfx = Sfx(run_dir / "assets" / "sfx")
    cues = derive_cues(design)
    if cues is None:
        log.log(runlog.NOTICE, "SFX: cue derivation failed — event sounds only")
    else:
        sfx.cues = cues
        log.log(
            runlog.NOTICE,
            "SFX: %d content cue(s) via %s (cap %d)",
            sum(1 for c in cues if c.strip()),
            sfx.model,
            _MAX_CUES,
        )
    return sfx
