"""Voice-over synthesis (TTS) for the VN's narration and dialogue.

One backend: the **trunk endpoint** — the model from ``V2G_TTS_MODEL`` with
the LLM's key (that model setting is also the on/off switch; unset = the VN
stays text-only). Two protocols, both on that endpoint:

- ``POST <base>/audio/speech`` — OpenAI voice clients (preset voice ids),
  tried first for preset-voice models;
- the documented chat protocol (:mod:`v2g.audio_api` ``chat_speech``) — the
  text to speak rides the ``assistant`` turn, an optional style/voice
  description the ``user`` turn. This is the *only* path for voice-design
  models (``mimo-v2.5-tts-voicedesign``), whose per-speaker voice is a text
  description of its own and which reject ``audio.voice``; it is also the
  fallback when the endpoint has no ``/audio/speech`` route. Which vendor
  actually serves the model is the user's routing concern.

Voice assignment has two modes:

- **preset voices** — ``V2G_TTS_VOICES`` ids (or the provider's list);
  narration keeps the first, characters rotate the rest;
- **voice design** — one cached LLM call derives a 1–2 sentence voice
  description per speaker from the finished design (narration + every
  character, keyed exactly like the story's speaker keys), so each role and
  the narrator get a distinct, personality-fitting voice.

To support another backend, subclass :class:`TTSProvider` and wire it into
``build_synth`` — the rest of the pipeline only ever sees a ``Synthesizer``.

Contract: synthesis NEVER fails a run. A provider-wide problem (no key,
dead endpoint) warns once and disables voice-over for the rest of the run; a
per-line problem drops that line's audio only — the story text is untouched.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
from pathlib import Path

from v2g import audio_api, cache, runlog
from v2g.config import settings
from v2g.llm import jsonfix
from v2g.llm.analyzer import GameDesign, safe_name, speaker_key
from v2g.llm.client import ChatResult, chat

log = logging.getLogger(__name__)

_MAX_CHARS = 4000  # /audio/speech input cap — guard, lines are far below
_MAX_VOICE_DESC = 300  # characters — a voice is 1–2 sentences, not a script
_DEFAULT_VOICE_DESIGN = "一位中年旁白，嗓音沉稳自然，语速中等，吐字清晰。"

_VOICE_DESIGN_SYSTEM = """\
You are the voice director of a visual novel. You receive one JSON object with:
- `speaker_keys`: the exact speaker keys of the story; the FIRST one is always
  "" — the narrator (旁白, title card and narration lines);
- `design`: the game design JSON (`characters` has role/personality for each).

Return ONLY a JSON object `{"voices": {key: description}}` with one entry for
EVERY speaker key, describing a voice for a voice-design text-to-speech model
that builds a brand-new voice from your text.

Each description is ONE or TWO plain sentences (10–60 Chinese characters):
- who the voice IS: age band + gender (+ an optional style anchor like
  纪录片旁白风格 / 播音员风格 / 电台DJ风格),
- how it sounds: timbre, resonance, articulation (用具体动词或比喻，不堆形容词),
- pace/rhythm and the default emotional baseline.

Hard rules:
- Describe the VOICE only — never a scene, an action, or what will be said.
- No real actor names and no names of existing IP characters.
- Mandarin (普通话) unless the design clearly demands a dialect.
- Every voice must be clearly DISTINCT from the others and fit the
  character's role and personality in `design`; the narrator must be a calm,
  clear storyteller who never collides with the cast.
- Output raw JSON only: no markdown fences, no commentary.
"""


class TTSUnavailable(Exception):
    """Provider-level failure: voice-over is disabled for the rest of the run."""


class TTSProvider(abc.ABC):
    """One synthesis backend: fixed output format plus a default voice list."""

    name: str = ""
    ext: str = "mp3"

    @abc.abstractmethod
    def default_voices(self) -> list[str]:
        """Ordered voices; the first one speaks narration and title cards."""

    @abc.abstractmethod
    def synthesize(self, text: str, voice: str) -> bytes | None:
        """Audio bytes for *text* in *voice*; None when this line has no audio.

        *voice* is a preset voice id — or, for a voice-design model, the
        speaker's voice **description**, which the provider sends as the
        chat ``user`` turn instead of ``audio.voice``.
        """


class OpenAITTS(TTSProvider):
    """``/audio/speech`` on the trunk endpoint (OpenAI-compatible)."""

    name = "api"
    ext = "mp3"

    def default_voices(self) -> list[str]:
        return ["nova", "alloy", "shimmer", "echo", "onyx", "fable"]

    def synthesize(self, text: str, voice: str) -> bytes | None:
        import openai

        from v2g.llm.client import _get_client

        model = settings.tts_model
        if audio_api.is_voice_design(model):
            # Voice design has no voice id to pass: the description *is* the
            # voice, so the OpenAI route (model+voice+input) cannot express it.
            # Straight to the documented chat protocol, description as `user`.
            try:
                return audio_api.chat_speech(model, text, style=voice)
            except audio_api.AudioAPIError as e:
                raise TTSUnavailable(f"chat-TTS via {settings.llm_base_url}: {e}") from e
        try:
            resp = _get_client().audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3",
            )
        except openai.OpenAIError as e:  # status / connection / auth — endpoint-wide
            if getattr(e, "status_code", None) != 404:
                raise TTSUnavailable(f"audio/speech via {settings.llm_base_url}: {e}") from e
            # Endpoint has no /audio/speech route (some gateways serve TTS only
            # through chat completions) — fall back to the documented chat
            # protocol; AudioAPIError detail (voice lists, message rules) shows up.
            try:
                return audio_api.chat_speech(model, text, voice=voice)
            except audio_api.AudioAPIError as e2:
                raise TTSUnavailable(f"chat-TTS via {settings.llm_base_url}: {e2}") from e2
        return resp.content or None


class Synthesizer:
    """Per-run voice-over builder: voice assignment plus cached synthesis.

    Two assignment modes:

    - **preset voices** (``voice_designs is None``): speakers get a voice id
      in first-appearance order — narration keeps the first of the list, so a
      run's cast sounds stable.
    - **voice design**: ``voice_designs`` maps story speaker keys
      (:func:`v2g.llm.analyzer.speaker_key`) to text descriptions; every
      speaker — narrator included — gets its own designed voice, and an
      unknown speaker borrows the narrator's.
    """

    def __init__(
        self,
        provider: TTSProvider,
        voice_dir: Path,
        *,
        voice_designs: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.voice_dir = voice_dir
        self.voice_designs = voice_designs
        configured = [v.strip() for v in settings.tts_voices.split(",") if v.strip()]
        self.voices = configured or provider.default_voices()
        self.enabled = True
        self.voiced = 0
        self._speaker_voice: dict[str, str] = {}
        self._warned_long = False

    def voice_for(self, speaker_key: str) -> str:
        """The voice spec for *speaker_key*: an id, or a voice description.

        Voice-design mode returns the speaker's description (narration's for
        anyone undesigned). Preset mode: narration (``""``) gets voice[0];
        other speakers rotate the rest.
        """
        if self.voice_designs is not None:
            return (
                self.voice_designs.get(speaker_key)
                or self.voice_designs.get("")
                or _DEFAULT_VOICE_DESIGN
            )
        if not self.voices:
            return ""
        if speaker_key == "":
            return self.voices[0]
        if speaker_key not in self._speaker_voice:
            others = len(self._speaker_voice)
            idx = 1 + others % (len(self.voices) - 1) if len(self.voices) > 1 else 0
            self._speaker_voice[speaker_key] = self.voices[idx]
        return self._speaker_voice[speaker_key]

    def speak(self, text: str, speaker_key: str) -> str | None:
        """``res://`` path of this line's clip inside the project, or None.

        Never raises: every failure mode either skips the line or disables
        voice-over for the rest of the run (warned exactly once).
        """
        text = (text or "").strip()
        if not text or not self.enabled:
            return None
        if len(text) > _MAX_CHARS:
            if not self._warned_long:
                log.warning("Voice-over: skipping lines over %d chars", _MAX_CHARS)
                self._warned_long = True
            return None
        voice = self.voice_for(speaker_key)
        if not voice:
            return None
        material = {
            "provider": self.provider.name,
            "model": settings.tts_model,
            "voice": voice,
            "text": text,
        }
        key = hashlib.sha256(
            json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        try:
            data = None
            found = cache.audio_find("tts", material, self.provider.ext)
            if found is not None:
                data = found.read_bytes()
            else:
                data = self.provider.synthesize(text, voice)
                if not data:
                    return None
                cache.audio_put("tts", material, data, self.provider.ext)
        except TTSUnavailable as e:
            self.enabled = False
            log.warning("Voice-over disabled for this run: %s", e)
            return None
        except Exception as e:  # noqa: BLE001 — one bad line ships without audio
            log.warning("Voice-over failed for %r: %s", text[:60], e)
            return None

        target = self.voice_dir / f"{key}.{self.provider.ext}"
        try:
            if not target.is_file() or target.stat().st_size == 0:
                self.voice_dir.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        except OSError as e:
            log.warning("Voice-over: cannot write %s: %s", target, e)
            return None
        self.voiced += 1
        return f"res://assets/voice/{target.name}"


def derive_voice_designs(design: GameDesign) -> dict[str, str] | None:
    """One voice description per speaker (``""`` = narrator); None on any failure.

    The keys are exactly the story's speaker keys
    (:func:`v2g.llm.analyzer.speaker_key`), so a designed voice lands on the
    speaker it was written for. One cached LLM call — stage 3 reads the
    finished cast, the same pattern as :func:`v2g.sfx.derive_cues`.
    """
    keys = [""]
    for ds in design.dialogue_samples:
        key = speaker_key(ds, design.characters)
        if key and key not in keys:
            keys.append(key)
    user = json.dumps(
        {"speaker_keys": keys, "design": json.loads(design.model_dump_json())},
        ensure_ascii=False,
    )
    try:
        res: ChatResult = chat(_VOICE_DESIGN_SYSTEM, [user], temperature=0.3)
    except Exception as e:  # noqa: BLE001 — chat layer decides what is fatal
        log.warning("Voice design derivation failed: %s", e)
        return None
    parsed: dict[str, str] | None = None
    for cand in jsonfix.json_candidates(res.text):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("voices"), dict):
            obj = obj["voices"]
        if not isinstance(obj, dict):
            continue
        # Exact keys win; aliases (narrator spelled out, name vs safe_name)
        # only fill gaps — the story's key is authoritative.
        cleaned: dict[str, str] = {}
        aliases: dict[str, str] = {}
        for k, v in obj.items():
            if not isinstance(v, str) or not v.strip():
                continue
            desc = v.strip()[:_MAX_VOICE_DESC]
            key = str(k).strip()
            if key in keys:
                cleaned[key] = desc
            elif key.lower() in {"narrator", "旁白", "narration"} and "" not in cleaned:
                aliases[""] = desc
            elif safe_name(key) in keys:
                aliases.setdefault(safe_name(key), desc)
        for key, desc in aliases.items():
            cleaned.setdefault(key, desc)
        if cleaned:
            parsed = cleaned
            break
    if parsed is None:
        log.warning("Voice design derivation returned no usable JSON object")
        return None
    if "" not in parsed:
        parsed[""] = _DEFAULT_VOICE_DESIGN
    return parsed


def build_synth(run_dir: Path, design: GameDesign | None = None) -> Synthesizer | None:
    """The run's voice-over builder; None when ``V2G_TTS_MODEL`` is unset or unusable.

    Voice-design models (``…-voicedesign``) derive one voice description per
    speaker from *design* by one cached LLM call; a failed derivation falls
    back to a single default voice rather than to preset ids the endpoint
    would reject.
    """
    if not settings.tts_model.strip():
        return None
    if not settings.llm_api_key:
        log.warning("Voice-over off: V2G_LLM_API_KEY is unset")
        return None
    voice_designs: dict[str, str] | None = None
    if audio_api.is_voice_design(settings.tts_model):
        derived = derive_voice_designs(design) if design is not None else None
        if derived is None:
            derived = {"": _DEFAULT_VOICE_DESIGN}
            log.log(
                runlog.NOTICE,
                "Voice design: derivation failed — one default voice for all speakers",
            )
        else:
            log.log(
                runlog.NOTICE,
                "Voice design: %d speaker voice(s) via %s",
                len(derived),
                settings.tts_model,
            )
        voice_designs = derived
    return Synthesizer(OpenAITTS(), run_dir / "assets" / "voice", voice_designs=voice_designs)
