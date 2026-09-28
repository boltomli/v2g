"""Voice-over synthesis (TTS) for the VN's narration and dialogue.

One backend: the **trunk endpoint** — the model from ``V2G_TTS_MODEL`` with
the LLM's key (that model setting is also the on/off switch; unset = the VN
stays text-only). ``POST <base>/audio/speech`` is tried first; when the
endpoint has no such route (some gateways serve TTS only through chat
completions) it falls back to the OpenAI chat-TTS protocol in
:mod:`v2g.audio_api`. Which vendor actually serves the model is the user's
routing concern.

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

from v2g import cache
from v2g.config import settings

log = logging.getLogger(__name__)

_MAX_CHARS = 4000  # /audio/speech input cap — guard, lines are far below


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
        """Audio bytes for *text* in *voice*; None when this line has no audio."""


class OpenAITTS(TTSProvider):
    """``/audio/speech`` on the trunk endpoint (OpenAI-compatible)."""

    name = "api"
    ext = "mp3"

    def default_voices(self) -> list[str]:
        return ["nova", "alloy", "shimmer", "echo", "onyx", "fable"]

    def synthesize(self, text: str, voice: str) -> bytes | None:
        import openai

        from v2g import audio_api
        from v2g.llm.client import _get_client

        try:
            resp = _get_client().audio.speech.create(
                model=settings.tts_model,
                voice=voice,
                input=text,
                response_format="mp3",
            )
        except openai.OpenAIError as e:  # status / connection / auth — endpoint-wide
            if getattr(e, "status_code", None) != 404:
                raise TTSUnavailable(f"audio/speech via {settings.llm_base_url}: {e}") from e
            # Endpoint has no /audio/speech route (some gateways serve TTS only
            # through chat completions) — fall back to the OpenAI chat-TTS
            # protocol; AudioAPIError detail (voice lists, message rules) shows up.
            try:
                return audio_api.chat_audio(settings.tts_model, text, voice=voice)
            except audio_api.AudioAPIError as e2:
                raise TTSUnavailable(f"chat-TTS via {settings.llm_base_url}: {e2}") from e2
        return resp.content or None


class Synthesizer:
    """Per-run voice-over builder: voice assignment plus cached synthesis.

    Stateful: speakers get their voice in first-appearance order (narration
    keeps the first voice of the list), so a run's cast sounds stable.
    """

    def __init__(self, provider: TTSProvider, voice_dir: Path) -> None:
        self.provider = provider
        self.voice_dir = voice_dir
        configured = [v.strip() for v in settings.tts_voices.split(",") if v.strip()]
        self.voices = configured or provider.default_voices()
        self.enabled = True
        self.voiced = 0
        self._speaker_voice: dict[str, str] = {}
        self._warned_long = False

    def voice_for(self, speaker_key: str) -> str:
        """Narrator (``""``) gets voice[0]; other speakers rotate the rest."""
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
        if not text or not self.enabled or not self.voices:
            return None
        if len(text) > _MAX_CHARS:
            if not self._warned_long:
                log.warning("Voice-over: skipping lines over %d chars", _MAX_CHARS)
                self._warned_long = True
            return None
        voice = self.voice_for(speaker_key)
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


def build_synth(run_dir: Path) -> Synthesizer | None:
    """The run's voice-over builder; None when ``V2G_TTS_MODEL`` is unset or unusable."""
    if not settings.tts_model.strip():
        return None
    if not settings.llm_api_key:
        log.warning("Voice-over off: V2G_LLM_API_KEY is unset")
        return None
    return Synthesizer(OpenAITTS(), run_dir / "assets" / "voice")
