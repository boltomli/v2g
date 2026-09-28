"""Shared client for OpenAI-format chat-completions audio.

The **protocol is OpenAI's** (verified against the official openai-python
SDK): ``POST <V2G_LLM_BASE_URL>/chat/completions`` with ``model``,
``messages``, ``modalities: ["text", "audio"]`` and
``audio: {"format": "mp3", "voice": ...}``; the audio comes back as
``choices[0].message.audio.data`` — base64 bytes. Music/SFX specific
guidance (length, instrumental) travels in the prompt text, so the body
stays a valid OpenAI request.

Response parsing also tolerates the OpenRouter/ACE-Step ``audio_url`` shapes
(audio_url as object, list entries, data URI or plain URL) — a superset
contract, OpenAI format first.

Which backend actually serves a given ``model`` is the user's routing
concern; an endpoint that answers with plain text simply yields no audio and
the caller skips the layer.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import urllib.error
import urllib.request

from v2g.config import settings

log = logging.getLogger(__name__)

_CHAT_TIMEOUT = 300.0  # audio generation is synchronous on this endpoint
_AUDIO_VOICE = "alloy"  # required by the OpenAI `audio` param; irrelevant to music
_NEUTRAL_DESIGN = "自然中性的嗓音，吐字清晰，语速中等。"  # voice-design needs SOME description


class AudioAPIError(Exception):
    """The chat-audio endpoint could not produce audio (transport / shape)."""


def _resolve_url(url: str) -> tuple[bytes, str]:
    """Fetch a ``data:`` URI or http(s) URL → (bytes, mime)."""
    if url.startswith("data:"):
        header, _, b64 = url.partition(",")
        mime = header[len("data:") :].partition(";")[0]
        try:
            return base64.b64decode(b64), mime
        except (ValueError, TypeError) as e:
            raise AudioAPIError(f"bad data URI in response: {e}") from e
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=120.0) as resp:
            return resp.read(), ""
    except (urllib.error.URLError, OSError) as e:
        raise AudioAPIError(f"audio download failed: {e}") from e


def _audio_payload(body: object) -> tuple[bytes, str]:
    """Extract (audio bytes, mime) from a chat-completions response.

    OpenAI format first (``message.audio.data`` = base64), then the
    OpenRouter/ACE-Step ``audio_url`` variants.
    """
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        raise AudioAPIError(f"endpoint error: {str(body['error'])[:200]}")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise AudioAPIError("response carries no choices")
    message = choices[0].get("message") or {}

    audio = message.get("audio")
    # OpenAI: {"id": ..., "data": "<base64>", "transcript": ...}
    if isinstance(audio, dict):
        data = audio.get("data")
        if isinstance(data, str) and data:
            try:
                return base64.b64decode(data), ""
            except (ValueError, TypeError) as e:
                raise AudioAPIError(f"bad base64 in message.audio.data: {e}") from e

    # OpenRouter / ACE-Step: list entries {"type": "audio_url", "audio_url": {"url": ...}}
    candidates: list[object] = []
    if isinstance(audio, list):
        for entry in audio:
            if isinstance(entry, dict):
                candidates.append(entry.get("audio_url"))
    candidates.append(message.get("audio_url"))
    for item in candidates:
        if isinstance(item, dict):
            item = item.get("url")
        if isinstance(item, str) and item:
            return _resolve_url(item)
    raise AudioAPIError("no audio in response — the model is not an audio model")


def _looks_like_mp3(data: bytes) -> bool:
    if data[:3] == b"ID3":
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _to_mp3(data: bytes, mime: str) -> bytes:
    """Normalize to mp3; pass mp3 through untouched."""
    if mime.startswith("audio/mpeg") or _looks_like_mp3(data):
        return data
    # ffmpeg (a hard project prerequisite) sniffs the input and re-encodes.
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "mp3", "-q:a", "4", "pipe:1"],
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace")[:200]
        raise AudioAPIError(f"ffmpeg could not normalize audio to mp3: {err}")
    return proc.stdout


def _post_chat(payload: dict) -> tuple[int, bytes]:
    """POST the chat payload → (status, body); transport errors raise."""
    req = urllib.request.Request(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    if settings.llm_api_key:
        req.add_header("Authorization", f"Bearer {settings.llm_api_key}")
    try:
        with urllib.request.urlopen(req, timeout=_CHAT_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise AudioAPIError(f"endpoint unreachable: {e.reason}") from e


def is_voice_design(model: str) -> bool:
    """True for voice-design models (``mimo-v2.5-tts-voicedesign``).

    They build the voice from a natural-language description carried in the
    ``user`` turn and reject a preset ``audio.voice`` id outright
    (``audio.voice is not supported for voice design model``).
    """
    return "voicedesign" in model.lower().replace("-", "").replace("_", "")


def _finish_chat(status: int, raw: bytes) -> bytes:
    """Shared response tail: HTTP status → JSON → audio → mp3 bytes."""
    if status != 200:
        detail = raw[:300].decode("utf-8", errors="replace").replace("\n", " ")
        raise AudioAPIError(f"chat/completions → HTTP {status}: {detail}")
    try:
        body = json.loads(raw)
    except ValueError as e:
        raise AudioAPIError("non-JSON response from the chat endpoint") from e
    data, mime = _audio_payload(body)
    return _to_mp3(data, mime)


def chat_audio(model: str, prompt: str, *, voice: str | None = None) -> bytes:
    """One instruction clip (BGM) for *prompt* from the trunk endpoint → mp3 bytes.

    The prompt travels as a ``user`` turn — it is a direction, not spoken
    text. Voice-over and sound effects use :func:`chat_speech` instead,
    which follows the documented text-to-speech turn shape. ``voice``
    defaults to the first ``V2G_TTS_VOICES`` entry (gateways name their own
    voices), else ``alloy``; some gateways reject user-only conversations
    ("messages must contain an assistant role") — such a 400 is retried
    once with the prompt mirrored into an assistant turn.

    Raises :class:`AudioAPIError` for every failure mode (bad transport,
    non-audio answer, unparsable payload) — never a raw urllib exception;
    HTTP error bodies (bounded) ride the message so misconfiguration is
    visible in the run log.
    """
    configured = [v.strip() for v in settings.tts_voices.split(",") if v.strip()]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "modalities": ["text", "audio"],
        "audio": {
            "format": "mp3",
            "voice": voice or (configured[0] if configured else _AUDIO_VOICE),
        },
    }
    status, raw = _post_chat(payload)
    if status == 400 and b"assistant" in raw:
        payload["messages"] = payload["messages"] + [{"role": "assistant", "content": prompt}]
        status, raw = _post_chat(payload)
    return _finish_chat(status, raw)


def chat_speech(
    model: str,
    text: str,
    *,
    voice: str | None = None,
    style: str | None = None,
) -> bytes:
    """One spoken clip from the trunk endpoint → mp3 bytes, official TTS protocol.

    Turn shape follows the vendor's documented format (verified live): the
    ``assistant`` turn carries *text* — the line to speak — and the optional
    ``user`` turn carries *style*: a voice description for a voice-design
    model, a tone/direction note for a preset-voice model. Preset-voice
    models speak with just the ``assistant`` turn; voice-design models
    *require* the ``user`` turn (an empty one is a 400), so a neutral
    description stands in when the caller passes no ``style``.

    ``voice`` is a preset voice id for ``audio.voice`` and is **omitted for
    voice-design models**, which take the voice from ``style`` and 400 on an
    id (see :func:`is_voice_design`).

    Raises :class:`AudioAPIError` for every failure mode (bad transport,
    non-audio answer, unparsable payload) — never a raw urllib exception;
    HTTP error bodies (bounded) ride the message so misconfiguration is
    visible in the run log.
    """
    designed = is_voice_design(model)
    style = (style or "").strip()
    if designed and not style:
        style = _NEUTRAL_DESIGN
    messages: list[dict] = []
    if style:
        messages.append({"role": "user", "content": style})
    messages.append({"role": "assistant", "content": text})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "modalities": ["text", "audio"],
        "audio": {"format": "mp3"},
    }
    if not designed:
        configured = [v.strip() for v in settings.tts_voices.split(",") if v.strip()]
        payload["audio"]["voice"] = voice or (configured[0] if configured else _AUDIO_VOICE)
    status, raw = _post_chat(payload)
    return _finish_chat(status, raw)
