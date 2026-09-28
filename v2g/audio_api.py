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


def chat_audio(model: str, prompt: str) -> bytes:
    """One clip for *prompt* from the trunk endpoint, OpenAI format → mp3 bytes.

    Raises :class:`AudioAPIError` for every failure mode (bad transport,
    non-audio answer, unparsable payload) — never a raw urllib exception.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "modalities": ["text", "audio"],
        "audio": {"format": "mp3", "voice": _AUDIO_VOICE},
    }
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
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise AudioAPIError(f"chat/completions → HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise AudioAPIError(f"endpoint unreachable: {e.reason}") from e
    try:
        body = json.loads(raw)
    except ValueError as e:
        raise AudioAPIError("non-JSON response from the chat endpoint") from e

    data, mime = _audio_payload(body)
    return _to_mp3(data, mime)
