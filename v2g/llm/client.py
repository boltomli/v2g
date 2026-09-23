"""Thin wrapper around the OpenAI-compatible chat completions API."""

import base64
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from v2g import cache
from v2g.config import settings

log = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def _to_data_url(path: Path) -> str:
    """Convert a file to a base64 data URL with the correct MIME type."""
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def _file_to_content(path: Path) -> dict:
    """Build the appropriate content block for an image or video file."""
    suffix = path.suffix.lower()
    if suffix in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"):
        return {
            "type": "video_url",
            "video_url": {"url": _to_data_url(path)},
        }
    # Default to image
    return {
        "type": "image_url",
        "image_url": {"url": _to_data_url(path), "detail": "low"},
    }


@dataclass(frozen=True)
class ChatResult:
    """One raw response plus cache provenance (transport-layer output)."""

    text: str
    cached: bool
    key: str


def chat(
    system: str,
    user_parts: list[str | Path],
    *,
    max_tokens: int | None = None,
    temperature: float = 0.4,
    refresh: bool = False,
) -> ChatResult:
    """Send a multimodal chat request, served from the response cache when possible.

    - str parts → text content; Path parts → image/video (auto-detected by extension)
    - max_tokens omitted → settings.llm_max_tokens (callers with a fixed
      budget — analysis modes — pass their own value)
    - refresh=True skips the cache read (used when a cached answer proved
      invalid) and overwrites the stored entry with the fresh answer.
    - Truncated output (finish=length) is never cached.
    """
    if max_tokens is None:
        max_tokens = settings.llm_max_tokens
    key = cache.key_for(
        model=settings.llm_model,
        system=system,
        parts=user_parts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not refresh:
        hit = cache.get(key)
        if hit is not None:
            log.info("chat: cache hit key=%.12s chars=%d", key, len(hit))
            return ChatResult(hit, True, key)

    content: list[dict] = []
    for part in user_parts:
        if isinstance(part, Path):
            content.append(_file_to_content(part))
        else:
            content.append({"type": "text", "text": part})

    resp = _get_client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = resp.usage
    log.info(
        "chat: finish=%s chars=%d prompt_tokens=%s completion_tokens=%s max_tokens=%d key=%.12s",
        choice.finish_reason, len(text),
        getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
        max_tokens, key,
    )
    if choice.finish_reason == "length":
        log.warning(
            "chat output hit max_tokens=%d — response truncated; JSON may be incomplete",
            max_tokens,
        )
    else:
        cache.put(key, text)
    return ChatResult(text, False, key)
