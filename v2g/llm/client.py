"""Thin wrapper around the OpenAI-compatible chat completions API."""

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from v2g.config import settings


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


def chat(
    system: str,
    user_parts: list[str | Path],
    *,
    max_tokens: int = 8192,
    temperature: float = 0.4,
) -> str:
    """Send a multimodal chat request.

    - str parts → text content
    - Path parts → image or video content (auto-detected by extension)
    """
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
    return resp.choices[0].message.content or ""
