"""Thin wrapper around the OpenAI-compatible chat completions API."""

import base64
from pathlib import Path

from openai import OpenAI

from v2g.config import settings


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def _image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def chat(
    system: str,
    user_parts: list[str | Path],
    *,
    max_tokens: int = 8192,
    temperature: float = 0.4,
) -> str:
    """Send a multimodal chat request. String parts are text; Path parts are images."""
    content: list[dict] = []
    for part in user_parts:
        if isinstance(part, Path):
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(part), "detail": "low"},
            })
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
