"""Content-addressed cache for raw LLM responses.

Keys derive from the exact request (model, system prompt, text parts, file
*contents*, token budget, temperature), so rerunning a pipeline — even into
a fresh run directory — never re-pays for an answer already known. Entries
live in ``<output_root>/.v2g_cache/``: shared across runs, outside any
single run directory, purged with ``projects/`` or via V2G_LLM_CACHE=0.

Layering: the cache knows nothing about JSON or GameDesign. Callers decide
whether cached text is still valid and when to refresh — only a *cached*
response that proves bad is refetched; a fresh malformed response is never
blindly retried.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from v2g.config import settings

log = logging.getLogger(__name__)


def _cache_dir() -> Path:
    return settings.output_root / ".v2g_cache"


def _file_digest(path: Path) -> str:
    """Content hash — stable across run directories (paths differ per run)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_for(
    *,
    model: str,
    system: str,
    parts: list[str | Path],
    max_tokens: int,
    temperature: float,
) -> str:
    """Deterministic key for one exact chat request."""

    def part_key(part: str | Path) -> list[str]:
        if isinstance(part, Path):
            return ["file", _file_digest(part)]
        return ["text", part]

    material = json.dumps(
        {
            "model": model,
            "system": system,
            "parts": [part_key(p) for p in parts],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def get(key: str) -> str | None:
    """Cached raw response text, or None on miss / disabled / corrupt entry."""
    if not settings.llm_cache:
        return None
    try:
        text = json.loads((_cache_dir() / f"{key}.json").read_text(encoding="utf-8"))["text"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(text, str) or not text:
        return None
    log.debug("cache hit %.12s", key)
    return text


def put(key: str, text: str) -> None:
    if not settings.llm_cache or not text:
        return
    directory = _cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{key}.json").write_text(
        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"
    )
    log.debug("cache store %.12s (%d chars)", key, len(text))


def invalidate(key: str) -> None:
    """Drop one key — used when a cached response fails the caller's validation."""
    (_cache_dir() / f"{key}.json").unlink(missing_ok=True)
    log.debug("cache drop %.12s", key)
