"""Content-addressed caches for expensive, reusable pipeline artifacts.

Two layers, both under ``<output_root>/.v2g_cache/`` — shared across runs,
outside any single run directory, purged with ``projects/``:

- **LLM responses** (``get``/``put``): keys derive from the exact request
  (model, system prompt, text parts, file *contents*, token budget,
  temperature), so rerunning a pipeline — even into a fresh run directory —
  never re-pays for an answer already known. Off switch: ``V2G_LLM_CACHE=0``.
- **Media** (``media_entry``/``media_find``/``media_stage``): downloaded
  videos and transcode outputs, keyed by their inputs (source identity plus
  the settings that shape them), so reruns skip downloads and re-encodes.
  Off switch: ``V2G_MEDIA_CACHE=0`` (artifacts then land in the run's work
  directory instead).

Layering: the cache knows nothing about JSON or GameDesign. Callers decide
whether cached text is still valid and when to refresh — only a *cached*
response that proves bad is refetched; a fresh malformed response is never
blindly retried.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    log.debug("cache drop %s", key)


# ── Media artifacts ──────────────────────────────────────────────────────────


def media_entry(kind: str, material: dict, fallback: Path) -> Path:
    """Directory for one cached media artifact: ``.v2g_cache/media/<kind>-<sha>``.

    *material* (JSON-serializable) determines the artifact's content — source
    identity plus the settings that shape it. With the media cache disabled the
    entry moves under *fallback* (the run's work directory) and the rest of the
    flow stays identical.
    """
    key = hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    root = _cache_dir() / "media" if settings.media_cache else fallback
    return root / f"{kind}-{key}"


def media_find(entry: Path, probe: Callable[[Path], Path | None]) -> Path | None:
    """Cached artifact inside *entry* when *probe* finds one, else None.

    A present-but-unusable entry (partial or tampered) is dropped so the caller
    rebuilds it instead of trusting a broken artifact.
    """
    if not entry.is_dir():
        return None
    try:
        found = probe(entry)
    except OSError:
        found = None
    if found is None:
        log.warning("media cache entry %s unusable — dropping it", entry.name)
        shutil.rmtree(entry, ignore_errors=True)
        return None
    log.debug("media hit %s → %s", entry.name, found)
    return found


@contextmanager
def media_stage(entry: Path) -> Iterator[Path]:
    """Yield a scratch directory to build *entry* in.

    On success the staging directory is atomically renamed onto *entry*, so an
    entry either exists complete or not at all; on failure staging is removed
    and nothing is cached. If *entry* appeared meanwhile (concurrent run), the
    staging copy is discarded — first build wins.
    """
    stage = entry.parent / f"{entry.name}.staging"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        yield stage
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if entry.exists():
        shutil.rmtree(stage, ignore_errors=True)
    else:
        stage.rename(entry)
