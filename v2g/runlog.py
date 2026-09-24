"""Per-run project directory and file logging.

Every pipeline run starts by creating a fresh project directory that owns
everything the run produces: the generated game, intermediate work files
(downloads, frames, segments), raw LLM responses, and ``v2g.log``.
Deleting that one directory deletes the whole run.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from v2g.config import settings

log = logging.getLogger(__name__)

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Console-visible non-problem updates: recovery outcomes, run milestones.
NOTICE = 25
logging.addLevelName(NOTICE, "NOTICE")

_run_dir: Path | None = None
_handlers: list[logging.Handler] = []


def run_dir() -> Path | None:
    """The active run directory, or None outside a run."""
    return _run_dir


def start_run(source: str, output_dir: Path | None = None) -> Path:
    """Create this run's project directory and route all logging into it.

    Args:
        source: Original source string (file path or URL) — used to name
            the default run directory.
        output_dir: Explicit override; reused in place if it already exists.

    Returns:
        The run directory (which is also the generated project root).
    """
    global _run_dir
    if output_dir is None:
        directory = _fresh_dir(source)
        directory.mkdir(parents=True)
    else:
        directory = Path(output_dir)
        reused = directory.exists() and any(directory.iterdir())
        directory.mkdir(parents=True, exist_ok=True)
    _run_dir = directory
    _configure_logging(directory)
    if output_dir is not None and reused:
        log.warning(
            "Output directory %s already exists — run artifacts will be mixed into it", directory
        )
    log.info("Run started: source=%s", source)
    log.info(
        "Run config: model=%s base_url=%s max_duration=%ds video_max_mb=%d api_key=%s",
        settings.llm_model,
        settings.llm_base_url,
        settings.max_duration,
        settings.video_max_mb,
        "set" if settings.llm_api_key else "MISSING",
    )
    return directory


def reset() -> None:
    """Detach run logging and forget the run directory (tests / in-process reruns)."""
    global _run_dir
    for handler in _handlers:
        logging.getLogger().removeHandler(handler)
        handler.close()
    _handlers.clear()
    _run_dir = None


def work_dir() -> Path:
    """Scratch directory for downloads/frames/segments, inside the run directory."""
    if _run_dir is None:
        raise RuntimeError("No active run — call start_run() first")
    directory = _run_dir / "work"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def llm_dump(tag: str, text: str) -> Path | None:
    """Save one raw LLM response under ``<run>/llm/`` and return its path.

    Returns None when no run is active (library use outside the pipeline).
    """
    if _run_dir is None:
        return None
    directory = _run_dir / "llm"
    directory.mkdir(parents=True, exist_ok=True)
    n = 1
    while (directory / f"{n:03d}_{tag}.txt").exists():
        n += 1
    path = directory / f"{n:03d}_{tag}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _fresh_dir(source: str) -> Path:
    """Default run directory: projects/<timestamp>_<source-slug>/ — new every run."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 — local stamp for a directory name
    candidate = settings.output_root / f"{stamp}_{_slug(source)}"
    n = 2
    while candidate.exists():
        candidate = settings.output_root / f"{stamp}_{_slug(source)}_{n}"
        n += 1
    return candidate


def _slug(source: str) -> str:
    """Filesystem-safe slug of the source name (file stem or URL last segment)."""
    name = source.split("?")[0].split("#")[0].rstrip("/").rsplit("/", 1)[-1]
    stem = Path(name).stem or name
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
    return slug[:60] or "run"


def _configure_logging(directory: Path) -> None:
    """Point root logging at <run>/v2g.log (file: everything; console: NOTICE+)."""
    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()
    _handlers.clear()

    file_handler = logging.FileHandler(directory / "v2g.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FMT))

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(NOTICE)
    stream_handler.setFormatter(logging.Formatter(_FMT))

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    _handlers.extend([file_handler, stream_handler])

    root.setLevel(logging.INFO)
    logging.getLogger("v2g").setLevel(logging.DEBUG)
    # openai's own INFO lines duplicate what httpx already reports.
    logging.getLogger("openai").setLevel(logging.WARNING)
