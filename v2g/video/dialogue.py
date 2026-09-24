"""Extract dialogue transcripts from the source video.

Language contract of the pipeline: Simplified Chinese is the target language
of the generated game; every non-Chinese line shown in game MUST come from
the source video. This module is the only producer of such source-language
text. Sources, in order of preference:

1. Sidecar subtitle files next to the video (``.srt`` / ``.vtt`` / ``.ass``)
2. Embedded subtitle streams in the container

Nothing is ever machine-invented here — if no transcript exists, the pipeline
runs with an empty transcript and the analyzer must leave source lines empty.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_SIDECAR_SUFFIXES = (".srt", ".vtt", ".ass", ".ssa")

_TS_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass
class TranscriptLine:
    """One timed subtitle cue from the source video."""

    start: float  # seconds
    end: float  # seconds
    text: str


def _ts(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(content: str) -> list[TranscriptLine]:
    """Parse SRT/VTT subtitle text into timed lines. Robust to CRLF and indexes."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[TranscriptLine] = []
    i = 0
    while i < len(lines):
        m = _TS_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        start = _ts(*m.groups()[0:4])
        end = _ts(*m.groups()[4:8])
        i += 1
        buf: list[str] = []
        while i < len(lines) and lines[i].strip():
            buf.append(lines[i].strip())
            i += 1
        text = " ".join(buf).strip()
        if text:
            out.append(TranscriptLine(start, end, text))
    return out


def _ffmpeg_to_srt(src: Path) -> str | None:
    """Convert a subtitle file (any format) to SRT text via ffmpeg."""
    fd, dst_name = tempfile.mkstemp(suffix=".srt", prefix="v2g_sub_")
    os.close(fd)  # Windows: an open handle would block ffmpeg and the cleanup
    dst = Path(dst_name)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:s", "srt", str(dst)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not dst.is_file():
            return None
        return dst.read_text(encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        dst.unlink(missing_ok=True)


def _embedded_subtitles(video_path: Path) -> str | None:
    """Extract the first embedded subtitle stream of *video_path* as SRT text."""
    fd, dst_name = tempfile.mkstemp(suffix=".srt", prefix="v2g_embed_")
    os.close(fd)  # Windows: an open handle would block ffmpeg and the cleanup
    dst = Path(dst_name)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-map",
                "0:s:0",
                "-c:s",
                "srt",
                str(dst),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not dst.is_file():
            return None
        return dst.read_text(encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        dst.unlink(missing_ok=True)


def _sidecar_candidates(video_path: Path) -> list[Path]:
    """Subtitle sidecar files for *video_path*, best candidate first.

    Matches both ``clip.srt`` (hand-placed) and yt-dlp's language-tagged
    ``clip.<lang>.srt`` forms — URL downloads only ever produce the tagged
    names. Order: format priority (.srt first), exact stem before
    language-tagged, then name, so the pick is deterministic when several
    languages are present. Danmaku XML (``clip.danmaku.xml``) is not in
    ``_SIDECAR_SUFFIXES`` and is never read as dialogue.
    """
    stem = video_path.stem
    rank = {suffix: i for i, suffix in enumerate(_SIDECAR_SUFFIXES)}
    candidates = [
        p
        for p in video_path.parent.iterdir()
        if p.is_file()
        and p.suffix.lower() in rank
        and (p.stem == stem or p.stem.startswith(f"{stem}."))
    ]
    candidates.sort(key=lambda p: (rank[p.suffix.lower()], p.stem != stem, p.name))
    return candidates


def extract_dialogue(video_path: Path) -> list[TranscriptLine]:
    """Return all subtitle cues available for *video_path* (empty list if none)."""
    # 1. Sidecar subtitle files written next to the video
    for side in _sidecar_candidates(video_path):
        if side.suffix.lower() == ".srt":
            text = side.read_text(encoding="utf-8", errors="replace")
        else:
            text = _ffmpeg_to_srt(side)
        if text:
            lines = parse_srt(text)
            if lines:
                log.info("Transcript from sidecar %s: %d lines", side.name, len(lines))
                return lines

    # 2. Embedded subtitle stream
    text = _embedded_subtitles(video_path)
    if text:
        lines = parse_srt(text)
        if lines:
            log.info("Transcript from embedded subtitles: %d lines", len(lines))
            return lines

    log.info("No subtitles found for %s — source-language dialogue unavailable", video_path.name)
    return []


def format_transcript(
    lines: list[TranscriptLine],
    *,
    start: float = 0.0,
    end: float = float("inf"),
) -> str:
    """Render *[start, end)* of *lines* as numbered text for LLM injection."""
    picked = [ln for ln in lines if start <= ln.start < end]
    return "\n".join(f"[{ln.start:6.1f}s] {ln.text}" for ln in picked)
