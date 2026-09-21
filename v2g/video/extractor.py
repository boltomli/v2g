"""Extract representative frames from a video file or URL using ffmpeg / yt-dlp.

Provides two extraction strategies:
1. **Fixed-interval** (`extract_frames`): one frame every N seconds — good for short videos.
2. **Scene-detect** (`extract_keyframes`): ffmpeg scene-change detection — adapts to
   content, extracts frames where visuals change significantly. Works for any length.

The pipeline auto-selects based on video duration.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

from v2g.config import settings

log = logging.getLogger(__name__)


# ── Source resolution ────────────────────────────────────────────────────────


def _download_url(url: str, dest: Path) -> Path:
    """Download a video from *url* into *dest* via yt-dlp and return the file path."""
    out_tpl = str(dest / "video.%(ext)s")
    subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f", "bv*+ba/b",
            "--merge-output-format", "mp4",
            "-o", out_tpl,
            url,
        ],
        check=True,
        capture_output=True,
    )
    for f in dest.iterdir():
        if f.suffix in (".mp4", ".mkv", ".webm", ".mov"):
            return f
    raise FileNotFoundError(f"yt-dlp produced no video file in {dest}")


def resolve_source(source: str) -> tuple[Path, Path]:
    """Resolve *source* to a local video path. Returns (video_path, tmp_dir).

    The caller must keep *tmp_dir* alive for the duration of use.
    """
    tmp = Path(tempfile.mkdtemp(prefix="v2g_"))
    src = Path(source)
    if src.is_file():
        return src, tmp
    if source.startswith(("http://", "https://")):
        return _download_url(source, tmp), tmp
    raise FileNotFoundError(f"Not a file or URL: {source}")


def _get_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


# ── Frame extraction strategies ──────────────────────────────────────────────


def extract_frames(video_path: Path, out_dir: Path, interval: float = 2.0) -> list[Path]:
    """Extract one frame every *interval* seconds. Best for short videos (<5 min)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / interval
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-vsync", "vfr",
            str(out_dir / "frame_%04d.png"),
        ],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.png"))


def extract_keyframes(
    video_path: Path,
    out_dir: Path,
    *,
    threshold: float | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    """Extract keyframes using ffmpeg scene-change detection.

    Adapts to video content: gets frames where the visual scene changes
    significantly, rather than at fixed intervals. Works for any video length.

    Args:
        video_path: Source video file.
        out_dir: Directory to write extracted PNGs.
        threshold: Scene-change sensitivity (0.0–1.0). Lower = more frames.
            Default: ``V2G_SCENE_THRESHOLD`` (0.3).
        max_frames: Cap on extracted frames. Default: ``V2G_FRAME_BUDGET`` (40).
            If exceeded, threshold is auto-raised to reduce count.

    Returns:
        Sorted list of extracted frame paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if threshold is None:
        threshold = settings.scene_threshold
    if max_frames is None:
        max_frames = settings.frame_budget

    # First pass: extract with scene detection
    log.info("Scene-detect extraction (threshold=%.2f, max=%d)...", threshold, max_frames)
    _run_scene_extract(video_path, out_dir, threshold)
    frames = sorted(out_dir.glob("frame_*.png"))

    # If too many frames, re-extract with higher threshold
    attempts = 0
    while len(frames) > max_frames and attempts < 4:
        threshold = min(threshold + 0.15, 0.95)
        attempts += 1
        log.info("  %d frames > %d, raising threshold to %.2f (attempt %d)",
                 len(frames), max_frames, threshold, attempts)
        # Clean up and re-extract
        for f in frames:
            f.unlink()
        _run_scene_extract(video_path, out_dir, threshold)
        frames = sorted(out_dir.glob("frame_*.png"))

    # If still too many, keep evenly-spaced subset
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        keep = [frames[int(i * step)] for i in range(max_frames)]
        for f in frames:
            if f not in keep:
                f.unlink()
        frames = keep
        log.info("  Kept %d frames (evenly spaced from scene-detect set)", len(frames))

    # Fallback: if scene detection found very few frames (e.g. static video),
    # supplement with interval extraction
    if len(frames) < 5:
        log.info("  Only %d scene-change frames — supplementing with interval extraction", len(frames))
        duration = _get_duration(video_path)
        interval = max(duration / max_frames, 2.0)
        extra_dir = out_dir / "_interval"
        extra = extract_frames(video_path, extra_dir, interval)
        # Merge: scene frames + interval frames, deduplicate by keeping all
        for i, f in enumerate(extra):
            dest = out_dir / f"interval_{i:04d}.png"
            f.rename(dest)
        extra_dir.rmdir()
        frames = sorted(out_dir.glob("*.png"))
        # Cap again
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            keep = [frames[int(i * step)] for i in range(max_frames)]
            for f in frames:
                if f not in keep:
                    f.unlink()
            frames = keep

    log.info("Extracted %d keyframes", len(frames))
    return frames


def _run_scene_extract(video_path: Path, out_dir: Path, threshold: float) -> None:
    """Run ffmpeg scene-change detection, writing frames to *out_dir*."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select='gt(scene,{threshold:.2f})',setpts=N/FRAME_RATE/TB",
            "-vsync", "vfr",
            "-q:v", "2",
            str(out_dir / "frame_%04d.png"),
        ],
        capture_output=True,
        check=True,
    )


# ── Video segmentation (for long videos in detail mode) ──────────────────────


def split_video(video_path: Path, tmp_dir: Path, segment_duration: int = 600) -> list[Path]:
    """Split a video into segments of *segment_duration* seconds.

    Returns a list of segment file paths in order.
    Used for detail-mode analysis of long videos that exceed LLM upload limits.
    """
    duration = _get_duration(video_path)
    if duration <= segment_duration:
        return [video_path]

    segments: list[Path] = []
    seg_idx = 0
    start = 0.0

    while start < duration:
        seg_path = tmp_dir / f"segment_{seg_idx:03d}.mp4"
        end = min(start + segment_duration, duration)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.2f}",
                "-i", str(video_path),
                "-t", f"{end - start:.2f}",
                "-c", "copy",
                str(seg_path),
            ],
            capture_output=True,
            check=True,
        )
        # Compress segment if too large for upload
        size_mb = seg_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.video_max_mb:
            compressed = tmp_dir / f"segment_{seg_idx:03d}_c.mp4"
            target_bitrate = int(settings.video_max_mb * 8 * 1024 / segment_duration)
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(seg_path),
                    "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-b:v", f"{target_bitrate}k",
                    "-b:a", "64k",
                    "-preset", "fast",
                    str(compressed),
                ],
                capture_output=True,
                check=True,
            )
            seg_path.unlink(missing_ok=True)
            seg_path = compressed

        segments.append(seg_path)
        log.info("  Segment %d: %.0fs–%.0fs (%.1f MB)",
                 seg_idx, start, end, seg_path.stat().st_size / (1024 * 1024))
        start = end
        seg_idx += 1

    return segments


# ── Video preparation for LLM upload ─────────────────────────────────────────


def prepare_video_for_upload(video_path: Path, tmp_dir: Path, max_mb: int = 20) -> Path:
    """Prepare video for LLM upload: trim to max_duration, compress if over max_mb.

    Returns path to the processed video file.
    """
    duration = _get_duration(video_path)
    max_dur = settings.max_duration

    # Step 1: trim duration if needed
    trimmed = tmp_dir / "trimmed.mp4"
    trim_args: list[str] = ["ffmpeg", "-y", "-i", str(video_path)]
    if duration > max_dur:
        trim_args += ["-t", str(max_dur)]
    trim_args += ["-c", "copy", str(trimmed)]
    subprocess.run(trim_args, capture_output=True, check=True)

    current = trimmed

    # Step 2: compress if file is too large
    size_mb = current.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        compressed = tmp_dir / "compressed.mp4"
        target_bitrate = int(max_mb * 8 * 1024 / max_dur)  # kbps
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(current),
                "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-b:v", f"{target_bitrate}k",
                "-b:a", "64k",
                "-preset", "fast",
                str(compressed),
            ],
            capture_output=True,
            check=True,
        )
        current = compressed

    return current


# ── Legacy convenience ───────────────────────────────────────────────────────


def extract(source: str) -> list[Path]:
    """Extract frames using the best strategy for the video length.

    Short video (<5 min): fixed-interval extraction.
    Long video (≥5 min): scene-detect keyframe extraction.
    """
    video_path, tmp = resolve_source(source)
    frames_dir = tmp / "frames"
    duration = _get_duration(video_path)

    if duration > 300:  # >5 minutes → scene detection
        log.info("Long video (%.0fs) — using scene-detect keyframe extraction", duration)
        return extract_keyframes(video_path, frames_dir)
    else:
        log.info("Short video (%.0fs) — using fixed-interval extraction", duration)
        return extract_frames(video_path, frames_dir, settings.frame_interval)
