"""Extract representative frames from a video file or URL using ffmpeg / yt-dlp.

Provides two extraction strategies:
1. **Fixed-interval** (`extract_frames`): one frame every N seconds — good for short videos.
2. **Scene-detect** (`extract_keyframes`): ffmpeg scene-change detection — adapts to
   content, extracts frames where visuals change significantly. Works for any length.

The pipeline auto-selects based on video duration.
"""

import logging
import math
import subprocess
from pathlib import Path

from v2g import cache
from v2g.config import settings

log = logging.getLogger(__name__)

_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")


# ── Source resolution ────────────────────────────────────────────────────────


def _run_yt_dlp(args: list[str]) -> str | None:
    """Run yt-dlp once; None on success, else a short failure description."""
    result = subprocess.run(args, capture_output=True)
    if result.returncode == 0:
        return None
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return f"exit {result.returncode}: {lines[-1] if lines else 'no output'}"


def _usable_video(directory: Path) -> Path | None:
    """Complete, ffprobe-readable ``video.<ext>`` in *directory*, else None.

    Rejects yt-dlp intermediates (``video.f137.mp4`` fragments, ``.part``
    files) and truncated merges left behind by a failed attempt.
    """
    video = next(
        (
            f
            for f in directory.iterdir()
            if f.stem == "video"
            and f.suffix.lower() in _VIDEO_SUFFIXES
            and f.stat().st_size > 0
        ),
        None,
    )
    if video is None:
        return None
    try:
        _get_duration(video)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
    return video


def _download_url(url: str, work_dir: Path) -> Path:
    """Download *url* via yt-dlp into the media cache and return the video file.

    The download is cached across runs keyed by URL — rerunning the same
    source never re-downloads. Subtitles (including auto-generated ones) are
    best-effort: fetched first as SRT sidecars next to the video (danmaku
    bullet-screen XML is excluded — it is not dialogue), and any yt-dlp
    failure falls back to a subtitle-less retry; a completed video from a
    failed attempt is kept when ffprobe validates it. *work_dir* is the
    fallback location when ``V2G_MEDIA_CACHE=0``.
    """
    entry = cache.media_entry("dl", {"url": url}, fallback=work_dir)
    hit = cache.media_find(entry, _usable_video)
    if hit is not None:
        log.info("Download cache hit: %s → %s", url, hit)
        return hit

    base = ["yt-dlp", "--no-playlist", "-f", "bv*+ba/b", "--merge-output-format", "mp4"]
    sub_args = [
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "all,-live,-danmaku",
        "--convert-subs", "srt",
    ]
    with cache.media_stage(entry) as stage:
        out_tpl = str(stage / "video.%(ext)s")
        err = _run_yt_dlp([*base, *sub_args, "-o", out_tpl, url])
        if err is not None:
            log.warning("yt-dlp with subtitles failed (%s) — retrying without subtitles", err)
            err = _run_yt_dlp([*base, "-o", out_tpl, url])
        video = _usable_video(stage)
        if video is None:
            detail = f": {err}" if err else ""
            raise FileNotFoundError(f"yt-dlp produced no usable video for {url}{detail}")
        if err is not None:
            log.warning("yt-dlp exited with %s — keeping the video it produced", err)
        name = video.name
    return entry / name


def resolve_source(source: str, work_dir: Path) -> Path:
    """Resolve *source* to a local video path.

    URL downloads (and their subtitle sidecars) land in the shared media
    cache ``<output_root>/.v2g_cache/media/`` so reruns never re-download;
    *work_dir* is only the fallback location when ``V2G_MEDIA_CACHE=0``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    if src.is_file():
        return src
    if source.startswith(("http://", "https://")):
        return _download_url(source, work_dir)
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
            "-fps_mode", "vfr",
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
            "-fps_mode", "vfr",
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

    Segments are cached across runs, keyed by source identity (size + mtime)
    and the split/compression settings — reruns skip the re-cut entirely.
    *tmp_dir* is only the fallback location when ``V2G_MEDIA_CACHE=0``.
    """
    duration = _get_duration(video_path)
    if duration <= segment_duration:
        return [video_path]

    count = math.ceil(duration / segment_duration)
    names = [f"segment_{i:03d}.mp4" for i in range(count)]
    stat = video_path.stat()
    entry = cache.media_entry(
        "split",
        {
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime_ns,
            "segment_duration": segment_duration,
            "video_max_mb": settings.video_max_mb,
        },
        fallback=tmp_dir,
    )

    def probe(directory: Path) -> Path | None:
        first = directory / names[0]
        return first if all((directory / n).is_file() for n in names) else None

    hit = cache.media_find(entry, probe)
    if hit is not None:
        log.info("Segment cache hit: %d segments from %s", count, video_path.name)
        return [entry / n for n in names]

    with cache.media_stage(entry) as stage:
        seg_idx = 0
        start = 0.0

        while start < duration:
            seg_path = stage / names[seg_idx]
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
                compressed = stage / "compressed.mp4"
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
                compressed.replace(seg_path)

            log.info("  Segment %d: %.0fs–%.0fs (%.1f MB)",
                     seg_idx, start, end, seg_path.stat().st_size / (1024 * 1024))
            start = end
            seg_idx += 1

    return [entry / n for n in names]


# ── Video preparation for LLM upload ─────────────────────────────────────────


def prepare_video_for_upload(video_path: Path, tmp_dir: Path, max_mb: int = 20) -> Path:
    """Prepare video for LLM upload: trim to max_duration, compress if over max_mb.

    Returns path to the processed video file. The output is cached across runs,
    keyed by source identity (size + mtime) and the trim/compression settings —
    reruns skip the re-encode. *tmp_dir* is only the fallback location when
    ``V2G_MEDIA_CACHE=0``.
    """
    stat = video_path.stat()
    entry = cache.media_entry(
        "prep",
        {
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime_ns,
            "max_mb": max_mb,
            "max_duration": settings.max_duration,
        },
        fallback=tmp_dir,
    )

    def probe(directory: Path) -> Path | None:
        upload = directory / "upload.mp4"
        return upload if upload.is_file() and upload.stat().st_size > 0 else None

    hit = cache.media_find(entry, probe)
    if hit is not None:
        log.info("Prepared-video cache hit: %s", hit)
        return hit

    with cache.media_stage(entry) as stage:
        duration = _get_duration(video_path)
        max_dur = settings.max_duration

        # Step 1: trim duration if needed
        upload = stage / "upload.mp4"
        trim_args: list[str] = ["ffmpeg", "-y", "-i", str(video_path)]
        if duration > max_dur:
            trim_args += ["-t", str(max_dur)]
        trim_args += ["-c", "copy", str(upload)]
        subprocess.run(trim_args, capture_output=True, check=True)

        # Step 2: compress if file is too large
        size_mb = upload.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            compressed = stage / "compressed.mp4"
            target_bitrate = int(max_mb * 8 * 1024 / max_dur)  # kbps
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(upload),
                    "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-b:v", f"{target_bitrate}k",
                    "-b:a", "64k",
                    "-preset", "fast",
                    str(compressed),
                ],
                capture_output=True,
                check=True,
            )
            compressed.replace(upload)

    return entry / "upload.mp4"


# ── Legacy convenience ───────────────────────────────────────────────────────


def extract(video_path: Path, work_dir: Path) -> list[Path]:
    """Extract frames for *video_path* using the best strategy for its length.

    Short video (<5 min): fixed-interval extraction.
    Long video (≥5 min): scene-detect keyframe extraction.
    Frames are written under *work_dir*.
    """
    frames_dir = work_dir / "frames"
    duration = _get_duration(video_path)

    if duration > 300:  # >5 minutes → scene detection
        log.info("Long video (%.0fs) — using scene-detect keyframe extraction", duration)
        return extract_keyframes(video_path, frames_dir)
    else:
        log.info("Short video (%.0fs) — using fixed-interval extraction", duration)
        return extract_frames(video_path, frames_dir, settings.frame_interval)
