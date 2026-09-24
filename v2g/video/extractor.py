"""Extract representative frames from a video file or URL using ffmpeg / yt-dlp.

Provides two extraction strategies:
1. **Fixed-interval** (`extract_frames`): one frame every N seconds — good for short videos.
2. **Scene-detect** (`extract_keyframes`): ffmpeg scene-change detection — adapts to
   content, extracts frames where visuals change significantly. Works for any length.

The pipeline auto-selects based on video duration.
"""

import logging
import subprocess
from pathlib import Path

from v2g import cache
from v2g.config import settings

log = logging.getLogger(__name__)

_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")


# ── Source resolution ────────────────────────────────────────────────────────


def _run_yt_dlp(args: list[str]) -> str | None:
    """Run yt-dlp once; None on success, else a short failure description."""
    result = subprocess.run(args, capture_output=True, check=False)
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
            if f.stem == "video" and f.suffix.lower() in _VIDEO_SUFFIXES and f.stat().st_size > 0
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
        "--sub-langs",
        "all,-live,-danmaku",
        "--convert-subs",
        "srt",
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
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
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
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-fps_mode",
            "vfr",
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
        log.info(
            "  %d frames > %d, raising threshold to %.2f (attempt %d)",
            len(frames),
            max_frames,
            threshold,
            attempts,
        )
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
        log.info(
            "  Only %d scene-change frames — supplementing with interval extraction", len(frames)
        )
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
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{threshold:.2f})',setpts=N/FRAME_RATE/TB",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            str(out_dir / "frame_%04d.png"),
        ],
        capture_output=True,
        check=True,
    )


# ── Video segmentation (for long videos in detail mode) ──────────────────────


def _probe_manifest(directory: Path) -> Path | None:
    """``manifest.txt`` in *directory* naming artifacts that all exist and are non-empty."""
    manifest = directory / "manifest.txt"
    if not manifest.is_file():
        return None
    names = _read_manifest(manifest)
    if not names or not all(
        (directory / n).is_file() and (directory / n).stat().st_size > 0 for n in names
    ):
        return None
    return manifest


def _read_manifest(manifest: Path) -> list[str]:
    return [n for n in manifest.read_text(encoding="utf-8").splitlines() if n.strip()]


def _fit_size(path: Path, cap_mb: float) -> list[Path]:
    """Losslessly halve *path* by duration until every piece is ≤ *cap_mb*.

    Only ``-c copy`` is used — bytes are divided, never re-encoded, and encode
    parameters are never re-tuned. Relies on the compression pass beforehand:
    ``-bf 0`` (dts==pts, so ``-t`` cuts exactly at the midpoint) and a keyframe
    at every ⅛ of the segment (midpoints of the first two split levels always
    land on one; ``-ss`` never starts a tail at the piece's own beginning).
    *path* is deleted; leaf pieces come back in chronological order.

    Raises RuntimeError when a midpoint cut fails to shrink either half, i.e.
    splitting can no longer make progress.
    """
    if path.stat().st_size <= cap_mb * 1024 * 1024:
        return [path]
    dur = _get_duration(path)
    half = dur / 2
    children = [path.with_name(f"{path.stem}_{i}{path.suffix}") for i in range(2)]
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-t", f"{half:.3f}", "-c", "copy", str(children[0])],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{half:.3f}", "-i", str(path), "-c", "copy", str(children[1])],
        capture_output=True,
        check=True,
    )
    head_dur = _get_duration(children[0])
    tail_dur = _get_duration(children[1])
    if head_dur >= dur - 0.05 or tail_dur >= dur - 0.05:
        raise RuntimeError(
            f"{path.name}: midpoint {half:.2f}s made no progress "
            f"({dur:.2f}s → head {head_dur:.2f}s, tail {tail_dur:.2f}s — no "
            f"cuttable boundary); {path.stat().st_size / (1024 * 1024):.2f} MB "
            f"still exceeds {cap_mb} MB"
        )
    path.unlink()
    return _fit_size(children[0], cap_mb) + _fit_size(children[1], cap_mb)


def split_video(video_path: Path, tmp_dir: Path, segment_duration: int = 60) -> list[Path]:
    """Split a video into uploadable segments of *segment_duration* seconds.

    Each segment is cut with ``-c copy``, compressed at most once when it
    exceeds ``video_max_mb`` and — still over after that single pass — halved
    losslessly by duration until every piece fits. Returns the leaf pieces in
    order; a ``manifest.txt`` in the cache entry records them.

    Segments are cached across runs, keyed by source identity (size + mtime)
    and the split/compression settings — reruns skip the re-cut entirely.
    *tmp_dir* is only the fallback location when ``V2G_MEDIA_CACHE=0``.
    """
    duration = _get_duration(video_path)
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

    hit = cache.media_find(entry, _probe_manifest)
    if hit is not None:
        names = _read_manifest(hit)
        log.info("Segment cache hit: %d segments from %s", len(names), video_path.name)
        return [entry / n for n in names]

    with cache.media_stage(entry) as stage:
        leaves: list[str] = []
        seg_idx = 0
        start = 0.0

        while start < duration:
            seg_path = stage / f"segment_{seg_idx:03d}.mp4"
            end = min(start + segment_duration, duration)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start:.2f}",
                    "-i",
                    str(video_path),
                    "-t",
                    f"{end - start:.2f}",
                    "-c",
                    "copy",
                    str(seg_path),
                ],
                capture_output=True,
                check=True,
            )
            # Compress segment if too large for upload — at most once. ``-bf 0``
            # keeps dts==pts (the lossless midpoint split below trims by dts)
            # and a keyframe every eighth puts a cut on the midpoint of the
            # first two split levels.
            size_mb = seg_path.stat().st_size / (1024 * 1024)
            if size_mb > settings.video_max_mb:
                seg_len = end - start
                compressed = stage / "compressed.mp4"
                target_bitrate = int(settings.video_max_mb * 8 * 1024 / segment_duration)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(seg_path),
                        "-vf",
                        "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                        "-b:v",
                        f"{target_bitrate}k",
                        "-b:a",
                        "64k",
                        "-bf",
                        "0",
                        "-force_key_frames",
                        f"expr:gte(t,n_forced*{seg_len / 8:.3f})",
                        "-preset",
                        "fast",
                        str(compressed),
                    ],
                    capture_output=True,
                    check=True,
                )
                compressed.replace(seg_path)
            pieces = _fit_size(seg_path, settings.video_max_mb)
            leaves.extend(p.name for p in pieces)
            log.info(
                "  Segment %d: %.0fs–%.0fs → %d piece(s), %s",
                seg_idx,
                start,
                end,
                len(pieces),
                ", ".join(f"{p.name} {p.stat().st_size / (1024 * 1024):.1f} MB" for p in pieces),
            )
            start = end
            seg_idx += 1

        (stage / "manifest.txt").write_text("\n".join(leaves) + "\n", encoding="utf-8")

    return [entry / n for n in leaves]


# ── Video preparation for LLM upload ─────────────────────────────────────────


def prepare_video_for_upload(video_path: Path, tmp_dir: Path, max_mb: int = 20) -> list[Path]:
    """Prepare video for LLM upload: trim to max_duration, compress at most once
    if over *max_mb*, then losslessly halve by duration until every piece fits.

    Returns the pieces in order (usually one). The output is cached across runs,
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

    hit = cache.media_find(entry, _probe_manifest)
    if hit is not None:
        names = _read_manifest(hit)
        log.info("Prepared-video cache hit: %d file(s)", len(names))
        return [entry / n for n in names]

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

        # Step 2: compress once if the file is too large — never retried.
        # ``-bf 0`` keeps dts==pts (the lossless split below trims by dts) and
        # a keyframe every eighth of the clip covers the midpoints of the first
        # two split levels, so any overshoot is split away instead of re-tuned.
        size_mb = upload.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            upload_dur = _get_duration(upload)
            compressed = stage / "compressed.mp4"
            target_bitrate = int(max_mb * 8 * 1024 / max_dur)  # kbps
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(upload),
                    "-vf",
                    "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-b:v",
                    f"{target_bitrate}k",
                    "-b:a",
                    "64k",
                    "-bf",
                    "0",
                    "-force_key_frames",
                    f"expr:gte(t,n_forced*{upload_dur / 8:.3f})",
                    "-preset",
                    "fast",
                    str(compressed),
                ],
                capture_output=True,
                check=True,
            )
            compressed.replace(upload)

        # Step 3: split away any overshoot — lossless halves, no re-encode
        pieces = _fit_size(upload, max_mb)
        names = [p.name for p in pieces]
        (stage / "manifest.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    return [entry / n for n in names]


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
