"""Extract representative frames from a video file or URL using ffmpeg / yt-dlp."""

import subprocess
import tempfile
from pathlib import Path

from v2g.config import settings


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


def extract_frames(video_path: Path, out_dir: Path, count: int) -> list[Path]:
    """Extract *count* evenly-spaced keyframes from *video_path* into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select='not(mod(n\\,max(1,floor(n_total/{count}))))',setpts=N/FRAME_RATE/TB",
            "-vsync", "vfr",
            "-frames:v", str(count),
            str(out_dir / "frame_%03d.png"),
        ],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.png"))


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
        # Scale down and reduce bitrate proportionally
        target_bitrate = int(max_mb * 8 * 1024 / max_dur)  # kbps
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(current),
                "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease",
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


def extract(source: str) -> list[Path]:
    """Return a list of frame image paths extracted from *source* (local path or URL)."""
    count = settings.frame_count
    video_path, tmp = resolve_source(source)
    frames_dir = tmp / "frames"
    return extract_frames(video_path, frames_dir, count)
