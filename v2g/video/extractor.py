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
    # Find the downloaded file
    for f in dest.iterdir():
        if f.suffix in (".mp4", ".mkv", ".webm", ".mov"):
            return f
    raise FileNotFoundError(f"yt-dlp produced no video file in {dest}")


def _extract_frames(video_path: Path, out_dir: Path, count: int) -> list[Path]:
    """Extract *count* evenly-spaced keyframes from *video_path* into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use ffmpeg select filter to pick N evenly spaced frames
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
    frames = sorted(out_dir.glob("frame_*.png"))
    return frames


def extract(source: str) -> list[Path]:
    """Return a list of frame image paths extracted from *source* (local path or URL).

    Frames are written to a temporary directory that the caller must keep alive.
    """
    count = settings.frame_count
    tmp = Path(tempfile.mkdtemp(prefix="v2g_"))

    src = Path(source)
    if src.is_file():
        video_path = src
    elif source.startswith(("http://", "https://")):
        video_path = _download_url(source, tmp)
    else:
        raise FileNotFoundError(f"Not a file or URL: {source}")

    frames_dir = tmp / "frames"
    return _extract_frames(video_path, frames_dir, count)
