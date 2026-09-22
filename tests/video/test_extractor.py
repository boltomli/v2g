import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from v2g.config import settings
from v2g.video import extractor

_FFMPEG_REQUIRED = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


def _make_video(path: Path, *, size: str = "1080x1920", duration: int = 1) -> Path:
    """Generate a lossless test clip with frequent keyframes."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f", "lavfi",
            "-i", f"testsrc2=size={size}:rate=5",
            "-t", str(duration),
            "-c:v", "libx264",
            "-crf", "0",
            "-g", "5",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _fake_yt_dlp(
    *, fail_with_subs: bool = False, fail_always: bool = False, write_video: bool = True
):
    """A subprocess.run stand-in mimicking yt-dlp.

    ``fail_with_subs`` reproduces the original crash: yt-dlp downloads the
    video successfully, then dies converting a subtitle sidecar — so the video
    file exists even on the failing subtitled attempt.
    """
    calls: list[list[str]] = []

    def run(args, capture_output=True):
        calls.append(list(args))
        if write_video:
            out = Path(str(args[args.index("-o") + 1]).replace("%(ext)s", "mp4"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake-video")
        failed = fail_always or (fail_with_subs and "--write-subs" in args)
        if failed:
            return subprocess.CompletedProcess(args, 1, b"", b"ERROR: Preprocessing: boom\n")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    return run, calls


@_FFMPEG_REQUIRED
def test_compression_produces_encoder_compatible_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path / "out")
    source = _make_video(tmp_path / "portrait.mp4")

    output = extractor.prepare_video_for_upload(source, tmp_path, max_mb=0.5)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]

    assert stream["width"] % 2 == 0
    assert stream["height"] % 2 == 0


def test_download_url_retries_without_subtitles_then_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path / "out")
    monkeypatch.setattr(extractor, "_get_duration", lambda path: 1.0)
    fake, calls = _fake_yt_dlp(fail_with_subs=True)
    monkeypatch.setattr(extractor.subprocess, "run", fake)

    url = "https://example.com/watch?v=abc"
    video = extractor._download_url(url, tmp_path / "run1")

    assert video.read_bytes() == b"fake-video"
    # subtitled attempt failed → retried once without subtitle flags
    assert len(calls) == 2
    assert "--write-subs" in calls[0]
    lang_value = calls[0][calls[0].index("--sub-langs") + 1]
    assert "-danmaku" in lang_value  # bullet-screen XML is not dialogue
    assert "--write-subs" not in calls[1]

    # second run: fully served from the media cache, no yt-dlp at all
    calls.clear()
    again = extractor._download_url(url, tmp_path / "run2")
    assert again == video
    assert calls == []


def test_download_url_reports_yt_dlp_error_when_no_video(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path / "out")
    monkeypatch.setattr(extractor, "_get_duration", lambda path: 1.0)
    fake, calls = _fake_yt_dlp(fail_always=True, write_video=False)
    monkeypatch.setattr(extractor.subprocess, "run", fake)

    with pytest.raises(FileNotFoundError, match="Preprocessing"):
        extractor._download_url("https://example.com/watch?v=bad", tmp_path / "run1")
    assert len(calls) == 2  # subtitled attempt, then the subtitle-less retry

    media = tmp_path / "out" / ".v2g_cache" / "media"
    assert not media.exists() or not any(media.iterdir())  # nothing half-cached


def test_download_url_keeps_completed_video_after_failed_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path / "out")
    monkeypatch.setattr(extractor, "_get_duration", lambda path: 1.0)
    fake, calls = _fake_yt_dlp(fail_always=True)
    monkeypatch.setattr(extractor.subprocess, "run", fake)

    video = extractor._download_url("https://example.com/watch?v=odd", tmp_path / "run1")

    assert video.read_bytes() == b"fake-video"
    assert len(calls) == 2  # both attempts failed, yet the completed video was kept


def test_download_url_falls_back_to_work_dir_when_cache_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_cache", False)
    monkeypatch.setattr(extractor, "_get_duration", lambda path: 1.0)
    fake, _ = _fake_yt_dlp()
    monkeypatch.setattr(extractor.subprocess, "run", fake)

    work = tmp_path / "work"
    video = extractor._download_url("https://example.com/watch?v=nocache", work)

    assert video.is_relative_to(work)  # keyed subdir inside the run's work dir


@_FFMPEG_REQUIRED
def test_prepare_video_for_upload_caches_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path / "out")
    source = _make_video(tmp_path / "clip.mp4")

    first = extractor.prepare_video_for_upload(source, tmp_path / "run1", max_mb=0.5)
    assert first.is_file()
    assert first.parent.parent == tmp_path / "out" / ".v2g_cache" / "media"

    def no_subprocess(*args, **kwargs):
        pytest.fail("re-encoded on cache hit")

    monkeypatch.setattr(extractor.subprocess, "run", no_subprocess)
    second = extractor.prepare_video_for_upload(source, tmp_path / "run2", max_mb=0.5)
    assert second == first


@_FFMPEG_REQUIRED
def test_split_video_caches_segments_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path / "out")
    source = _make_video(tmp_path / "long.mp4", size="320x240", duration=3)
    expected = math.ceil(extractor._get_duration(source))

    segments = extractor.split_video(source, tmp_path / "run1", segment_duration=1)
    assert [p.name for p in segments] == [
        f"segment_{i:03d}.mp4" for i in range(expected)
    ]
    assert all(p.is_file() and p.stat().st_size > 0 for p in segments)

    original_run = subprocess.run

    def guard(args, *a, **k):
        if args[0] == "ffmpeg":
            pytest.fail("re-cut segments on cache hit")
        return original_run(args, *a, **k)

    monkeypatch.setattr(extractor.subprocess, "run", guard)
    again = extractor.split_video(source, tmp_path / "run2", segment_duration=1)
    assert again == segments


