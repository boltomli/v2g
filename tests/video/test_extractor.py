import json
import shutil
import subprocess

import pytest

from v2g.video.extractor import prepare_video_for_upload


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_compression_produces_encoder_compatible_dimensions(tmp_path):
    source = tmp_path / "portrait.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1080x1920:rate=5",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            str(source),
        ],
        check=True,
    )

    output = prepare_video_for_upload(source, tmp_path, max_mb=0.5)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]

    assert stream["width"] % 2 == 0
    assert stream["height"] % 2 == 0
