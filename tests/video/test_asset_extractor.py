import hashlib
import shutil
import subprocess

import pytest

from v2g.llm.analyzer import Character, GameDesign
from v2g.video.asset_extractor import extract_assets, probe_video_size

_CHAR_NAMES = [
    "Lady / The Bride",
    "Guard Knight / Protector",
    "Fernando Turbay / Patriarch",
    "Prince Andrew",
    "The King",
    "Green Companion",
]


def _design() -> GameDesign:
    return GameDesign(
        title="T",
        genre="visual novel",
        summary="s",
        mechanics=[],
        controls=[],
        style="s",
        objects=[],
        scenes=[],
        characters=[
            Character(name=name, role="npc", visual="v") for name in _CHAR_NAMES
        ],
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_character_sprites_are_byte_distinct(tmp_path):
    source = tmp_path / "clip.mp4"
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
            "testsrc2=size=640x360:rate=10",
            "-t",
            "4",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(source),
        ],
        check=True,
    )

    assets = extract_assets(source, _design(), tmp_path / "assets")

    char_paths = [p for key, p in assets.items() if key.startswith("characters/")]
    assert len(char_paths) == len(_CHAR_NAMES)
    digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in char_paths}
    assert len(digests) == len(char_paths), "character sprites must be byte-distinct"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_probe_video_size_reports_pixels_and_missing_file(tmp_path):
    """Viewport adaptation feeds on this: exact pixel size, None when unknown."""
    source = tmp_path / "clip.mp4"
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
            "testsrc2=size=640x360:rate=10",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(source),
        ],
        check=True,
    )

    assert probe_video_size(source) == (640, 360)
    assert probe_video_size(tmp_path / "missing.mp4") is None
