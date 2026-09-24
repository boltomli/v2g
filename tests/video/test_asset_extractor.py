import hashlib
import shutil
import subprocess

import pytest
from PIL import Image

from v2g.llm.analyzer import Character, GameDesign, GameObject, SceneDesign
from v2g.video.asset_extractor import (
    _KEY_OBJECT_ROLES,
    _establishing_shot,
    _is_ui_only,
    _match_scene,
    _parse_spatial_hint,
    _role_tokens,
    _scene_window,
    extract_assets,
    probe_video_size,
)

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
        characters=[Character(name=name, role="npc", visual="v") for name in _CHAR_NAMES],
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


# ── Scene anchoring + Chinese spatial hints ─────────────────────────────────


def test_parse_spatial_hint_understands_chinese_and_english():
    """Chinese design hints must crop — English-only matching was the bug."""
    assert _parse_spatial_hint("画面左侧近景，前景层") == (0.0, 0.0, 0.4, 1.0)
    assert _parse_spatial_hint("拱门上方正中，顶部前景层") == (0.25, 0.0, 0.5, 0.45)
    # worn props crop to the wear region, not the wearer's whole figure
    assert _parse_spatial_hint("画面中景左侧人物胸前，前景层") == (0.0, 0.25, 0.4, 0.35)
    assert _parse_spatial_hint("绿袍青年右腰侧，人物中层") == (0.6, 0.45, 0.4, 0.3)
    assert _parse_spatial_hint("骑士胸口正中，人物特写层") == (0.25, 0.25, 0.5, 0.35)
    # English hints keep working
    assert _parse_spatial_hint("left side of frame") == (0.0, 0.0, 0.4, 1.0)
    assert _parse_spatial_hint("center of frame") == (0.2, 0.1, 0.6, 0.8)
    assert _parse_spatial_hint("") == (0.0, 0.0, 1.0, 1.0)


def test_scene_window_tiles_the_timeline():
    assert _scene_window(30.1, 0, 4) == pytest.approx((0.0, 3.7625))
    assert _scene_window(30.1, 1, 4) == pytest.approx((3.7625, 11.2875))
    assert _scene_window(30.1, 3, 4) == pytest.approx((18.8125, 30.1))
    assert _scene_window(10.0, 0, 1) == (0.0, 10.0)


def test_establishing_shot_starts_at_the_boundary_nearest_the_cell():
    cuts = [3.366667, 6.233333, 10.766667, 14.733333]
    # scene 1 cell (3.76,11.29): nearest boundary = cut3.37 → the 2.5 s prop
    # wide (midpoint ≈5.0), NOT the longer dialogue beat and NOT the cell tail
    assert _establishing_shot((3.7625, 11.2875), cuts) == pytest.approx((3.366667, 6.233333))
    assert _establishing_shot((0.0, 5.0), cuts) == pytest.approx((0.0, 3.366667))
    assert _establishing_shot((0.0, 5.0), []) == (0.0, 5.0)  # no cuts → whole window


def _anchoring_design() -> GameDesign:
    return GameDesign(
        title="T",
        genre="visual novel",
        summary="s",
        mechanics=[],
        controls=[],
        style="s",
        objects=[
            GameObject(
                name="龙舌兰与陶盆", role="decoration", visual="灰绿色龙舌兰", spatial="画面左侧"
            ),
            GameObject(
                name="狼头银徽",
                role="decoration",
                visual="圆形银质胸针",
                behavior="固定于紫袍贵族左胸",
            ),
        ],
        scenes=[
            SceneDesign(name="石廊行进", description="紫袍年长贵族居中", layout="狭长石砌走廊"),
            SceneDesign(
                name="城堡前庭列队", description="队伍步出", layout="左侧陶盆龙舌兰，上方悬铁灯笼"
            ),
        ],
        characters=[],
    )


def test_match_scene_by_name_then_by_wearer():
    design = _anchoring_design()
    # direct name match → courtyard scene
    assert _match_scene(design, design.objects[0]) == 1
    # badge names no scene, but its behavior names the wearer in scene 0
    assert _match_scene(design, design.objects[1]) == 0
    # nothing shared → None (falls back to the seeded window)
    lonely = GameObject(name="XYZQ", role="decoration", visual="qwxz")
    assert _match_scene(design, lonely) is None


def test_role_tokens_fold_glosses_in_any_order():
    """Glosses come in either language on either side of the slash."""
    assert "decoration" in _role_tokens("decoration / 身份标识")
    assert "decoration" in _role_tokens("身份标识")  # fully translated role
    assert "collectible" in _role_tokens("身份标识 / 互动道具")  # Chinese-first
    assert "decoration" in _role_tokens("装饰 / 遮挡物")
    assert not _role_tokens("环境 / 场景转换触发") & _KEY_OBJECT_ROLES
    assert not _role_tokens("UI") & _KEY_OBJECT_ROLES
    assert "narrator" in _role_tokens("narrator / 旁白")


def test_ui_only_props_are_detected():
    ui = GameObject(
        name="赦免令", role="collectible", visual="情报面板图标", spatial="UI 层 · 面板"
    )
    prop = GameObject(
        name="悬挂铁灯笼", role="decoration", visual="锻铁灯笼", spatial="拱门上方正中"
    )
    assert _is_ui_only(ui)
    assert not _is_ui_only(prop)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_object_extraction_anchors_to_the_scene_that_names_it(tmp_path):
    """Two-shot video (red 0-2 s, blue 2-4 s): the object named by scene 2 must
    come from the blue master shot — and glossed/UI roles must be handled.

    Pre-fix failures this guards: the name-seeded window grabbed the RED half,
    and the role gloss "collectible / 剧情道具" skipped extraction entirely.
    """
    source = tmp_path / "halves.mp4"
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
            "color=c=red:size=160x120:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=160x120:rate=10:duration=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(source),
        ],
        check=True,
    )
    design = GameDesign(
        title="T",
        genre="g",
        summary="s",
        mechanics=[],
        controls=[],
        style="s",
        objects=[
            GameObject(
                name="BlueGem", role="collectible / 剧情道具", visual="blue gem", spatial=""
            ),
            GameObject(
                name="菜单图标", role="collectible", visual="情报面板图标", spatial="UI 层 · 面板"
            ),
        ],
        scenes=[
            SceneDesign(name="红色大厅", description="红色的房间", layout="红色墙壁"),
            SceneDesign(
                name="蓝色密室", description="蓝色的房间", layout="BlueGem sits on the altar"
            ),
        ],
        characters=[],
    )

    assets = extract_assets(source, design, tmp_path / "assets")

    obj = assets.get("objects/bluegem")
    assert obj is not None, "glossed role 'collectible / 剧情道具' must still extract"
    r, g, b = Image.open(obj).convert("RGB").resize((1, 1)).getpixel((0, 0))
    assert b > r + 50, f"object frame must come from the scene that names it, got rgb=({r},{g},{b})"
    assert "objects/菜单图标" not in assets, "UI-layer props never appear in a video frame"
