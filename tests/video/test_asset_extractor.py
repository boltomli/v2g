import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from v2g.llm.analyzer import Character, GameDesign, GameObject, SceneDesign
from v2g.video import asset_extractor
from v2g.video.asset_extractor import (
    _BLANK_LUMA,
    _KEY_OBJECT_ROLES,
    SPRITE_SIZE,
    _box_crop,
    _candidate_windows,
    _capture_distinct,
    _contact_sheet,
    _establishing_shot,
    _is_ui_only,
    _match_scene,
    _mean_luma,
    _parse_box,
    _parse_spatial_hint,
    _region_crop,
    _role_tokens,
    _scene_window,
    _Target,
    _Verifier,
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

    assets = extract_assets(source, _design(), tmp_path / "assets", verify=False)

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


def test_candidate_windows_search_outward_from_the_scene_cell():
    """Establishing shot first, then the cell's other shots, then outside ones —
    a scene whose cell misses its content must still have shots to search."""
    cuts = [5.0, 10.0, 16.0, 20.0]
    assert _candidate_windows((7.0, 12.0), cuts, 24.0) == [
        (5.0, 10.0),  # establishing shot: boundary nearest the cell start
        (10.0, 16.0),  # the cell's other shot, widest overlap first
        (0.0, 5.0),  # outside shots, nearest to the cell first
        (16.0, 20.0),
        (20.0, 24.0),
    ]
    assert len(_candidate_windows((7.0, 12.0), cuts, 24.0, limit=2)) == 2
    # no cuts → one window, the cell itself
    assert _candidate_windows((0.0, 10.0), [], 30.0) == [(0.0, 10.0)]


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


def _halves_video(tmp_path) -> Path:
    """Red 0–2 s, blue 2–4 s, 160×120 — a video whose halves are tellable apart."""
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
    return source


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
    source = _halves_video(tmp_path)
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

    assets = extract_assets(source, design, tmp_path / "assets", verify=False)

    obj = assets.get("objects/bluegem")
    assert obj is not None, "glossed role 'collectible / 剧情道具' must still extract"
    r, g, b = Image.open(obj).convert("RGB").resize((1, 1)).getpixel((0, 0))
    assert b > r + 50, f"object frame must come from the scene that names it, got rgb=({r},{g},{b})"
    assert "objects/菜单图标" not in assets, "UI-layer props never appear in a video frame"


# ── Sprite geometry: 512×512 squares, stills follow the video ────────────────


def test_blank_gate_skips_black_cards_but_keeps_frames_with_content(tmp_path):
    """A transition frame must never ship first — but a subtitle/credit card or
    a dim (real) scene is content and has to survive the gate."""
    black = tmp_path / "black.png"
    Image.new("RGB", (320, 180), (0, 0, 0)).save(black)
    card = Image.new("RGB", (320, 180), (0, 0, 0))
    card.paste((255, 255, 255), (10, 10, 150, 60))
    card_path = tmp_path / "card.png"
    card.save(card_path)
    dim = tmp_path / "dim.png"
    Image.new("RGB", (320, 180), (40, 45, 60)).save(dim)

    assert _mean_luma(black) < _BLANK_LUMA
    assert _mean_luma(card_path) > _BLANK_LUMA
    assert _mean_luma(dim) > _BLANK_LUMA


def test_box_crop_frames_characters_half_body_and_objects_whole():
    # full-figure box in a 16:9 frame → a square that starts at the head and
    # stops around the waist: never the whole body, never a clipped head
    left, top, right, bottom = _box_crop([0.4, 0.05, 0.2, 0.9], 1920, 1080, "characters")
    side = right - left
    assert side == bottom - top, "sprite crop must be square"
    assert top == pytest.approx(0.05 * 1080, abs=2), "the head sits at the top of the crop"
    assert 0.5 * 0.9 * 1080 <= side < 0.9 * 1080, "half body: inside the full figure"

    # objects keep the whole subject, centred, with margin
    l, t, r, b = _box_crop([0.4, 0.4, 0.1, 0.1], 1920, 1080, "objects")
    assert r - l == b - t
    assert r - l == pytest.approx(0.1 * 1920 * 1.25, abs=1)


def test_region_crop_is_the_largest_square_covering_the_fallback_hint():
    assert _region_crop((0.0, 0.0, 1.0, 1.0), 1920, 1080) == (420, 0, 1500, 1080)
    assert _region_crop((0.0, 0.0, 0.4, 1.0), 1920, 1080) == (0, 0, 1080, 1080)


def test_letterbox_bars_are_found_and_never_reach_a_sprite(tmp_path):
    """A bar baked into the source is not picture: the crop must stay inside."""
    from PIL import Image

    framed = Image.new("RGB", (160, 120), (0, 0, 0))
    framed.paste((90, 60, 30), (0, 30, 160, 90))  # 30 px bars, top and bottom
    assert asset_extractor._content_rect(framed) == (0, 30, 160, 90)

    # a model box reaching into the bar is clamped into the picture area
    left, top, right, bottom = _box_crop(
        [0.0, 0.0, 1.0, 0.5], 160, 120, "objects", (0, 30, 160, 90)
    )
    assert top >= 30 and bottom <= 90, "the sprite must not carry the bar"
    assert right - left == bottom - top

    full_bleed = Image.new("RGB", (160, 120), (90, 60, 30))
    assert asset_extractor._content_rect(full_bleed) is None


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_a_letterboxed_source_yields_bars_free_sprites(tmp_path):
    """End-to-end: a 60-line picture padded into a 160×120 frame must produce a
    sprite whose own rows are all picture."""
    source = tmp_path / "padded.mp4"
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
            "testsrc2=size=160x60:rate=10",
            "-vf",
            "pad=160:120:0:30:black",
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
    assets = extract_assets(source, _shaped_design(), tmp_path / "assets", verify=False)

    sprite = assets["objects/bluegem"]
    with Image.open(sprite) as im:
        assert im.size == (SPRITE_SIZE, SPRITE_SIZE)
        g = im.convert("L")
        top_bar = g.crop((0, 0, SPRITE_SIZE, 1)).getextrema()[1] <= 8
        bottom_bar = g.crop((0, SPRITE_SIZE - 1, SPRITE_SIZE, SPRITE_SIZE)).getextrema()[1] <= 8
    assert not top_bar and not bottom_bar, "letterbox bars must not survive into the sprite"


def test_parse_box_clamps_and_rejects_unusable_answers():
    assert _parse_box([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]
    assert _parse_box([0.9, 0.9, 0.5, 0.5]) == pytest.approx(
        [0.9, 0.9, 0.1, 0.1]
    )  # clamped inside the frame
    assert _parse_box([0.5, 0.5, 0.01, 0.5]) is None  # degenerate
    assert _parse_box("on the left") is None


def _shaped_design() -> GameDesign:
    return GameDesign(
        title="T",
        genre="g",
        summary="s",
        mechanics=[],
        controls=[],
        style="s",
        objects=[GameObject(name="BlueGem", role="collectible", visual="blue gem", spatial="")],
        scenes=[
            SceneDesign(name="红色大厅", description="红色的房间", layout="红色墙壁"),
            SceneDesign(
                name="蓝色密室", description="蓝色的房间", layout="BlueGem sits on the altar"
            ),
        ],
        characters=[Character(name="Hero", role="npc", visual="a young hero")],
    )


def _testsrc_video(tmp_path) -> Path:
    """4 s of animated test pattern at 320×180 — non-blank, cheap, deterministic."""
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
            "testsrc2=size=320x180:rate=10",
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
    return source


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_assets_are_shaped_for_their_kind(tmp_path):
    """Sprites are square 512×512; backgrounds and scene stills keep the
    source video's own pixels — the window and the portrait slot both match."""
    assets = extract_assets(
        _testsrc_video(tmp_path), _shaped_design(), tmp_path / "a", verify=False
    )

    assert Image.open(assets["characters/hero"]).size == (SPRITE_SIZE, SPRITE_SIZE)
    assert Image.open(assets["objects/bluegem"]).size == (SPRITE_SIZE, SPRITE_SIZE)
    assert Image.open(assets["background"]).size == (320, 180)
    assert Image.open(assets["scenes/蓝色密室"]).size == (320, 180)


# ── Model check: content gate + retry ────────────────────────────────────────


class _StubVerifier:
    """Scripted stand-in for `_Verifier`: sheet rankings, then locate verdicts."""

    def __init__(
        self,
        picks: list[list[int]] | None = None,
        locates: list[tuple[bool, list[float] | None] | None] | None = None,
    ) -> None:
        self.enabled = True
        self.picks = list(picks or [])
        self.locates = list(locates or [None])
        self.pick_calls = 0
        self.locate_calls = 0
        self.sheet_count = 0

    def pick(self, sheet: Path, target: _Target, count: int) -> list[int] | None:
        self.pick_calls += 1
        self.sheet_count = count
        if not self.picks:
            return None
        return self.picks.pop(0) if len(self.picks) > 1 else self.picks[0]

    def locate(self, image: Path, target: _Target) -> tuple[bool, list[float] | None] | None:
        self.locate_calls += 1
        if not self.locates:
            return None
        return self.locates.pop(0) if len(self.locates) > 1 else self.locates[0]


_OBJECT = _Target("objects", "BlueGem: a blue gem")
_STILL = _Target("scenes", "蓝色密室: a blue room", square=False)


def _capture(
    source: Path,
    out: Path,
    verifier,
    *,
    target: _Target,
    pool: list[tuple[float, float]] | None = None,
    name: str = "asset.png",
) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    return _capture_distinct(
        source,
        out,
        out / name,
        used_hashes=set(),
        target=target,
        pool=pool or [(0.0, 4.0)],
        verifier=verifier,
    )


def _mean_rgb(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        return im.convert("RGB").resize((1, 1)).getpixel((0, 0))


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_the_contact_sheet_pick_decides_which_still_ships(tmp_path):
    """Every shot goes to the model as ONE contact sheet; the cell it ranks
    first is the asset — here the blue shot, not the red stamps before it."""
    stub = _StubVerifier(picks=[[1]])  # 0-based: cell 1 is the second (blue) round-robin slot

    assert _capture(
        _halves_video(tmp_path),
        tmp_path / "assets",
        stub,
        target=_STILL,
        pool=[(0.0, 2.0), (2.0, 4.0)],
        name="scene_blue.png",
    )
    assert stub.pick_calls == 1, "one call must rank every candidate"
    assert stub.sheet_count == 12, "the whole sheet is offered in one call"
    assert stub.locate_calls == 0, "a still the model vouched for ships as is"

    r, _g, b = _mean_rgb(tmp_path / "assets" / "scene_blue.png")
    assert b > r + 50, "the picked cell must be the one that ships"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_a_sprite_is_picked_from_the_sheet_then_checked_full_size(tmp_path):
    """A sprite is ranked from the same sheet, then each ranked pick is checked
    full size (which also yields the box) — a refused pick never ships."""
    stub = _StubVerifier(picks=[[0, 1]], locates=[(False, None), (True, [0.25, 0.25, 0.5, 0.5])])

    assert _capture(
        _halves_video(tmp_path),
        tmp_path / "assets",
        stub,
        target=_OBJECT,
        pool=[(0.0, 2.0), (2.0, 4.0)],
        name="obj_gem.png",
    )
    assert stub.pick_calls == 1, "one call ranks the shots for a sprite too"
    assert stub.locate_calls == 2, "then every ranked pick is checked at full size"

    r, _g, b = _mean_rgb(tmp_path / "assets" / "obj_gem.png")
    assert b > r + 50, "the refused pick must not ship"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_an_entity_the_model_cannot_vouch_for_still_ships_its_asset(tmp_path):
    """The model answers "none of these" → the first candidate ships anyway:
    an asset may be imperfect, it may not silently disappear."""
    still = _StubVerifier(picks=[[]])
    assert _capture(
        _halves_video(tmp_path),
        tmp_path / "assets",
        still,
        target=_STILL,
        pool=[(0.0, 2.0)],
        name="scene.png",
    )
    assert still.locate_calls == 0, "no pick means no full-size check either"
    assert Image.open(tmp_path / "assets" / "scene.png").size == (160, 120)

    sprite = _StubVerifier(locates=[(False, None)])
    assert _capture(
        _halves_video(tmp_path),
        tmp_path / "assets",
        sprite,
        target=_OBJECT,
        pool=[(0.0, 2.0)],
        name="obj.png",
    )
    assert Image.open(tmp_path / "assets" / "obj.png").size == (SPRITE_SIZE, SPRITE_SIZE)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_every_extracted_asset_is_checked_by_the_model(monkeypatch, tmp_path):
    """Characters, objects, backgrounds and scene stills all go through the
    gate — not just the object sprites."""
    seen: list[str] = []

    class _RecordingVerifier:
        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled

        def pick(self, sheet: Path, target: _Target, count: int) -> list[int]:
            seen.append(target.kind)
            return list(range(count))

        def locate(self, image: Path, target: _Target) -> tuple[bool, list[float] | None]:
            seen.append(target.kind)
            return True, [0.2, 0.1, 0.5, 0.8]

    monkeypatch.setattr(asset_extractor, "_Verifier", _RecordingVerifier)
    assets = extract_assets(_testsrc_video(tmp_path), _shaped_design(), tmp_path / "a")

    assert assets
    assert set(seen) >= {"background", "characters", "objects", "scenes"}


def test_contact_sheet_tiles_every_candidate_into_one_grid(tmp_path):
    """The model ranks candidates in one call — the grid has to hold them all
    at a legible size, cells left-to-right then top-to-bottom."""
    frames = []
    for i in range(5):
        path = tmp_path / f"f{i}.png"
        Image.new("RGB", (320, 180), (i * 40, 0, 0)).save(path)
        frames.append(path)
    sheet = tmp_path / "sheet.jpg"

    assert _contact_sheet(frames, sheet) == (3, 2)
    with Image.open(sheet) as im:
        assert im.size == (3 * 448, 2 * 252)  # tiles keep the frame's aspect


def test_verifier_parses_picks_and_boxes_from_json(monkeypatch, tmp_path):
    class _Response:
        text = '{"cells": [3, 7, 3], "box": [0.1, 0.2, 0.3, 0.4], "reason": "it is there"}'

    monkeypatch.setattr(asset_extractor, "chat", lambda *a, **k: _Response())
    image = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (180, 40, 40)).save(image)
    verifier = _Verifier(True)
    object_target = _Target("objects", "悬挂铁灯笼: an iron lantern")

    # cells are 1-based, deduplicated, out-of-range dropped
    assert verifier.pick(image, object_target, count=9) == [2, 6]

    class _Found:
        text = '{"found": true, "box": [0.1, 0.2, 0.3, 0.4], "reason": "it is there"}'

    monkeypatch.setattr(asset_extractor, "chat", lambda *a, **k: _Found())
    assert verifier.locate(image, object_target) == (True, [0.1, 0.2, 0.3, 0.4])
    # a background is never cropped, so no box is asked for
    assert verifier.locate(image, _Target("scenes", "石廊", square=False)) == (True, None)

    class _Garbage:
        text = "no json here"

    monkeypatch.setattr(asset_extractor, "chat", lambda *a, **k: _Garbage())
    assert verifier.pick(image, object_target, count=9) is None, "an unusable answer is no verdict"


def test_a_model_outage_disables_the_gate_instead_of_retrying(monkeypatch, tmp_path):
    calls: list[int] = []

    def boom(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("no api key")

    monkeypatch.setattr(asset_extractor, "chat", boom)
    image = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (180, 40, 40)).save(image)
    target = _Target("objects", "悬挂铁灯笼: an iron lantern")

    verifier = _Verifier(True)
    assert verifier.pick(image, target, count=4) is None
    assert verifier.locate(image, target) is None
    assert len(calls) == 1, "one outage silences the gate for the whole run"

    assert _Verifier(False).pick(image, target, count=4) is None
    assert len(calls) == 1, "V2G_ASSET_VERIFY=0 must never call the model"
