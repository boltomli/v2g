"""Stage 2's redraw: always on, theme optional, and never silent about failure.

Each asset is redrawn from a kind-specific brief (character / object / scene)
and — with `imagegen_ab` — twice: one candidate with the source frame as
visual reference, one drawn from the brief alone. A vision judge keeps one;
without a usable verdict the candidate that changed the source most wins.
"""

import logging
from pathlib import Path

import pytest
from PIL import Image

from v2g import runlog
from v2g.godot import generator as G
from v2g.llm import image_gen
from v2g.llm.analyzer import Character, GameDesign, GameObject, SceneDesign
from v2g.llm.client import ChatResult

BLUE = (0, 0, 255)  # source frames
NAVY = (0, 0, 128)  # reference candidate — barely moved from the source
GREEN = (0, 128, 0)  # text-only candidate — clearly changed


def _design() -> GameDesign:
    return GameDesign(
        title="T",
        genre="visual novel",
        summary="s",
        narrative="",
        mechanics=[],
        controls=[],
        style="s",
        characters=[
            Character(name="Lady / The Bride", role="protagonist", visual="white gown, silver hair")
        ],
        objects=[GameObject(name="Signet Ring", role="collectible", visual="gold ring")],
        scenes=[SceneDesign(name="Courtyard", description="moonlit stone courtyard")],
    )


def _assets(tmp_path: Path) -> dict[str, Path]:
    return {
        "background": _write(tmp_path / "background.png", BLUE),
        "characters/lady___the_bride": _write(tmp_path / "char_lady___the_bride.png", BLUE),
        "objects/signet_ring": _write(tmp_path / "obj_signet_ring.png", BLUE),
    }


def _write(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 8), color).save(path)
    return path


def _color(path: Path) -> tuple[int, ...]:
    with Image.open(path) as img:
        return img.getpixel((0, 0))


class _Recording:
    def __init__(self, *, fail_on: str = "", fail_text_only: bool = False, available: bool = True):
        self.calls: list[tuple[str, str, bool]] = []  # (source name, brief, condition)
        self.fail_on = fail_on
        self.fail_text_only = fail_text_only
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def transform(
        self,
        path: Path,
        prompt: str,
        *,
        size: str = "",
        condition: bool = True,
        dest: Path | None = None,
    ) -> Path:
        if self.fail_on and self.fail_on in path.name:
            raise RuntimeError("boom")
        if self.fail_text_only and not condition:
            raise RuntimeError("no t2i today")
        self.calls.append((path.name, prompt, condition))
        out = dest or path
        Image.new("RGB", (8, 8), NAVY if condition else GREEN).save(out)
        return out


class _Chat:
    """Stub for the vision judge — tests set `text` to control the verdict."""

    def __init__(self, text: str = "not json"):
        self.text = text
        self.calls: list[list] = []
        self.raises = False

    def __call__(self, system: str, parts: list, **kwargs) -> ChatResult:
        self.calls.append(parts)
        if self.raises:
            raise RuntimeError("api down")
        return ChatResult(self.text, False, "judge-key")


@pytest.fixture(autouse=True)
def judge(monkeypatch) -> _Chat:
    """No real LLM in tests: default verdict is unusable → difference fallback.

    A/B is switched on here so the two-candidate behaviour is what the tests
    exercise; a test that wants a single candidate sets ``imagegen_ab`` itself.
    """
    stub = _Chat()
    monkeypatch.setattr(G, "chat", stub)
    monkeypatch.setattr(G.settings, "imagegen_ab", True)
    return stub


def test_restyle_is_gated_on_assets_not_on_the_theme(monkeypatch, tmp_path):
    """Stage 2 redraws with a theme or without one — only an empty asset set
    stops it (there would be nothing to draw)."""
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    G.restyle_assets(_design(), {}, "vampire style")
    G.restyle_assets(_design(), {}, None)
    assert provider.calls == []

    G.restyle_assets(_design(), _assets(tmp_path), None)
    assert provider.calls, "no theme must still redraw — the art only has to differ"


def test_restyle_draws_both_candidates_with_kind_specific_briefs(monkeypatch, tmp_path):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)

    G.restyle_assets(_design(), assets, "vampire style")

    briefs = {name: prompt for name, prompt, _ in provider.calls}
    conds = {(name, condition) for name, _, condition in provider.calls}
    for name in ("background.png", "char_lady___the_bride.png", "obj_signet_ring.png"):
        assert {(name, True), (name, False)} <= conds  # reference AND text-only
    assert all(p.startswith("Redraw in this theme: vampire style") for p in briefs.values())

    char = briefs["char_lady___the_bride.png"]
    assert "half-body portrait" in char and "must differ from the source frame" in char
    assert "SUBJECT below" in char  # the design, not the frame, defines the look
    assert "Lady / The Bride (protagonist): white gown, silver hair" in char

    obj = briefs["obj_signet_ring.png"]
    assert "cut out and centered" in obj and "no scenery" in obj
    assert "Signet Ring (collectible): gold ring" in obj

    bg = briefs["background.png"]
    assert "Redesign this LOCATION" in bg
    assert "Courtyard: moonlit stone courtyard" in bg


@pytest.mark.parametrize(("pick", "color"), [("ref", NAVY), ("text", GREEN)])
def test_judge_pick_keeps_that_candidate(monkeypatch, tmp_path, judge, pick, color):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)
    bg = assets["background"]
    judge.text = f'{{"pick": "{pick}", "reason": "follows the brief"}}'

    G.restyle_assets(_design(), assets, "vampire style")

    assert _color(assets["characters/lady___the_bride"]) == color
    assert _color(assets["background"]) == color
    assert assets["background"] != bg  # the winner lands in a sibling redraw file
    assert assets["background"].name == "background.redraw.png"
    assert _color(bg) == BLUE  # the extracted frame itself is never written to
    assert len(judge.calls[0]) == 4  # brief + source frame + both candidates


def test_without_a_usable_judge_the_bigger_change_wins(monkeypatch, tmp_path, judge, caplog):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)

    with caplog.at_level(logging.INFO, logger=G.log.name):
        G.restyle_assets(_design(), assets, "vampire style")

    assert _color(assets["background"]) == GREEN  # text-only moved furthest from BLUE
    assert "changed the source most" in caplog.text


def test_judge_outage_still_restyles_via_visual_difference(monkeypatch, tmp_path, judge, caplog):
    """A judge that cannot even be reached must not fail the run or the asset."""
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)
    judge.raises = True

    with caplog.at_level(logging.WARNING, logger=G.log.name):
        G.restyle_assets(_design(), assets, "vampire style")

    assert "Redraw judge unavailable" in caplog.text
    assert _color(assets["background"]) == GREEN
    assert "kept their original frames" not in caplog.text


def test_one_surviving_candidate_wins_without_the_judge(monkeypatch, tmp_path, judge, caplog):
    """ref ok + text failed → the survivor is used directly, judge never consulted."""
    provider = _Recording(fail_text_only=True)
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)

    with caplog.at_level(logging.WARNING, logger=G.log.name):
        G.restyle_assets(_design(), assets, "vampire style")

    assert judge.calls == []
    assert all(condition for _, _, condition in provider.calls)  # only ref attempts ran
    assert _color(assets["background"]) == NAVY
    assert "kept their original frames" not in caplog.text


def test_ab_off_draws_one_reference_candidate_beside_the_frame(monkeypatch, tmp_path, judge):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    monkeypatch.setattr(G.settings, "imagegen_ab", False)
    assets = _assets(tmp_path)

    G.restyle_assets(_design(), assets, "vampire style")

    assert [(name, cond) for name, _, cond in provider.calls] == [
        ("background.png", True),
        ("char_lady___the_bride.png", True),
        ("obj_signet_ring.png", True),
    ]
    assert judge.calls == []  # nothing to compare
    assert _color(assets["background"]) == NAVY
    assert _color(tmp_path / "background.png") == BLUE  # the frame is left alone
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "background.png",
        "background.redraw.png",
        "char_lady___the_bride.png",
        "char_lady___the_bride.redraw.png",
        "obj_signet_ring.png",
        "obj_signet_ring.redraw.png",
    ]  # one redraw beside each frame — no candidates staged


def test_both_candidates_are_kept_for_comparison(monkeypatch, tmp_path, judge):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)

    run = runlog.start_run("src.mp4", tmp_path / "run")
    try:
        G.restyle_assets(_design(), assets, "vampire style")
        kept = sorted(p.name for p in (run / "work" / "imagegen").iterdir())
    finally:
        runlog.reset()

    assert kept == [
        "background.ref.png",
        "background.text.png",
        "characters__lady___the_bride.ref.png",
        "characters__lady___the_bride.text.png",
        "objects__signet_ring.ref.png",
        "objects__signet_ring.text.png",
    ]


def test_restyle_keeps_originals_when_provider_unavailable(monkeypatch, tmp_path):
    provider = _Recording(available=False)
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)

    G.restyle_assets(_design(), assets, "vampire style")

    assert provider.calls == []
    assert _color(assets["background"]) == BLUE


def test_restyle_keeps_the_original_of_a_failing_asset(monkeypatch, tmp_path, caplog):
    provider = _Recording(fail_on="char_")
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)
    char = assets["characters/lady___the_bride"]

    with caplog.at_level(logging.WARNING, logger=G.log.name):
        G.restyle_assets(_design(), assets, "vampire style")

    assert assets["characters/lady___the_bride"] == char  # both candidates failed
    assert _color(char) == BLUE
    assert _color(assets["background"]) == GREEN  # others proceeded
    assert [name for name, _, _ in provider.calls] == [
        "background.png",
        "background.png",
        "obj_signet_ring.png",
        "obj_signet_ring.png",
    ]  # every other asset proceeded
    assert "1/3 asset(s) kept their original frames" in caplog.text


def test_without_a_theme_the_redraw_only_has_to_differ(monkeypatch, tmp_path):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    G.restyle_assets(_design(), _assets(tmp_path), None)

    assert provider.calls, "stage 2 runs without a theme"
    assert all(
        "differs from the source video" in prompt for _, prompt, _ in provider.calls
    )  # no theme: unlike the source is the whole brief


def test_the_theme_supplies_the_redraw_style(monkeypatch, tmp_path):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)

    G.restyle_assets(_design(), _assets(tmp_path), "vampire style")

    assert all(
        prompt.startswith("Redraw in this theme: vampire style") for _, prompt, _ in provider.calls
    )


def test_instruct_without_provider_is_reported_not_silent(monkeypatch, tmp_path, caplog):
    """-i makes restyle mandatory: an unrunnable backend must be said out loud."""
    monkeypatch.setattr(image_gen, "get_provider", lambda: image_gen.NullProvider())
    assets = _assets(tmp_path)

    with caplog.at_level(logging.WARNING, logger=G.log.name):
        G.restyle_assets(_design(), assets, "vampire style")

    assert "NOT restyled" in caplog.text
    assert _color(assets["background"]) == BLUE


def test_redraw_without_provider_is_a_notice(monkeypatch, tmp_path, caplog):
    """Stage 2 with no backend: said out loud, frames kept as they were."""
    monkeypatch.setattr(image_gen, "get_provider", lambda: image_gen.NullProvider())
    assets = _assets(tmp_path)

    with caplog.at_level(runlog.NOTICE, logger=G.log.name):
        G.restyle_assets(_design(), assets, None)

    assert "Redraw skipped" in caplog.text
    assert _color(assets["background"]) == BLUE
