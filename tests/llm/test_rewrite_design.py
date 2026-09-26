"""Stage 2's re-skin: a full redesign that cannot touch stage 1's identities."""

from v2g.llm import analyzer
from v2g.llm.analyzer import (
    Character,
    DialogueSample,
    GameDesign,
    GameObject,
    SceneDesign,
    rewrite_design,
)


def _design() -> GameDesign:
    return GameDesign(
        title="T",
        genre="g",
        summary="s",
        mechanics=[],
        controls=[],
        style="faithful to the source",
        characters=[Character(name="Hero", role="npc", visual="red hoodie, black hair")],
        objects=[GameObject(name="挂钟", role="decoration", visual="white wall clock", spatial="")],
        scenes=[SceneDesign(name="教室", description="night classroom")],
        dialogue_samples=[DialogueSample(speaker="Hero", line="take 1", line_zh="第一条")],
    )


def _capture_rewrite(monkeypatch, rewritten):
    """Run rewrite_design against a stub and return the prompt it was given."""
    seen: dict[str, str] = {}

    def fake(system: str, parts, **_kwargs):
        seen["prompt"] = parts[0]
        return rewritten

    monkeypatch.setattr(analyzer, "_request_design", fake)
    return seen


def test_a_theme_forces_a_full_redesign_and_keeps_identity(monkeypatch):
    seen = _capture_rewrite(monkeypatch, _design())

    rewrite_design(_design(), "vampire theme")

    prompt = seen["prompt"]
    assert "vampire theme" in prompt
    assert "FULL REDESIGN" in prompt
    for kind in ("characters:", "objects:", "scenes:"):  # every kind is re-invented
        assert kind in prompt
    assert "theme wins" in prompt  # the theme beats source fidelity in stage 2
    assert "KEEP UNCHANGED" in prompt  # stage 1's asset keys depend on these names


def test_without_a_theme_the_design_still_differs_from_the_source(monkeypatch):
    seen = _capture_rewrite(monkeypatch, _design())

    rewrite_design(_design(), None)

    assert "No theme was given" in seen["prompt"]
    assert "does not look like the source video" in seen["prompt"]


def test_a_rewrite_that_renames_an_identity_is_rejected(monkeypatch, tmp_path):
    renamed = _design()
    renamed.characters[0].name = "Someone Else"
    _capture_rewrite(monkeypatch, renamed)

    faithful = _design()
    kept = rewrite_design(faithful, "vampire theme")

    assert kept.characters[0].name == "Hero", "renaming would orphan the extracted sprite"


def test_a_rewrite_that_drops_an_identity_is_rejected(monkeypatch):
    shrunken = _design()
    shrunken.scenes = []
    _capture_rewrite(monkeypatch, shrunken)

    kept = rewrite_design(_design(), "vampire theme")
    assert [s.name for s in kept.scenes] == ["教室"]


def test_dialogue_is_restored_from_the_source_design(monkeypatch):
    silent = _design()
    silent.dialogue_samples = []
    _capture_rewrite(monkeypatch, silent)

    kept = rewrite_design(_design(), "vampire theme")
    assert kept.dialogue_samples[0].line == "take 1", (
        "transcript lines are not the model's to rewrite"
    )


def test_a_failed_rewrite_keeps_the_faithful_design(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("no api key")

    monkeypatch.setattr(analyzer, "_request_design", boom)
    faithful = _design()

    assert rewrite_design(faithful, "vampire theme") is faithful
