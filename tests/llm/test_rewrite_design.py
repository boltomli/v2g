"""Stage 2's re-skin: renames ride along, source identities stay resolvable."""

from pathlib import Path

import pytest

from v2g.config import settings
from v2g.godot import templates as T
from v2g.llm import analyzer
from v2g.llm.analyzer import (
    Character,
    DialogueSample,
    GameDesign,
    GameObject,
    SceneDesign,
    SceneTransition,
    asset_key_renames,
    rewrite_design,
)
from v2g.llm.errors import LLMOutputError


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
    seen: dict[str, object] = {}

    def fake(system: str, parts, **kwargs):
        seen["prompt"] = parts[0]
        seen["kwargs"] = kwargs
        return rewritten

    monkeypatch.setattr(analyzer, "_request_design", fake)
    return seen


def _boom(exc: Exception):
    def raiser(*_args, **_kwargs):
        raise exc

    return raiser


def test_a_theme_forces_a_full_redesign_including_names(monkeypatch):
    seen = _capture_rewrite(monkeypatch, _design())

    rewrite_design(_design(), "vampire theme")

    prompt = seen["prompt"]
    assert "vampire theme" in prompt
    assert "FULL REDESIGN" in prompt
    for kind in ("characters:", "objects:", "scenes:"):  # every kind is re-invented
        assert kind in prompt
    assert "theme wins" in prompt  # the theme beats source fidelity in stage 2
    assert "new name that fits the" in prompt  # names are re-skinned too
    assert "KEEP UNCHANGED" in prompt  # face_id and gameplay anchor the redesign


def test_without_a_theme_the_design_still_differs_from_the_source(monkeypatch):
    seen = _capture_rewrite(monkeypatch, _design())

    rewrite_design(_design(), None)

    assert "No theme was given" in seen["prompt"]
    assert "does not look like the source video" in seen["prompt"]


def test_the_rewrite_drops_baked_in_text_from_the_design(monkeypatch):
    """Stage 1 faithfully reports the source's burned-in subtitles and
    watermark — the re-skin must not carry them into the art direction, or the
    redraw briefs would ask the image model to paint text."""
    seen = _capture_rewrite(monkeypatch, _design())

    rewrite_design(_design(), "vampire theme")

    assert "DROP — all baked-in text" in seen["prompt"]
    assert "the concept itself" in seen["prompt"]  # not just the source's spelling
    assert "describe the surface, not its lettering" in seen["prompt"]


def test_renames_are_applied_and_the_speaker_follows(monkeypatch):
    """Renaming is the contract now: characters, objects and scenes all move,
    and the transcript's speaker labels follow so portraits still resolve."""
    src = _design()
    src.characters[0].face_id = "char_01"
    renamed = _design()
    renamed.characters[0].name = "Vampire Lord"
    renamed.characters[0].face_id = "char_01"  # the anchor, not the name, survives
    renamed.objects[0].name = "Cursed Clock"
    renamed.scenes[0].name = "Crypt"
    _capture_rewrite(monkeypatch, renamed)

    kept = rewrite_design(src, "vampire theme")

    assert [c.name for c in kept.characters] == ["Vampire Lord"]
    assert [o.name for o in kept.objects] == ["Cursed Clock"]
    assert [s.name for s in kept.scenes] == ["Crypt"]
    assert kept.characters[0].face_id == "char_01"
    assert kept.dialogue_samples[0].speaker == "Vampire Lord"  # "Hero" followed the rename
    assert kept.dialogue_samples[0].line == "take 1"  # the transcript itself stays verbatim


def test_rename_map_lets_the_extracted_frames_follow(monkeypatch):
    """The pipeline contract: keys stage 1 cut under the source names keep
    resolving — asset keys move, and the portrait lookup finds them."""
    renamed = _design()
    renamed.characters[0].name = "Vampire Lord"
    _capture_rewrite(monkeypatch, renamed)

    source = _design()
    kept = rewrite_design(source, "vampire theme")
    assets = {"characters/hero": Path("/x/char_hero.png"), "background": Path("/x/background.png")}
    moved = {asset_key_renames(source, kept).get(k, k): p for k, p in assets.items()}

    assert moved == {
        "characters/vampire_lord": Path("/x/char_hero.png"),
        "background": Path("/x/background.png"),
    }
    steps, _, portraits, names = T._build_story(kept, moved)
    assert portraits == {"vampire_lord": "res://assets/char_hero.png"}
    assert any(s.get("sprite") == "vampire_lord" for s in steps)
    assert names["vampire_lord"] == "Vampire Lord"


def test_a_reordered_cast_is_put_back_into_source_order(monkeypatch):
    """The model may reorder the list — face_id re-anchors it, so asset keys
    and portraits pair up with the entities they were extracted from."""
    first = Character(name="Hero", role="protagonist", visual="v", face_id="char_01")
    second = Character(name="Maid", role="npc", visual="v", face_id="char_02")
    src = _design()
    src.characters = [first, second]

    swapped = _design()
    swapped.characters = [
        Character(name="Vampire Lord", role="protagonist", visual="v", face_id="char_01"),
        Character(name="Clockwork Maid", role="npc", visual="v", face_id="char_02"),
    ]
    swapped.characters.reverse()  # the model answers in its own order
    _capture_rewrite(monkeypatch, swapped)

    kept = rewrite_design(src, "vampire theme")

    assert [c.name for c in kept.characters] == ["Vampire Lord", "Clockwork Maid"]
    moves = asset_key_renames(src, kept)
    assert moves == {
        "characters/hero": "characters/vampire_lord",
        "characters/maid": "characters/clockwork_maid",
    }


def test_duplicate_new_names_are_rejected(monkeypatch):
    """Two entities collapsing onto one safe name would collide asset keys."""
    src = _design()
    src.characters = [
        Character(name="Hero", role="protagonist", visual="a"),
        Character(name="Maid", role="npc", visual="b"),
    ]
    collided = _design()
    collided.characters = [
        Character(name="Same", role="protagonist", visual="a"),
        Character(name="Same!", role="npc", visual="b"),  # safe name: "same"
    ]
    _capture_rewrite(monkeypatch, collided)

    with pytest.raises(LLMOutputError, match="duplicate character name"):
        rewrite_design(src, "vampire theme")


def test_scene_transitions_follow_renamed_scenes(monkeypatch):
    src = _design()
    src.scenes.append(SceneDesign(name="走廊", description="lit corridor"))
    src.scene_transitions = [SceneTransition(source="教室", destination="走廊")]

    renamed = _design()
    renamed.scenes = [
        SceneDesign(name="Crypt", description="night classroom"),
        SceneDesign(name="Cloister", description="lit corridor"),
    ]
    renamed.scene_transitions = [SceneTransition(source="教室", destination="走廊")]
    _capture_rewrite(monkeypatch, renamed)

    kept = rewrite_design(src, "vampire theme")
    assert [(t.source, t.destination) for t in kept.scene_transitions] == [("Crypt", "Cloister")]


def test_a_rewrite_that_drops_an_identity_is_rejected(monkeypatch):
    shrunken = _design()
    shrunken.scenes = []
    _capture_rewrite(monkeypatch, shrunken)

    with pytest.raises(LLMOutputError, match="reshaped the scene list"):
        rewrite_design(_design(), "vampire theme")


def test_dialogue_is_restored_from_the_source_design(monkeypatch):
    silent = _design()
    silent.dialogue_samples = []
    _capture_rewrite(monkeypatch, silent)

    kept = rewrite_design(_design(), "vampire theme")
    assert kept.dialogue_samples[0].line == "take 1", (
        "transcript lines are not the model's to rewrite"
    )


def test_a_failed_rewrite_stops_the_run(monkeypatch):
    """A re-skin that never arrived must not carry on: the redraw stage would
    burn image generation (and stage 3) on a source-faithful design."""
    monkeypatch.setattr(
        analyzer, "_request_design", _boom(LLMOutputError("truncated at max_tokens"))
    )

    with pytest.raises(LLMOutputError, match="stopping before the redraw stage"):
        rewrite_design(_design(), "vampire theme")


def test_a_transport_failure_also_stops_the_run(monkeypatch):
    """The old code caught every exception and continued — an outage during
    the re-skin is a stop too, same as anywhere else in the pipeline."""
    monkeypatch.setattr(analyzer, "_request_design", _boom(RuntimeError("no api key")))

    with pytest.raises(RuntimeError, match="no api key"):
        rewrite_design(_design(), "vampire theme")


def test_the_rewrite_budget_is_the_configured_ceiling(monkeypatch):
    """Regression: a fixed 16384 let the reasoning model think the whole cap
    away (finish=length, empty body) — the re-skin parse failed with an empty
    response and the run shipped the source design."""
    seen = _capture_rewrite(monkeypatch, _design())

    rewrite_design(_design(), "vampire theme")

    assert seen["kwargs"]["max_tokens"] == settings.llm_max_tokens
