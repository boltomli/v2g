"""Character-merge quality: what the cast dedup treats as evidence.

Stage 1 lists the same person once per segment, so `_merge_designs` has to
decide which entries are one person. Name and visual features are that
evidence — and both used to answer questions nobody asked.
"""

from v2g.llm import analyzer
from v2g.llm.analyzer import (
    Character,
    DialogueSample,
    GameDesign,
    _absorb_character,
    _extract_features,
    _merge_designs,
)


def _design(**over) -> GameDesign:
    base = {
        "title": "T",
        "genre": "g",
        "summary": "s",
        "mechanics": [],
        "controls": [],
        "style": "st",
        "objects": [],
    }
    base.update(over)
    return GameDesign(**base)


def test_features_are_matched_on_word_boundaries():
    """Regression: a substring test read "hundred" as red hair, "bold" as old
    and "instant" as tanned skin — phantom features that made unrelated
    characters look like the same person."""
    assert _extract_features("a hundred soldiers march past") == set()
    assert _extract_features("the bold knight") == set()
    assert _extract_features("an instant later") == set()
    assert _extract_features("bright golden light") == set()
    assert _extract_features("darkness everywhere") == set()

    assert "hair:red" in _extract_features("red hair")
    assert "hair:red" in _extract_features("her hair is red.")
    assert "age:old" in _extract_features("an old man")
    assert "skin:dark" in _extract_features("dark skin")
    assert "skin:tan" in _extract_features("tan skin")


def test_inflected_features_still_match():
    """Word boundaries must not cost real descriptions their features."""
    assert "face:beard" in _extract_features("a bearded sailor")
    assert "distinguishing:mask" in _extract_features("a masked figure")
    assert "distinguishing:hood" in _extract_features("hooded stranger")
    assert "face:scar" in _extract_features("scarred cheek")


def test_phantom_features_no_longer_merge_two_people():
    """Two different people, one of them described with words that merely
    contain feature keywords — the old matcher merged them by mistake."""
    a = Character(name="Alice", role="npc", visual="a hundred candles behind her, bold stance")
    b = Character(name="Bruno", role="npc", visual="tall, athletic build")

    assert analyzer._find_character_match(b, [a]) is None


def test_abilities_merge_case_insensitively():
    existing = Character(name="A", role="npc", visual="v", abilities=["Fireball", "Dash"])
    incoming = Character(name="A", role="npc", visual="v", abilities=["fireball", "Heal"])

    _absorb_character(existing, incoming)

    assert existing.abilities == ["Fireball", "Dash", "Heal"]


def test_merging_does_not_write_back_into_the_source_designs():
    """The merge absorbs, appends and extends — all of that must land on the
    copy, or the caller's segment designs come back mutated."""
    first = _design(
        characters=[Character(name="Hero", role="npc", visual="v", abilities=["A"])],
        dialogue_samples=[DialogueSample(speaker="Hero", line="hi", line_zh="你好")],
    )
    second = _design(
        characters=[Character(name="Hero", role="npc", visual="v", abilities=["B"])],
        dialogue_samples=[DialogueSample(speaker="Hero", line="there", line_zh="那边")],
    )

    merged = _merge_designs([first, second])

    assert merged.characters[0].abilities == ["A", "B"]
    assert first.characters[0].abilities == ["A"], "source design was mutated"
    assert len(first.dialogue_samples) == 1, "source dialogue list was appended to"
