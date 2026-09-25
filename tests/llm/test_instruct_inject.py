"""The theme instruction must force a full redesign, not a light reskin."""

from v2g.llm.analyzer import _inject_instruct


def test_instruct_is_injected_and_requires_per_kind_redesign():
    out = _inject_instruct("BASE MSG", "vampire theme")

    assert out.startswith("BASE MSG")
    assert "vampire theme" in out
    assert "FULL REDESIGN" in out
    # each asset kind carries its own redesign requirement
    for kind in ("characters:", "objects:", "scenes:"):
        assert kind in out
    # the theme overrides the source-faithfulness rules when they disagree
    assert "theme wins" in out


def test_without_an_instruct_the_message_is_untouched():
    assert _inject_instruct("BASE MSG", None) == "BASE MSG"
    assert _inject_instruct("BASE MSG", "") == "BASE MSG"
