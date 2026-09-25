"""Regression tests for LLM JSON repair: the pipeline crashed on truncated output."""

import pytest

from v2g.llm.analyzer import _parse
from v2g.llm.errors import LLMOutputError

_MINIMAL = (
    '{"title":"T","genre":"g","summary":"s","mechanics":["m"],"controls":["c"],'
    '"style":"st","objects":[]}'
)


def test_parse_plain_json():
    design = _parse(_MINIMAL)
    assert design.title == "T"


def test_parse_markdown_fenced_json():
    design = _parse(f"```json\n{_MINIMAL}\n```")
    assert design.title == "T"


def test_parse_json_with_surrounding_prose():
    design = _parse(f"Here is the design:\n{_MINIMAL}\nHope it helps!")
    assert design.title == "T"


def test_parse_truncated_mid_array_is_salvaged():
    """Regression: truncation must be closed, not crash the run (issue traceback).

    Truncation point sits after all required fields, as in the real failure
    (cut deep inside a long document).
    """
    raw = (
        '{"title":"T","genre":"g","summary":"s","mechanics":["m"],"controls":["c"],'
        '"style":"st","objects":[],"scenes":[{"name":"A","description":"d"},'
        '{"name":"B","description":"lonely isl'
    )
    design = _parse(raw)
    assert design.title == "T"
    assert [s.name for s in design.scenes] == ["A", "B"]
    assert design.scenes[1].description == "lonely isl"  # cut mid-string, closed


def test_parse_truncated_inside_string_value_is_salvaged():
    raw = (
        '{"title":"T","genre":"g","summary":"s","mechanics":["m"],"controls":["c"],'
        '"style":"st","objects":[],"narrative":"My Real Titl'
    )
    design = _parse(raw)
    assert design.title == "T"
    assert design.narrative == "My Real Titl"


def test_parse_truncated_after_comma_is_salvaged():
    raw = (
        '{"title":"T","genre":"g","summary":"s","mechanics":["m"],"controls":["c"],'
        '"style":"st","objects":[],"narrative":'
    )
    design = _parse(raw)
    assert design.title == "T"
    assert design.objects == []


def test_parse_stray_token_mid_document_is_repaired():
    """Regression: a stray `",` the model emitted mid-JSON must not lose the tail.

    Real run (001 segment): `…要素"],","controls":…` broke the decode at that
    point; cut-back salvage kept only the five keys before the error and the
    segment died on missing controls/style/objects. A deletion repair keeps
    the whole document, and schema validation must skip the parseable-but
    mangled variant (key `,controls`) in favor of the exact one.
    """
    raw = (
        '{"title":"T","genre":"g","summary":"s","mechanics":["m"],'
        '"controls":["c"],"style":"st","objects":[]}'
    )
    broken = raw.replace('],"controls"', '],","controls"')
    assert broken != raw  # the corruption actually landed

    design = _parse(broken)

    assert design.controls == ["c"]  # not the mangled `,controls` variant
    assert design.style == "st"
    assert design.objects == []


def test_parse_garbage_raises_llm_output_error():
    with pytest.raises(LLMOutputError) as exc:
        _parse("not json at all")
    assert "Cannot parse LLM design JSON" in str(exc.value)


def test_parse_valid_json_missing_required_fields_raises_llm_output_error():
    with pytest.raises(LLMOutputError) as exc:
        _parse('{"only":"junk"}')
    assert "validation" in str(exc.value)


def test_llm_output_error_is_value_error():
    """Chunked analysis recovers by catching ValueError — contract must hold."""
    assert issubclass(LLMOutputError, ValueError)
