"""Scripts JSON parse layer: truncation salvages complete files, never crashes.

Regression for a real run where chat hit max_tokens mid-envelope and every
LLM-generated script was lost to the template fallback.
"""

import json

from v2g.godot.generator import _try_load_scripts


def test_complete_json_roundtrips():
    raw = json.dumps({"gm.gd": "extends Node\n", "hud.gd": "extends Control\n"})
    assert _try_load_scripts(raw) == {
        "gm.gd": "extends Node\n",
        "hud.gd": "extends Control\n",
    }


def test_markdown_fenced_json_parses():
    raw = '```json\n{"gm.gd": "extends Node\\n"}\n```'
    assert _try_load_scripts(raw) == {"gm.gd": "extends Node\n"}


def test_truncated_mid_value_keeps_complete_scripts():
    """Regression: a max_tokens cut mid-source must not lose finished files."""
    raw = '{"gm.gd": "extends Node\\nscore := 1", "hud.gd": "extends Control\\nfunc _ready() -> voi'
    out = _try_load_scripts(raw)
    assert out is not None
    assert out["gm.gd"] == "extends Node\nscore := 1"
    assert out["hud.gd"] == "extends Control\nfunc _ready() -> voi"  # cut closed verbatim


def test_truncated_mid_key_drops_partial_entry():
    raw = '{"gm.gd": "extends Node\\n", "hu'
    assert _try_load_scripts(raw) == {"gm.gd": "extends Node\n"}


def test_unusable_text_returns_none_for_template_fallback():
    assert _try_load_scripts("not json") is None
    assert _try_load_scripts("[1, 2, 3]") is None
