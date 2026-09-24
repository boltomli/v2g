"""Per-script compile check + repair/fallback: broken LLM extras never ship.

The headless boot only parses scripts main.tscn references, so alliance/save
overlays previously reached the editor with parse errors. Regressions cover
the Godot check itself (real godot, skipped when absent) and the fallback
policy around it (content-based fake check, no godot needed).
"""

import shutil
from pathlib import Path

import pytest

from v2g.config import settings
from v2g.godot import generator as G
from v2g.godot import templates as T
from v2g.llm.analyzer import GameDesign

_BROKEN = "SCRIPT ERROR: Parse Error: _broken_"


def _design() -> GameDesign:
    return GameDesign(
        title="T", genre="vn", summary="s", mechanics=[], controls=[],
        style="st", objects=[], scenes=[],
    )


def _project(tmp_path: Path) -> Path:
    (tmp_path / "project.godot").write_text(T.project_dot_godot("T"), encoding="utf-8")
    return tmp_path


def _fake_check(root: Path, fname: str) -> list[str]:
    """Content-based stand-in for godot --check-only: marker = parse error."""
    source = (root / fname).read_text(encoding="utf-8")
    return [_BROKEN] if "_broken_" in source else []


_GODOT = shutil.which(settings.godot_path) or (
    settings.godot_path if Path(settings.godot_path).is_file() else None
)
requires_godot = pytest.mark.skipif(_GODOT is None, reason="godot not on PATH")


@requires_godot
def test_check_script_reports_parse_error(tmp_path):
    """Regression: `_visible` on CanvasLayer must be caught at generation time."""
    proj = _project(tmp_path)
    (proj / "bad.gd").write_text(
        "extends CanvasLayer\n\nfunc _ready() -> void:\n\t_visible = false\n",
        encoding="utf-8",
    )

    errs = G._check_script(proj, "bad.gd")

    assert errs and any("_visible" in e for e in errs)


@requires_godot
def test_check_script_accepts_valid_script(tmp_path):
    proj = _project(tmp_path)
    (proj / "ok.gd").write_text("extends Node\n", encoding="utf-8")

    assert G._check_script(proj, "ok.gd") == []


def test_validate_keeps_llm_repaired_script(tmp_path, monkeypatch, caplog):
    from v2g import runlog

    proj = _project(tmp_path)
    broken = "extends CanvasLayer\n# _broken_\n"
    (proj / "alliance_map.gd").write_text(broken, encoding="utf-8")
    monkeypatch.setattr(G, "_check_script", _fake_check)
    monkeypatch.setattr(
        G, "_repair_scripts",
        lambda d, s, fail: {"alliance_map.gd": "extends CanvasLayer\n"},
    )

    with caplog.at_level(runlog.NOTICE, logger="v2g.godot.generator"):
        out = G._validate_scripts(proj, _design(), {"alliance_map.gd": broken})

    assert out["alliance_map.gd"] == "extends CanvasLayer\n"
    assert (proj / "alliance_map.gd").read_text(encoding="utf-8") == "extends CanvasLayer\n"
    # the repair outcome must be console-visible (NOTICE outranks the console gate)
    assert any(
        r.levelno == runlog.NOTICE and "alliance_map.gd" in r.getMessage()
        for r in caplog.records
    )


def test_validate_falls_back_when_repair_fails(tmp_path, monkeypatch):
    """Still-broken scripts: extras dropped, game_manager → template, VN kept."""
    proj = _project(tmp_path)
    scripts = {
        "vn_manager.gd": "extends Control\n# _broken_\n",
        "game_manager.gd": "extends Node\n# _broken_\n",
        "alliance_map.gd": "extends CanvasLayer\n# _broken_\n",
    }
    for fname, source in scripts.items():
        (proj / fname).write_text(source, encoding="utf-8")
    monkeypatch.setattr(G, "_check_script", _fake_check)
    monkeypatch.setattr(G, "_repair_scripts", lambda d, s, fail: {})

    out = G._validate_scripts(proj, _design(), scripts)

    # unreferenced extra: removed from disk and from the manifest
    assert "alliance_map.gd" not in out
    assert not (proj / "alliance_map.gd").exists()
    # essential state script: known-good template replaces it on disk too
    assert "func add_score" in out["game_manager.gd"]
    assert (proj / "game_manager.gd").read_text(encoding="utf-8") == out["game_manager.gd"]
    # template-owned runtime: never silently dropped — error stays surfaced
    assert out["vn_manager.gd"] == scripts["vn_manager.gd"]
    assert (proj / "vn_manager.gd").exists()
