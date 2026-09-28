"""Tests for the shared codegen runner (model-written scripts → mp3)."""

import pytest

from v2g import codegen
from v2g.audio_api import AudioAPIError

_WAV_SCRIPT = """\
import sys, wave
w = wave.open(sys.argv[1], "wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(8000)
w.writeframes(b"\\x00\\x00" * 800)
w.close()
"""


def test_extract_code_strips_fences_and_prefers_the_fenced_block():
    assert codegen.extract_code("```python\nx = 1\n```") == "x = 1"
    assert codegen.extract_code("```\nx = 1\n```\n") == "x = 1"
    assert codegen.extract_code("  x = 1  ") == "x = 1"
    # Multiple fences: the last block wins (prose first, code last).
    assert codegen.extract_code("intro\n```python\na\n```\nmore\n```python\nb\n```") == "b"


def test_render_runs_the_script_with_a_relative_work_dir(tmp_path, monkeypatch):
    """Regression: argv[1] resolves against cwd (the work dir) — a relative
    work path must be absolutized or the script opens `<work>/<work>/out.wav`."""
    monkeypatch.chdir(tmp_path)

    data = codegen.render(_WAV_SCRIPT, "work/sfx_llm", stem="cue_0", timeout=30)

    assert data[:3] == b"ID3"
    assert (tmp_path / "work" / "sfx_llm" / "cue_0.py").is_file()


def test_render_keeps_the_source_for_inspection(tmp_path):
    data = codegen.render(_WAV_SCRIPT, tmp_path / "w", stem="bgm_llm", timeout=30)

    assert data[:3] == b"ID3"
    assert (tmp_path / "w" / "bgm_llm.py").read_text(encoding="utf-8") == _WAV_SCRIPT


def test_a_failing_script_raises_audio_api_error_with_the_tail(tmp_path):
    with pytest.raises(AudioAPIError) as exc:
        codegen.render("import sys\nsys.exit(3)\n", tmp_path / "w", stem="bad", timeout=30)

    assert "exit 3" in str(exc.value)
