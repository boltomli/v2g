"""Tests for sound effects — fake LLM answers, real local synthesis scripts."""

from v2g import sfx
from v2g.config import settings
from v2g.llm.analyzer import DialogueSample, GameDesign
from v2g.llm.client import ChatResult

# A minimal synthesis script: writes silence, ffmpeg turns it into an mp3.
_WAV_SCRIPT = """\
import sys, wave
w = wave.open(sys.argv[1], "wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(8000)
w.writeframes(b"\\x00\\x00" * 800)
w.close()
"""


def _design(samples: int = 2) -> GameDesign:
    return GameDesign(
        title="Sfx Test",
        genre="visual novel",
        summary="s",
        mechanics=[],
        controls=["Space: advance"],
        style="cinematic",
        objects=[],
        atmosphere="mysterious",
        dialogue_samples=[
            DialogueSample(speaker="", line="", line_zh=f"第{i}句") for i in range(samples)
        ],
    )


def _settings(monkeypatch, tmp_path, **over) -> None:
    """Point SFX at local codegen inside a per-test output root."""
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "sfx_provider", "llm")
    for key, value in over.items():
        monkeypatch.setattr(settings, key, value)


def _fake_chat(*answers: str):
    """A chat stand-in returning the given script texts in order (last repeats)."""
    queue = list(answers)

    def _chat(*_a, **_k):
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return ChatResult(text, False, "k")

    return _chat


def test_event_and_cue_clips_land_in_project_and_cache(tmp_path, monkeypatch):
    _settings(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        sfx, "chat", lambda *a, **k: calls.append(a) or ChatResult(_WAV_SCRIPT, False, "k")
    )
    run_sfx = sfx.Sfx(tmp_path / "run")
    run_sfx.cues = ["door knock", ""]

    select = run_sfx.event("select")
    cue = run_sfx.step_cue(0)

    assert select == "res://assets/sfx/select.mp3"
    assert cue == "res://assets/sfx/cue_0.mp3"
    assert (tmp_path / "run" / "assets" / "sfx" / "select.mp3").read_bytes()[:3] == b"ID3"
    assert run_sfx.step_cue(1) is None  # empty cue stays silent
    assert run_sfx.generated == 2
    # The cue reaches the code writer verbatim as the sound to render.
    assert len(calls) == 2
    system, parts = calls[0]
    assert "sound effect" in system
    assert "no speech" in system
    assert parts == ["清脆的电子提示音，短促的 UI 点击反馈"]
    assert calls[1][1] == ["door knock"]
    # The generated scripts stay in the work dir for inspection.
    assert (tmp_path / "run" / "work" / "sfx_llm" / "select.py").is_file()

    # A fresh builder (rerun): the audio cache answers — no new chat calls.
    def _boom(*_a, **_k):
        raise AssertionError("chat must not run on a cache hit")

    monkeypatch.setattr(sfx, "chat", _boom)
    rerun = sfx.Sfx(tmp_path / "run2")
    rerun.cues = ["door knock", ""]
    assert rerun.event("select") == "res://assets/sfx/select.mp3"
    assert rerun.step_cue(0) == "res://assets/sfx/cue_0.mp3"
    assert rerun.generated == 2  # counted per run, served from cache


def test_unknown_event_and_out_of_range_cues_are_none(tmp_path):
    builder = sfx.Sfx(tmp_path)
    builder.cues = ["rain"]

    assert builder.event("nope") is None
    assert builder.step_cue(1) is None
    assert builder.step_cue(-1) is None
    assert builder.generated == 0


def test_chat_failure_disables_sfx_with_one_warning(tmp_path, monkeypatch, caplog):
    _settings(monkeypatch, tmp_path)

    def _boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(sfx, "chat", _boom)
    builder = sfx.Sfx(tmp_path / "run")

    with caplog.at_level("WARNING"):
        assert builder.event("select") is None
        assert builder.enabled is False
        assert builder.event("transition") is None

    assert sum("Sound effects disabled" in m for m in caplog.messages) == 1


def test_a_bad_script_drops_only_that_clip(tmp_path, monkeypatch, caplog):
    _settings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sfx,
        "chat",
        _fake_chat("import sys\nsys.exit(3)\n", _WAV_SCRIPT),
    )
    builder = sfx.Sfx(tmp_path / "run")
    builder.cues = ["bad cue", "good cue"]

    with caplog.at_level("WARNING"):
        assert builder.step_cue(0) is None

    # One bad clip never disables the layer — the next cue still renders.
    assert builder.enabled is True
    assert builder.step_cue(1) == "res://assets/sfx/cue_1.mp3"
    assert any("failed" in m for m in caplog.messages)


def test_a_cached_bad_script_is_refetched_once(tmp_path, monkeypatch):
    """A failing script in the response cache would fail every rerun — it is
    invalidated and refetched once, exactly like a cached analysis answer."""
    _settings(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        sfx,
        "chat",
        lambda *a, **k: (
            calls.append(k.get("refresh", False)),
            ChatResult("import sys\nsys.exit(3)\n", True, "k")
            if len(calls) == 1
            else ChatResult(_WAV_SCRIPT, False, "k"),
        )[1],
    )
    builder = sfx.Sfx(tmp_path / "run")

    assert builder.event("select") == "res://assets/sfx/select.mp3"
    assert calls == [False, True]  # cache read first, one forced refetch


def test_a_failing_fresh_script_is_not_repaid(tmp_path, monkeypatch):
    """A fresh answer that fails is never re-requested — that just re-pays
    for the same input (the analyzer's policy, mirrored here)."""
    _settings(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        sfx,
        "chat",
        lambda *a, **k: (
            calls.append(k.get("refresh", False)),
            ChatResult("import sys\nsys.exit(3)\n", False, "k"),
        )[1],
    )
    builder = sfx.Sfx(tmp_path / "run")

    assert builder.event("select") is None
    assert calls == [False]


def test_build_sfx_follows_the_provider_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)

    # No provider → off, no builder, no LLM call.
    monkeypatch.setattr(settings, "sfx_provider", "")
    assert sfx.build_sfx(tmp_path, _design()) is None

    # Unknown provider → skipped with a NOTICE, never an exception.
    monkeypatch.setattr(settings, "sfx_provider", "supercollider")
    assert sfx.build_sfx(tmp_path, _design()) is None

    # The code-writing provider builds a builder with derived cues.
    monkeypatch.setattr(settings, "sfx_provider", "llm")
    monkeypatch.setattr(sfx, "derive_cues", lambda design: ["", "咚咚咚"])
    built = sfx.build_sfx(tmp_path, _design())
    assert built is not None
    assert built.cues == ["", "咚咚咚"]
    assert built.enabled is True


def test_derive_cues_parses_normalizes_and_caps(monkeypatch):
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult('["", "door knock"]', False, "k"))
    assert sfx.derive_cues(_design(2)) == ["", "door knock"]

    # Length mismatch pads with silence.
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult('["only one"]', False, "k"))
    assert sfx.derive_cues(_design(3)) == ["only one", "", ""]

    # Object envelope is accepted too.
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult('{"cues": ["a", "b"]}', False, "k"))
    assert sfx.derive_cues(_design(2)) == ["a", "b"]

    # Unusable output → None, never an exception.
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult("no json here", False, "k"))
    assert sfx.derive_cues(_design(2)) is None

    # Chat raising → None.
    def _boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(sfx, "chat", _boom)
    assert sfx.derive_cues(_design(2)) is None


def test_derive_cues_caps_non_empty_entries(monkeypatch):
    n = sfx._MAX_CUES + 3
    answers = ",".join(f'"cue {i}"' for i in range(n))
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult(f"[{answers}]", False, "k"))

    cues = sfx.derive_cues(_design(n))

    assert cues is not None
    assert len(cues) == n
    assert sum(1 for c in cues if c) == sfx._MAX_CUES


# ── Cue handling: a sound description in, one clip out ──────────────────────


def test_clip_passes_the_cue_as_a_sound_to_render(tmp_path, monkeypatch):
    _settings(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        sfx, "chat", lambda *a, **k: calls.append(a) or ChatResult(_WAV_SCRIPT, False, "k")
    )
    builder = sfx.Sfx(tmp_path / "run")
    builder.cues = ["沉闷的关门声", "（哗哗的大雨）", "x" * 80]

    assert builder.step_cue(0) is not None
    assert builder.step_cue(1) is not None
    assert builder.step_cue(2) is not None

    sent = [parts for _system, parts in calls]
    # Brackets normalized away even when the LLM already wrapped it…
    assert sent[0] == ["沉闷的关门声"]
    assert sent[1] == ["哗哗的大雨"]
    # …and an overlong cue is bounded before it reaches the prompt.
    assert sent[2] == ["x" * sfx._MAX_CUE_CHARS]
