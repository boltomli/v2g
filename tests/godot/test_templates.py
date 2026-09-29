import json
import shutil
import subprocess
from pathlib import Path

import pytest

from v2g.config import settings
from v2g.godot import templates as T
from v2g.llm.analyzer import (
    Character,
    DialogueChoice,
    DialogueSample,
    GameDesign,
    SceneDesign,
)

_GODOT = shutil.which(settings.godot_path) or (
    settings.godot_path if Path(settings.godot_path).is_file() else None
)
requires_godot = pytest.mark.skipif(_GODOT is None, reason="godot not on PATH")


def _design() -> GameDesign:
    return GameDesign(
        title="Test Game",
        genre="visual novel",
        summary="s",
        narrative="",
        mechanics=["dialogue choices"],
        controls=["Space: advance"],
        style="cinematic",
        objects=[],
        scenes=[SceneDesign(name="Throne Room", description="d")],
        characters=[
            Character(name="Lady / The Bride", role="protagonist", visual="v"),
            Character(name="Guard Knight / Protector", role="companion", visual="v"),
        ],
        dialogue_samples=[
            DialogueSample(
                speaker="Guard Knight",
                line="¿Entonces será la carta de repudio?",
                line_zh="那就会招来一纸休书。",
                context="opening",
                choices=[
                    DialogueChoice(line="", line_zh="把一切都告诉他", score=2),
                    DialogueChoice(line="", line_zh="先观察再开口", score=1),
                ],
            ),
            DialogueSample(speaker="", line="", line_zh="她悄然回到自己的房间。"),
        ],
    )


def test_project_input_is_advance_only():
    text = T.project_dot_godot("My Game")

    assert "advance={" in text
    for removed in ("move_left", "move_right", "jump", "interact"):
        assert removed not in text


def test_main_scene_uid_is_quoted_and_vn_shaped():
    scripts = {"vn_manager.gd": "extends Control", "game_manager.gd": "extends Node"}
    text = T.main_scene(_design(), scripts)

    assert 'uid="uid://' in text
    assert 'type="Control"' in text
    assert "CharacterBody2D" not in text
    assert "res://vn_manager.gd" in text
    assert "res://game_manager.gd" in text


def test_vn_story_embeds_verbatim_source_and_chinese_only_ui():
    design = _design()
    assets = {
        "background": Path("/x/background.png"),
        "scenes/throne_room": Path("/x/scene_throne_room.png"),
        "characters/lady___the_bride": Path("/x/char_lady___the_bride.png"),
    }
    script = T.vn_manager_script(design, assets)

    # Source-language line verbatim from the design (video transcript stand-in)
    assert "¿Entonces será la carta de repudio?" in script
    # Chinese subtitle and narration present
    assert "那就会招来一纸休书。" in script
    assert "她悄然回到自己的房间。" in script
    # Assets only for entries that exist
    assert "res://assets/background.png" in script
    assert "res://assets/scene_throne_room.png" in script
    assert "res://assets/char_lady___the_bride.png" in script
    # Template-authored UI strings are Chinese-only (language contract)
    assert "声望:" in script
    assert "空格 / 回车 / 点击 继续" in script
    assert "Espacio" not in script and "avanzar" not in script
    # No template-invented foreign-language narration: story JSON zh-only for narration
    assert "Reputación" not in script


def test_game_manager_template_keeps_signal_contract():
    source = T.game_manager_script(_design())

    assert "signal score_changed(new_score: int)" in source
    assert "func add_score" in source


def test_llm_prompt_forbids_template_owned_files():
    prompt = T.LLM_SCRIPT_SYSTEM

    assert "vn_manager.gd" in prompt
    assert "NEVER generate" in prompt
    assert "score_changed" in prompt
    assert '"advance"' in prompt


def test_viewport_follows_video_aspect():
    """Window matches the source video so the frame fills edge to edge."""
    assert T.viewport_for((1920, 1080)) == (1280, 720)  # landscape unchanged
    assert T.viewport_for((1080, 1922)) == (720, 1280)  # source video: portrait
    assert T.viewport_for(None) == (1280, 720)  # unknown → default
    assert T.viewport_for((0, 1080)) == (1280, 720)  # invalid → default
    # Extreme aspect ratio: usable window wins over exact match (letterbox there)
    assert T.viewport_for((8000, 1000)) == (1280, 480)


# ── Embedded JSON must survive GDScript's own escape pass ────────────────────


def _gdscript_unescape(body: str) -> str:
    """Decode a GDScript string body the way the engine's parser does.

    Only the two escapes :func:`T._gd_const` is allowed to emit are understood;
    anything else raises, so a future change that lets a raw ``\\t`` or an
    unescaped quote through fails loudly instead of silently mangling a story.
    """
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if nxt == "\\":
            out.append("\\")
        elif nxt == '"':
            out.append('"')
        else:
            raise AssertionError(f"unescaped GDScript sequence {ch + nxt!r} in embedded JSON")
        i += 2
    return "".join(out)


def _const_body(script: str, name: str) -> str:
    head = f'const {name} := """'
    start = script.index(head) + len(head)
    return script[start : script.index('"""', start)]


def _decoded_const(script: str, name: str):
    """One embedded JSON const as the game's runtime sees it.

    Assertions about embedded *data* have to go through this: the literal is
    escaped for GDScript, so its raw text carries ``\\"`` where the JSON has
    ``"`` and matching on the script source would test the escaping, not the
    value that reaches ``JSON.parse_string``.
    """
    return json.loads(_gdscript_unescape(_const_body(script, name)))


def _quote_heavy_design() -> GameDesign:
    """Story text carrying exactly what a naive embed mangles."""
    return GameDesign(
        title='Test "Game"',
        genre="vn",
        summary="s",
        mechanics=[],
        controls=[],
        style="st",
        objects=[],
        scenes=[],
        characters=[Character(name='Lady "Bee"', role="protagonist", visual="v")],
        dialogue_samples=[
            DialogueSample(speaker="Lady", line='He said "hello"', line_zh='他说"你好"。'),
            DialogueSample(speaker="", line="", line_zh="A path: C:\\temp\\new"),
            DialogueSample(speaker="", line="", line_zh="tab\there"),
        ],
    )


def test_embedded_story_json_survives_gdscript_escaping():
    """Regression: raw ``json.dumps`` output inside a GDScript triple-quoted
    string is rewritten by the engine's escape pass before it is ever parsed —
    an ASCII quote in one line closed the JSON string early and
    ``JSON.parse_string`` returned null, so the whole game booted to the
    "（无剧本）" card while logging no error at all."""
    design = _quote_heavy_design()
    script = T.vn_manager_script(design, {})

    for name in ("STORY_JSON", "TEX_JSON", "PORTRAIT_JSON", "SPEAKER_JSON", "SFX_JSON"):
        assert isinstance(_decoded_const(script, name), (list, dict))

    story = _decoded_const(script, "STORY_JSON")
    lines = {step["es"] for step in story if step.get("es")}
    zh = {step["zh"] for step in story if step.get("zh")}
    assert 'He said "hello"' in lines
    assert 'Test "Game"' in lines
    assert '他说"你好"。' in zh
    assert "A path: C:\\temp\\new" in zh  # backslashes survive too
    assert "tab\there" in zh


def test_const_body_never_terminates_its_own_literal():
    """No embedded value may close the triple-quoted string it lives in."""
    script = T.vn_manager_script(_quote_heavy_design(), {})
    body = _const_body(script, "STORY_JSON")
    # Every quote is escaped, so `"""` cannot occur inside the body.
    assert '"""' not in body
    assert body.count('"') == body.count('\\"')


@requires_godot
def test_godot_parses_the_embedded_story(tmp_path):
    """The same contract, checked by Godot's own JSON parser rather than ours."""
    design = _quote_heavy_design()
    (tmp_path / "project.godot").write_text(T.project_dot_godot("probe"), encoding="utf-8")
    script = T.vn_manager_script(design, {})
    (tmp_path / "vn_manager.gd").write_text(script, encoding="utf-8")

    consts = "\n".join(ln for ln in script.splitlines() if ln.startswith("const "))
    (tmp_path / "check.gd").write_text(
        "extends SceneTree\n\n"
        + consts
        + """

func _init() -> void:
	for name in ["STORY_JSON", "TEX_JSON", "PORTRAIT_JSON", "SPEAKER_JSON"]:
		var raw = get_script().get_script_constant_map()[name]
		if JSON.parse_string(raw) == null:
			print("PARSE_FAILED ", name)
		else:
			print("PARSED ", name)
	quit()
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["godot", "--headless", "--path", str(tmp_path), "--script", "res://check.gd"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "PARSE_FAILED" not in output, output
    for name in ("STORY_JSON", "TEX_JSON", "PORTRAIT_JSON", "SPEAKER_JSON"):
        assert f"PARSED {name}" in output, output


# ── Voice-over + background music ────────────────────────────────────────────


class _FakeSynth:
    """Stands in for v2g.tts.Synthesizer: records what would be spoken."""

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []

    def speak(self, text: str, speaker_key: str) -> str | None:
        if not text.strip():
            return None
        self.spoken.append((text, speaker_key))
        return f"res://assets/voice/{len(self.spoken):02d}.mp3"


def test_vn_story_voices_title_dialogue_narration_but_never_the_end_card():
    synth = _FakeSynth()
    script = T.vn_manager_script(_design(), None, synth=synth)

    spoken = dict(synth.spoken)
    assert "Test Game" in spoken  # title, narrator voice
    assert spoken["Test Game"] == ""
    assert "那就会招来一纸休书。" in spoken  # dialogue speaks the Chinese line
    assert spoken["那就会招来一纸休书。"] != ""  # keyed by character, not narrator
    assert "她悄然回到自己的房间。" in spoken  # narration, narrator voice
    assert spoken["她悄然回到自己的房间。"] == ""
    # The end card is UI, not story — it must never be synthesized.
    assert not any("完 · 按空格" in text for text, _ in synth.spoken)
    # The runtime reads the embedded paths.
    assert any(
        str(step.get("voice", "")).startswith("res://assets/voice/")
        for step in _decoded_const(script, "STORY_JSON")
    )
    assert "_voice.play()" in script


def test_vn_story_without_synth_has_no_voice_paths_but_keeps_player_wiring():
    script = T.vn_manager_script(_design())

    assert not any("voice" in step for step in _decoded_const(script, "STORY_JSON"))
    assert "AudioStreamPlayer.new()" in script  # players exist either way
    assert 'var voice_res: String = step.get("voice", "")' in script


# ── Transcript-less runs: dialogue lives in line_zh alone ───────────────────


def _transcriptless_design() -> GameDesign:
    """A video with no subtitles: every `line` is empty, `line_zh` carries
    both narration and dialogue (the analyzer's no-transcript contract)."""
    return GameDesign(
        title="T",
        genre="vn",
        summary="s",
        mechanics=[],
        controls=["Space"],
        style="cinematic",
        objects=[],
        scenes=[SceneDesign(name="Stone Corridor", description="d")],
        characters=[Character(name="Guard Knight", role="companion", visual="v")],
        dialogue_samples=[
            DialogueSample(speaker="", line="", line_zh="夜色落下。"),
            DialogueSample(speaker="Guard Knight", line="", line_zh="跟我来。"),
            DialogueSample(speaker="Guard Knight", line="", line_zh=""),  # degenerate
        ],
    )


def test_transcriptless_dialogue_still_keys_speaker_sprite_name_and_voice():
    """Regression: keying on `line` demoted a whole subtitle-less cast to the
    narrator — no sprites, no name labels, one voice for everyone."""
    design = _transcriptless_design()
    assets = {"characters/guard_knight": Path("/x/char_guard_knight.png")}
    synth = _FakeSynth()

    steps, _, _portraits, names = T._build_story(design, assets, synth=synth)

    dialogue = steps[2]  # steps[0] is the title card
    assert dialogue["speaker"] == "guard_knight"
    assert dialogue["sprite"] == "guard_knight"
    assert names["guard_knight"] == "Guard Knight"
    assert names[""] == "旁白"
    # Narration keys to the narrator; the speaker row keys to its character.
    spoken = dict(synth.spoken)
    assert spoken["夜色落下。"] == ""
    assert spoken["跟我来。"] == "guard_knight"
    # A speaker row with no text anywhere is still narration, not a name label.
    assert steps[3]["speaker"] == ""


def test_voice_design_derives_keys_for_transcriptless_speakers(monkeypatch):
    """The voice-design call must receive the character keys too — otherwise
    every speaker borrows the narrator's card."""
    import json

    from v2g import tts
    from v2g.llm.client import ChatResult

    captured = {}

    def _chat(_system, parts, **_kw):
        captured.update(json.loads(parts[0]))
        return ChatResult('{"voices": {"": "旁白卡", "guard_knight": "卫兵卡"}}', False, "k")

    monkeypatch.setattr(tts, "chat", _chat)
    designs = tts.derive_voice_designs(_transcriptless_design())

    assert captured["speaker_keys"] == ["", "guard_knight"]
    assert designs is not None
    assert designs["guard_knight"] == "卫兵卡"


def test_vn_story_speaks_the_source_line_in_source_mode(monkeypatch):
    monkeypatch.setattr(settings, "tts_text", "source")
    synth = _FakeSynth()

    T.vn_manager_script(_design(), None, synth=synth)

    spoken = dict(synth.spoken)
    assert "¿Entonces será la carta de repudio?" in spoken  # verbatim source line
    assert "那就会招来一纸休书。" not in spoken
    # Narration has no source line — Chinese regardless of mode.
    assert "她悄然回到自己的房间。" in spoken


def test_voice_text_falls_back_across_languages(monkeypatch):
    assert settings.tts_text == "zh"  # default mode: Chinese first
    assert T._voice_text("hola", "你好") == "你好"
    assert T._voice_text("", "旁白") == "旁白"
    assert T._voice_text("标题", "") == "标题"  # title card: zh empty → es

    monkeypatch.setattr(settings, "tts_text", "source")
    assert T._voice_text("hola", "你好") == "hola"
    assert T._voice_text("", "旁白") == "旁白"
    assert T._voice_text("标题", "") == "标题"


def test_vn_story_embeds_bgm_track_and_loops_it():
    script = T.vn_manager_script(_design(), None, bgm="res://assets/bgm/bgm.mp3")

    assert 'const BGM_RES := "res://assets/bgm/bgm.mp3"' in script
    assert "mp3.loop = true" in script
    assert "_bgm.play()" in script
    assert "_start_bgm()" in script


def test_vn_story_without_bgm_keeps_a_silent_player():
    script = T.vn_manager_script(_design())

    assert 'const BGM_RES := ""' in script
    # Guard stays in the source: an empty path must never reach load().
    assert 'if BGM_RES == "":' in script


class _FakeSfx:
    """Stands in for v2g.sfx.Sfx: cue only for the first sample, one event."""

    def step_cue(self, index: int) -> str | None:
        return "res://assets/sfx/cue_0.mp3" if index == 0 else None

    def event(self, name: str) -> str | None:
        if name == "select":
            return "res://assets/sfx/select.mp3"
        return None  # transition failed to synthesize


def test_vn_story_embeds_sfx_cues_and_event_map():
    script = T.vn_manager_script(_design(), None, sfx=_FakeSfx())

    # Per-step cue lands on the first sample's step, not the title/end card.
    assert "res://assets/sfx/cue_0.mp3" in {
        str(step.get("sfx", "")) for step in _decoded_const(script, "STORY_JSON")
    }
    # Event map: only the clips that were generated appear.
    events = _decoded_const(script, "SFX_JSON")
    assert events == {"select": "res://assets/sfx/select.mp3"}
    # Runtime wiring.
    assert "const SFX_JSON :=" in script
    assert "func _play_sfx" in script
    assert "_play_sfx(cue_res)" in script
    assert 'var select_res: String = sfx_map.get("select", "")' in script
    assert "_play_sfx(select_res)" in script


def test_vn_story_without_sfx_has_an_empty_event_map():
    script = T.vn_manager_script(_design())

    assert 'const SFX_JSON := """{}"""' in script
    assert not any("sfx" in step for step in _decoded_const(script, "STORY_JSON"))
