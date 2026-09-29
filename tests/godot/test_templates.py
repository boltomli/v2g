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

    for name in ("STORY_JSON", "TEX_JSON", "PORTRAIT_JSON", "SPEAKER_JSON"):
        decoded = json.loads(_gdscript_unescape(_const_body(script, name)))
        assert isinstance(decoded, (list, dict))

    story = json.loads(_gdscript_unescape(_const_body(script, "STORY_JSON")))
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
