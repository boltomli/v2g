from pathlib import Path

from v2g.godot import templates as T
from v2g.llm.analyzer import (
    Character,
    DialogueChoice,
    DialogueSample,
    GameDesign,
    SceneDesign,
)


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
