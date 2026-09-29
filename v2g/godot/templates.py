"""GDScript and Godot project file templates.

The generated game is a visual novel. Template-owned files are project.godot,
main.tscn, and the VN runtime (vn_manager.gd) — the story runtime embeds the
bilingual script built from the GameDesign. Other scripts come from the LLM
with template fallbacks.

Language contract: Chinese is the target language; every non-Chinese string in
generated game text originates from the source video (verbatim transcript),
never from template invention — template-authored UI strings are Chinese-only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from v2g.config import settings
from v2g.llm.analyzer import GameDesign, safe_name, speaker_key

if TYPE_CHECKING:
    from v2g.sfx import Sfx
    from v2g.tts import Synthesizer

log = logging.getLogger(__name__)

# ── Asset resolution (keys come from v2g.video.asset_extractor via safe_name) ─


def _res(assets: dict[str, Path] | None, key: str) -> str | None:
    path = (assets or {}).get(key)
    return f"res://assets/{path.name}" if path else None


# ── project.godot ────────────────────────────────────────────────────────────


def project_dot_godot(title: str, w: int = 1280, h: int = 720) -> str:
    return f"""\
; Engine configuration file.
; It's best edited using the editor UI and not directly.

[application]

config/name="{title}"
run/main_scene="res://main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")

[display]

window/size/viewport_width={w}
window/size/viewport_height={h}

[input]

advance={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":32,"key_label":0,"unicode":32,"location":0,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194320,"key_label":0,"unicode":0,"location":0,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194309,"key_label":0,"unicode":0,"location":0,"echo":false,"script":null)
]
}}

[rendering]

renderer/rendering_method="gl_compatibility"
"""


def viewport_for(video_size: tuple[int, int] | None, long_side: int = 1280) -> tuple[int, int]:
    """Window size matching the source video's aspect ratio (long side fixed).

    Unknown/invalid size falls back to the 1280x720 default. The short side is
    clamped to 480 px so extreme aspect ratios still get a usable window
    (letterboxing only there — matched aspects fill edge to edge).
    """
    if not video_size:
        return (long_side, 720)
    vw, vh = video_size
    if vw <= 0 or vh <= 0:
        return (long_side, 720)
    short = round(long_side * min(vw, vh) / max(vw, vh) / 2.0) * 2
    short = max(short, 480)
    return (long_side, short) if vw >= vh else (short, long_side)


# ── UID generator ────────────────────────────────────────────────────────────


def _uid() -> str:
    """Deterministic uid; Godot just needs uniqueness within a file."""
    _uid.counter += 1
    return f"uid://v2g{_uid.counter:04d}"


_uid.counter = 0


# ── Story building (GameDesign → VN steps) ──────────────────────────────────


def _voice_text(es: str, zh: str) -> str:
    """Which text a step's voice-over speaks (``V2G_TTS_TEXT``).

    Default ``zh``: the Chinese line (narration is Chinese anyway, ``line_zh``
    is always present, and one language keeps voice languages consistent).
    ``source`` speaks the verbatim transcript line, falling back to Chinese
    when there is none (narration, title card).
    """
    es, zh = es.strip(), zh.strip()
    if settings.tts_text == "source":
        return es or zh
    return zh or es


def _build_story(
    design: GameDesign,
    assets: dict[str, Path] | None = None,
    synth: Synthesizer | None = None,
    sfx: Sfx | None = None,
) -> tuple[list[dict], dict[str, str], dict[str, str], dict[str, str]]:
    """Build (story_steps, bg_textures, portraits, speaker_names) for the VN runtime.

    Story rules:
    - `es` holds ONLY verbatim source-video lines (empty for narration);
    - `zh` holds the Chinese translation / original Chinese narration;
    - choice options follow the same rule (Chinese-only unless verbatim);
    - background/portrait entries are only emitted for assets that exist;
    - with *synth*, every story step (not the end card) gets a `voice` res
      path when synthesis succeeds — a missing key means no audio, never an
      error;
    - with *sfx*, a step gains an `sfx` res path when its derived cue
      synthesized (title and end card never carry cues).
    """
    # Backgrounds in play order: main background, then scene stills
    bg_order: list[str] = []
    tex_map: dict[str, str] = {}
    bg = _res(assets, "background")
    if bg:
        bg_order.append("background")
        tex_map["background"] = bg
    for scene in design.scenes:
        sid = f"scene_{safe_name(scene.name)}"
        res = _res(assets, f"scenes/{safe_name(scene.name)}")
        if res:
            bg_order.append(sid)
            tex_map[sid] = res

    portrait_map: dict[str, str] = {}
    for c in design.characters:
        res = _res(assets, f"characters/{safe_name(c.name)}")
        if res:
            portrait_map[safe_name(c.name)] = res

    speaker_names: dict[str, str] = {"": "旁白"}
    steps: list[dict] = []
    n = len(design.dialogue_samples)

    if design.title:
        title_step: dict = {"speaker": "", "es": design.title, "zh": ""}
        if bg_order:
            title_step["bg"] = bg_order[0]
        if synth is not None:
            title_voice = synth.speak(_voice_text(design.title, ""), "")
            if title_voice:
                title_step["voice"] = title_voice
        steps.append(title_step)

    for i, ds in enumerate(design.dialogue_samples):
        step: dict = {}
        if bg_order:
            step["bg"] = bg_order[min(i * len(bg_order) // max(n, 1), len(bg_order) - 1)]
        key = speaker_key(ds, design.characters)
        step["speaker"] = key
        # Dialogue row (speaker + text in either language — a transcript-less
        # run carries it in line_zh alone): it may still key to "" for a
        # degenerate name — setdefault/sprite checks then no-op as before.
        has_text = bool(ds.line.strip() or ds.line_zh.strip())
        if ds.speaker.strip() and has_text:
            speaker_names.setdefault(key, ds.speaker)
            if key in portrait_map:
                step["sprite"] = key
            if not ds.line_zh.strip():
                log.warning("Dialogue missing line_zh (Chinese subtitle): %r", ds.line[:80])
        step["es"] = ds.line
        step["zh"] = ds.line_zh
        if ds.choices:
            step["choices"] = [
                {"es": c.line, "zh": c.line_zh, "score": c.score} for c in ds.choices
            ]
        if synth is not None:
            # step["speaker"] is "" for narration — the narrator voice.
            step_voice = synth.speak(_voice_text(step["es"], step["zh"]), step["speaker"])
            if step_voice:
                step["voice"] = step_voice
        if sfx is not None:
            cue_path = sfx.step_cue(i)
            if cue_path:
                step["sfx"] = cue_path
        steps.append(step)

    steps.append({"speaker": "", "es": "", "zh": "完 · 按空格或点击重新开始"})
    return steps, tex_map, portrait_map, speaker_names


def _gd_const(name: str, value: str) -> str:
    """A GDScript ``const NAME := \"\"\"…\"\"\"`` whose *value* is exactly *value*.

    GDScript resolves escape sequences inside its string literals *before* the
    engine ever sees the text, so raw JSON pasted into a triple-quoted string is
    silently rewritten on the way in: JSON's ``\\"`` (an escaped quote) becomes a
    bare quote that ends the JSON string, and JSON's ``\\\\`` (an escaped
    backslash) collapses to one backslash. Measured on Godot 4.7: a story line
    containing an ASCII quote made ``JSON.parse_string`` return null, so the
    game booted to the empty-story card "（无剧本）" while reporting no error at
    all — and a Windows-looking path in any field lost its separators.

    Escaping backslashes first and quotes second yields a literal that decodes
    back to the original text, so no story content can reach the runtime
    mangled. ``json.dumps`` never emits a raw newline, so every control
    character is already backslash-escaped and covered by the first pass.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'const {name} := """{escaped}"""'


def vn_manager_script(
    design: GameDesign,
    assets: dict[str, Path] | None = None,
    *,
    synth: Synthesizer | None = None,
    bgm: str | None = None,
    sfx: Sfx | None = None,
) -> str:
    """Generate the template-owned visual-novel runtime (vn_manager.gd).

    The bilingual story and asset maps are embedded as JSON and parsed at
    runtime. Chinese is the only template-authored UI language.

    *synth* synthesizes each step's voice-over (paths land in the story JSON's
    ``voice`` keys), *sfx* adds per-step cues plus the fixed event clips
    (``SFX_JSON``), and *bgm* is the ``res://`` path of the loopable
    background track (None = no music).
    """
    story, tex_map, portrait_map, speaker_names = _build_story(design, assets, synth, sfx)
    bgm_const = json.dumps(bgm or "")
    events: dict[str, str] = {}
    if sfx is not None:
        for event in ("select", "transition"):
            path = sfx.event(event)
            if path:
                events[event] = path
    consts = "\n".join(
        _gd_const(name, json.dumps(value, ensure_ascii=False))
        for name, value in (
            ("STORY_JSON", story),
            ("TEX_JSON", tex_map),
            ("PORTRAIT_JSON", portrait_map),
            ("SPEAKER_JSON", speaker_names),
            ("SFX_JSON", events),
        )
    )

    return f"""\
extends Control

# Visual-novel runtime (template-owned — do not edit via LLM generation).
# Story data is embedded JSON built from the GameDesign:
#   - `es` lines are verbatim transcripts of the source video (may be empty),
#   - `zh` lines are the target language (Simplified Chinese subtitles/narration).

{consts}
const BGM_RES := {bgm_const}

var story: Array = []
var tex_map: Dictionary = {{}}
var portrait_map: Dictionary = {{}}
var speaker_names: Dictionary = {{}}
var sfx_map: Dictionary = {{}}

var game_manager: Node
var _idx := 0
var _cur_bg := ""
var _started := false
var _awaiting_choice := false

var _bg: TextureRect
var _black: ColorRect
var _portrait: TextureRect
var _name_lbl: Label
var _es_lbl: Label
var _zh_lbl: Label
var _score_lbl: Label
var _hint_lbl: Label
var _choices: VBoxContainer
var _voice: AudioStreamPlayer
var _bgm: AudioStreamPlayer
var _sfx: AudioStreamPlayer
var ui_font: SystemFont

func _ready() -> void:
    set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    # Clicks fall through to _unhandled_input; only choice buttons intercept.
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    story = JSON.parse_string(STORY_JSON)
    tex_map = JSON.parse_string(TEX_JSON)
    portrait_map = JSON.parse_string(PORTRAIT_JSON)
    speaker_names = JSON.parse_string(SPEAKER_JSON)
    sfx_map = JSON.parse_string(SFX_JSON)
    if sfx_map == null:
        sfx_map = {{}}
    if story == null:
        story = []
    ui_font = SystemFont.new()
    ui_font.font_names = PackedStringArray([
        "Microsoft YaHei", "Microsoft YaHei UI", "PingFang SC", "SimHei",
    ])
    _build_ui()
    _start_bgm()
    game_manager = get_node("GameManager")
    game_manager.score_changed.connect(
        func(v: int) -> void: _score_lbl.text = "声望: %d" % v)
    if story.is_empty():
        _zh_lbl.text = "（无剧本）"
    else:
        _show_step(0)

func _build_ui() -> void:
    _bg = TextureRect.new()
    _bg.name = "Background"
    add_child(_bg)
    _bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    _bg.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
    _bg.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
    _bg.mouse_filter = Control.MOUSE_FILTER_IGNORE

    _black = ColorRect.new()
    _black.name = "Black"
    add_child(_black)
    _black.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    _black.color = Color(0, 0, 0, 1)
    _black.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _black.modulate = Color(1, 1, 1, 0)

    _portrait = TextureRect.new()
    _portrait.name = "Portrait"
    add_child(_portrait)
    _portrait.anchor_left = 1.0
    _portrait.anchor_right = 1.0
    _portrait.anchor_top = 0.0
    _portrait.anchor_bottom = 1.0
    _portrait.offset_left = -470.0
    _portrait.offset_right = 0.0
    _portrait.offset_top = 0.0
    _portrait.offset_bottom = 0.0
    _portrait.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
    _portrait.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
    _portrait.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _portrait.visible = false

    _score_lbl = Label.new()
    _score_lbl.name = "Score"
    add_child(_score_lbl)
    _score_lbl.offset_left = 18.0
    _score_lbl.offset_top = 14.0
    _score_lbl.offset_right = 420.0
    _score_lbl.offset_bottom = 44.0
    _score_lbl.text = "声望: 0"
    _score_lbl.add_theme_font_override("font", ui_font)
    _score_lbl.add_theme_font_size_override("font_size", 18)
    _score_lbl.add_theme_color_override("font_color", Color(0.91, 0.77, 0.42))
    _score_lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE

    _hint_lbl = Label.new()
    _hint_lbl.name = "Hint"
    add_child(_hint_lbl)
    _hint_lbl.anchor_left = 1.0
    _hint_lbl.anchor_right = 1.0
    _hint_lbl.offset_left = -640.0
    _hint_lbl.offset_right = -18.0
    _hint_lbl.offset_top = 14.0
    _hint_lbl.offset_bottom = 44.0
    _hint_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
    _hint_lbl.text = "空格 / 回车 / 点击 继续"
    _hint_lbl.add_theme_font_override("font", ui_font)
    _hint_lbl.add_theme_font_size_override("font_size", 16)
    _hint_lbl.add_theme_color_override("font_color", Color(1, 1, 1, 0.75))
    _hint_lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE

    var box := ColorRect.new()
    box.name = "Box"
    add_child(box)
    box.anchor_left = 0.0
    box.anchor_right = 1.0
    box.anchor_top = 1.0
    box.anchor_bottom = 1.0
    box.offset_top = -205.0
    box.color = Color(0.03, 0.03, 0.05, 0.84)
    box.mouse_filter = Control.MOUSE_FILTER_IGNORE

    var margin := MarginContainer.new()
    box.add_child(margin)
    margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    margin.add_theme_constant_override("margin_left", 26)
    margin.add_theme_constant_override("margin_right", 26)
    margin.add_theme_constant_override("margin_top", 16)
    margin.add_theme_constant_override("margin_bottom", 18)
    margin.mouse_filter = Control.MOUSE_FILTER_IGNORE

    var vbox := VBoxContainer.new()
    margin.add_child(vbox)
    vbox.add_theme_constant_override("separation", 6)
    vbox.mouse_filter = Control.MOUSE_FILTER_IGNORE

    _name_lbl = Label.new()
    _name_lbl.name = "SpeakerName"
    vbox.add_child(_name_lbl)
    _name_lbl.add_theme_font_override("font", ui_font)
    _name_lbl.add_theme_font_size_override("font_size", 26)
    _name_lbl.add_theme_color_override("font_color", Color(0.91, 0.77, 0.42))
    _name_lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE

    _es_lbl = Label.new()
    _es_lbl.name = "LineES"
    vbox.add_child(_es_lbl)
    _es_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    _es_lbl.add_theme_font_override("font", ui_font)
    _es_lbl.add_theme_font_size_override("font_size", 23)
    _es_lbl.add_theme_color_override("font_color", Color(0.96, 0.96, 0.94))
    _es_lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE

    _zh_lbl = Label.new()
    _zh_lbl.name = "LineZH"
    vbox.add_child(_zh_lbl)
    _zh_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    _zh_lbl.add_theme_font_override("font", ui_font)
    _zh_lbl.add_theme_font_size_override("font_size", 23)
    _zh_lbl.add_theme_color_override("font_color", Color(1.0, 0.85, 0.47))
    _zh_lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE

    _choices = VBoxContainer.new()
    _choices.name = "Choices"
    add_child(_choices)
    _choices.anchor_left = 0.15
    _choices.anchor_right = 0.85
    _choices.anchor_top = 0.0
    _choices.anchor_bottom = 0.0
    _choices.offset_top = 100.0
    _choices.offset_bottom = 460.0
    _choices.add_theme_constant_override("separation", 14)
    _choices.alignment = BoxContainer.ALIGNMENT_CENTER
    _choices.mouse_filter = Control.MOUSE_FILTER_IGNORE
    _choices.visible = false

    _voice = AudioStreamPlayer.new()
    _voice.name = "Voice"
    add_child(_voice)

    _bgm = AudioStreamPlayer.new()
    _bgm.name = "BGM"
    add_child(_bgm)
    _bgm.volume_db = -6.0  # the voice-over sits on top of the music

    _sfx = AudioStreamPlayer.new()
    _sfx.name = "SFX"
    add_child(_sfx)

func _start_bgm() -> void:
    if BGM_RES == "":
        return
    var stream := load(BGM_RES) as AudioStream
    if stream == null:
        return
    if stream is AudioStreamMP3:
        var mp3 := stream as AudioStreamMP3
        mp3.loop = true
    _bgm.stream = stream
    _bgm.play()

func _play_sfx(res_path: String) -> void:
    if res_path == "":
        return
    var stream := load(res_path) as AudioStream
    if stream == null:
        return
    _sfx.stream = stream
    _sfx.play()

func _flash() -> void:
    _black.modulate = Color(1, 1, 1, 0.55)
    var tw := create_tween()
    tw.tween_property(_black, "modulate:a", 0.0, 0.35)
    tw.set_trans(Tween.TRANS_QUAD)
    tw.set_ease(Tween.EASE_OUT)

func _show_step(i: int) -> void:
    var step: Dictionary = story[i]
    if step.has("bg"):
        var bg_id: String = step["bg"]
        if bg_id != _cur_bg:
            _cur_bg = bg_id
            if tex_map.has(bg_id):
                _bg.texture = load(tex_map[bg_id])
            if _started:
                _flash()
                var trans_res: String = sfx_map.get("transition", "")
                if trans_res != "":
                    _play_sfx(trans_res)
    _started = true

    var sp: String = step.get("sprite", "")
    if sp != "" and portrait_map.has(sp):
        _portrait.texture = load(portrait_map[sp])
        _portrait.visible = true
    else:
        _portrait.visible = false
        _portrait.texture = null

    var spk: String = step.get("speaker", "")
    _name_lbl.text = speaker_names.get(spk, spk)
    _es_lbl.text = step.get("es", "")
    _zh_lbl.text = step.get("zh", "")

    _voice.stop()
    var voice_res: String = step.get("voice", "")
    if voice_res != "":
        var vstream := load(voice_res) as AudioStream
        if vstream != null:
            _voice.stream = vstream
            _voice.play()

    var cue_res: String = step.get("sfx", "")
    if cue_res != "":
        _play_sfx(cue_res)

    _clear_choices()
    _awaiting_choice = false
    if step.has("choices"):
        _awaiting_choice = true
        var i_ch := 0
        for c: Dictionary in step["choices"]:
            var btn := Button.new()
            var es: String = c.get("es", "")
            var zh: String = c.get("zh", "")
            btn.text = zh if es.is_empty() else es + "\\n" + zh
            btn.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
            btn.add_theme_font_override("font", ui_font)
            btn.add_theme_font_size_override("font_size", 21)
            btn.pressed.connect(_on_choice.bind(i_ch))
            _choices.add_child(btn)
            i_ch += 1
        _choices.visible = true

func _clear_choices() -> void:
    for c in _choices.get_children():
        _choices.remove_child(c)
        # Deferred: this runs during pressed emission — an immediate free would
        # free the emitting button and trip the object lock.
        c.queue_free()

func _on_choice(i: int) -> void:
    if not _awaiting_choice:
        return
    _awaiting_choice = false
    _clear_choices()
    _choices.visible = false
    var select_res: String = sfx_map.get("select", "")
    if select_res != "":
        _play_sfx(select_res)
    var chosen: Dictionary = story[_idx]["choices"][i]
    game_manager.add_score(int(chosen.get("score", 1)))
    _advance_to(_idx + 1)

func advance() -> void:
    if _awaiting_choice:
        return
    if _idx >= story.size() - 1:
        get_tree().reload_current_scene()
        return
    _advance_to(_idx + 1)

func _advance_to(i: int) -> void:
    _voice.stop()
    _idx = i
    _show_step(_idx)

func _unhandled_input(event: InputEvent) -> void:
    if event.is_action_pressed("advance"):
        advance()
    elif event is InputEventMouseButton and event.pressed \\
            and event.button_index == MOUSE_BUTTON_LEFT:
        advance()
"""


# ── Scene (.tscn) ────────────────────────────────────────────────────────────


def main_scene(
    design: GameDesign | None = None,
    scripts: dict[str, str] | None = None,
) -> str:
    """Generate the visual-novel main.tscn.

    Minimal, template-owned tree:
      Main (Control, full rect, vn_manager.gd)
        GameManager (Node, game_manager.gd)
    All rendering (backgrounds, portraits, dialogue box, choices) is built by
    the vn_manager at runtime.
    """
    uid_main = _uid()

    ext_resources: list[str] = []
    script_id_map: dict[str, str] = {}
    for fname in ("vn_manager.gd", "game_manager.gd"):
        if scripts and fname in scripts:
            eid = f"{len(ext_resources) + 1}_{_uid()}"
            ext_resources.append(f'[ext_resource type="Script" path="res://{fname}" id="{eid}"]')
            script_id_map[fname] = eid

    load_steps = len(ext_resources) + 1

    nodes: list[str] = ['[node name="Main" type="Control"]']
    nodes.append("anchor_right = 1.0")
    nodes.append("anchor_bottom = 1.0")
    nodes.append("grow_horizontal = 2")
    nodes.append("grow_vertical = 2")
    if "vn_manager.gd" in script_id_map:
        nodes.append(f'script = ExtResource("{script_id_map["vn_manager.gd"]}")')
    nodes.append("")
    if "game_manager.gd" in script_id_map:
        nodes.append('[node name="GameManager" type="Node" parent="."]')
        nodes.append(f'script = ExtResource("{script_id_map["game_manager.gd"]}")')
        nodes.append("")
    node_block = "\n".join(nodes).rstrip()

    ext_block = "\n".join(ext_resources)
    # Godot expects the scene uid quoted: uid="uid://..."
    return f"""\
[gd_scene load_steps={load_steps} format=3 uid="{uid_main}"]

{ext_block}

{node_block}
"""


# ── Fallback GDScript generators ─────────────────────────────────────────────
# Used ONLY if the LLM script generation fails.


def game_manager_script(design: GameDesign) -> str:
    """Fallback: basic score + goals game manager (signal contract is required)."""
    goals = ", ".join(f'"{g}"' for scene in design.scenes for g in scene.goals)
    return f"""\
extends Node

signal score_changed(new_score: int)
signal game_over

var score: int = 0
var goals: Array = [{goals}]

func add_score(amount: int = 1) -> void:
    score += amount
    score_changed.emit(score)

func end_game() -> void:
    game_over.emit()
    print("Game Over! Final score: ", score)
"""


# ── LLM script generation prompt ─────────────────────────────────────────────

LLM_SCRIPT_SYSTEM = """\\
You are a Godot 4.x GDScript expert. You will receive a game design document
(as JSON) for a VISUAL NOVEL generated from a video. Your task is to generate
the optional GDScript files that surround the template-owned dialogue runtime.

TEMPLATE-OWNED FILES — NEVER generate these:
- vn_manager.gd (visual-novel runtime: story, dialogue box, choices, subtitles)
- main.tscn, project.godot (scene/config templates)

OUTPUT FORMAT:
Return a JSON object mapping filenames to their complete GDScript source code.

{
  "game_manager.gd": "extends Node\\n...",
  "any_extra_file.gd": "extends Node\\n..."
}

REQUIRED FILE:
- game_manager.gd — game state around the VN. MUST keep this exact contract
  (the VN runtime wires to it):
    signal score_changed(new_score: int)
    var score: int = 0
    func add_score(amount: int = 1) -> void:   # must emit score_changed
  Use the "scenes" and "goals" from the design for objectives/win conditions.

RECOMMENDED EXTRAS (only if the design calls for them):
- alliance_map.gd / menu.gd — alliance or inventory overlays (Tab/menu flow)
- minigame scripts (dance, debate, banquet) referenced by the design's mechanics
- audio_manager.gd — music/ambience cues from the "atmosphere" field
- save_load.gd — persistence for choices and reputation

INPUT CONTRACT:
- The ONLY input action is "advance" (Space/Enter) plus left mouse click.
  There are no movement/jump/combat actions — do not reference any.

LANGUAGE CONTRACT:
- Simplified Chinese is the target language. Your scripts may only contain
  Chinese UI strings (plus data passed in from the design). Never hardcode
  foreign-language dialogue — all story text lives in the design data.

GODOT 4.x RULES:
- Use `@onready`, `@export`, typed variables (`var x: int = 0`)
- Signals: `signal name(param: type)` and `name.emit(value)`
- No `yield` — use `await` for coroutines
- Input: `Input.is_action_just_pressed("advance")`
- Autoload-free: access siblings via `get_node("/root/Main/GameManager")`
- `CanvasLayer` visibility is the inherited `visible` property; never invent `_visible`
- Conditional values use `value_if_true if condition else value_if_false`; boolean
  `and` / `or` operands must both be booleans
- `Object.set()` property names must be `String` or `StringName`, never `NodePath`

CODE QUALITY:
- Self-contained scripts — no external dependencies beyond Godot built-ins
- Handle edge cases (null nodes, first-frame guards)
- Use `_ready()` for initialization and `_process()` for updates
"""
