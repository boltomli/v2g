"""GDScript and Godot project file templates."""

from v2g.llm.analyzer import GameDesign

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

move_left={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":65,"key_label":0,"unicode":97,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194319,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}}
move_right={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":68,"key_label":0,"unicode":100,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194321,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}}
jump={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":32,"key_label":0,"unicode":32,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194320,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}}

[rendering]

renderer/rendering_method="gl_compatibility"
"""


# ── Scene (.tscn) helpers ───────────────────────────────────────────────────

def _uid() -> str:
    """Deterministic-ish uid based on a counter; Godot just needs uniqueness."""
    _uid.counter += 1
    return f"uid://v2g{_uid.counter:04d}"
_uid.counter = 0


def main_scene(player_script: str = "res://player.gd") -> str:
    uid_main = _uid()
    uid_player = _uid()
    _uid()  # reserve camera uid
    return f"""\
[gd_scene load_steps=2 format=3 {uid_main}]

[ext_resource type="Script" path="{player_script}" id="1_{uid_player}"]

[node name="Main" type="Node2D"]

[node name="Player" type="CharacterBody2D" parent="."]
script = ExtResource("1_{uid_player}")

[node name="CollisionShape2D" type="CollisionShape2D" parent="Player"]
shape = SubResource("RectangleShape2D_1")

[node name="Camera2D" type="Camera2D" parent="Player"]

[sub_resource type="RectangleShape2D" id="RectangleShape2D_1"]
size = Vector2(32, 48)
"""


# ── GDScript generators ────────────────────────────────────────────────────

def player_script(design: GameDesign) -> str:
    """Generate a CharacterBody2D player controller matching the game design."""
    speed = "300.0"
    jump_vel = "-400.0"

    if "endless-runner" in design.genre.lower():
        speed = "400.0"
        jump_vel = "-500.0"

    return f"""\
extends CharacterBody2D

const SPEED = {speed}
const JUMP_VELOCITY = {jump_vel}

var gravity: float = ProjectSettings.get_setting("physics/2d/default_gravity")

@onready var sprite := $Sprite2D if has_node("Sprite2D") else null

func _physics_process(delta: float) -> void:
    # Gravity
    if not is_on_floor():
        velocity.y += gravity * delta

    # Jump
    if Input.is_action_just_pressed("jump") and is_on_floor():
        velocity.y = JUMP_VELOCITY

    # Horizontal movement
    var direction := Input.get_axis("move_left", "move_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)

    # Flip sprite
    if sprite and direction != 0.0:
        sprite.flip_h = direction < 0.0

    move_and_slide()
"""


def enemy_script() -> str:
    return """\
extends CharacterBody2D

const SPEED = 150.0
var direction := -1.0

var gravity: float = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta: float) -> void:
    if not is_on_floor():
        velocity.y += gravity * delta

    velocity.x = direction * SPEED
    move_and_slide()

    # Reverse at walls
    if is_on_wall():
        direction *= -1.0
"""


def game_manager_script(design: GameDesign) -> str:
    goals = ", ".join(f'"{g}"' for lvl in design.levels for g in lvl.goals)
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


# ── Full GDScript generation via LLM (advanced) ────────────────────────────

LLM_SCRIPT_SYSTEM = """\
You are a Godot 4.x GDScript expert. Given a game design document as JSON,
generate the GDScript files needed. Return a JSON object mapping filenames
to their full GDScript source code. Example:

{{
  "player.gd": "extends CharacterBody2D\\n...",
  "enemy.gd": "extends CharacterBody2D\\n...",
  "game_manager.gd": "extends Node\\n..."
}}

Rules:
- Use Godot 4.x syntax (no yield, use await).
- All scripts must be self-contained.
- Include comments explaining key logic.
- Return ONLY the JSON object, no markdown fences.
"""
