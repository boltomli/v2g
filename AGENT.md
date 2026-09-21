# v2g Agent Document

## Overview

`v2g` is a pipeline that transforms a video into a playable Godot 4.x game project.

```
Video (file / URL)
  │
  ├─ Fast mode (default): ffmpeg extracts N keyframes ──┐
  │                                                      ▼
  ├─ Detail mode (-d): ffmpeg trims/compresses ──▶ LLM analyzes ──▶ GameDesign JSON
  │                                                      │
  │                                                      ▼
  │                                              Asset Extractor (ffmpeg)
  │                                               ├─ Background frames
  │                                               ├─ Character sprites
  │                                               └─ Object sprites
  │                                                      │
  │                              ┌────────────────────────┘
  │                              ▼
  │                      [Image Gen API]  ← optional, future
  │                       (style transfer)
  │                              │
  └──────────────────────────────┼───────────────────────┘
                                 ▼
                         Godot Project Generator
                                 │
                                 ▼
                          projects/<title>/
                          ├── project.godot
                          ├── main.tscn
                          ├── player.gd
                          ├── game_manager.gd
                          ├── assets/          ← extracted video frames
                          │   ├── background.png
                          │   ├── characters/
                          │   └── objects/
                          └── game_design.json
```

## Modes

### Fast mode (default)
Extracts keyframe images and sends them to the LLM.
- Short video (<5 min): fixed-interval extraction (1 frame per `V2G_FRAME_INTERVAL` seconds).
- Long video (≥5 min): **scene-detect** extraction — ffmpeg identifies frames where visuals change
  significantly. Adapts threshold to stay within `V2G_FRAME_BUDGET` (default 40 frames).
- Works with any multimodal model (GPT-4o, Claude 3.5, Gemini, etc.)
- Lower token cost, faster analysis

### Detail mode (`--detail` / `-d`)
Sends the full video file directly to a video-capable LLM.
- Requires a model that supports `video_url` content type
- Short video (≤10 min): single upload, trimmed/compressed to fit `V2G_VIDEO_MAX_MB`
- **Long video (>10 min): automatic chunked analysis** — splits into `V2G_CHUNK_DURATION`-second
  segments (default 10 min each), analyzes each segment separately, then merges results.
  Characters are deduplicated (most detailed version kept), scenes concatenated, mechanics unioned.
- Produces richer design: object behaviors, physics rules, spatial layouts, progression
- Higher token cost, slower, but significantly more detailed output

## Configuration

All via env vars or `.env`:

| Variable | Default | Description |
|---|---|---|
| `V2G_LLM_API_KEY` | — | API key (**required**) |
| `V2G_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `V2G_LLM_MODEL` | `gpt-4o` | Model name |
| `V2G_GODOT_PATH` | `godot` | Godot executable path |
| `V2G_MAX_DURATION` | `120` | Max single-upload seconds (detail mode) |
| `V2G_FRAME_INTERVAL` | `2.0` | Seconds between frames (short video fast mode) |
| `V2G_FRAME_BUDGET` | `40` | Max keyframes to send to LLM (long video cap) |
| `V2G_SCENE_THRESHOLD` | `0.3` | ffmpeg scene-detect sensitivity (0.0–1.0) |
| `V2G_CHUNK_DURATION` | `600` | Seconds per analysis chunk (detail mode, long videos) |
| `V2G_VIDEO_MAX_MB` | `20` | Max upload size in MB per chunk |

## GameDesign Schema

The LLM returns a JSON document validated into `GameDesign`:

```python
class GameDesign:
    title: str                      # Game title
    genre: str                      # platformer, puzzle, action, adventure, RPG, etc.
    summary: str                    # 2-3 sentence pitch
    narrative: str                  # Full story arc with character motivations
    mechanics: list[str]            # Core gameplay mechanics
    controls: list[str]             # Control scheme descriptions
    style: str                      # Art/visual style
    physics: str                    # Physics rules (gravity, momentum, collisions)
    progression: str                # Difficulty scaling, area unlocking
    atmosphere: str                 # Mood, tone, sound design cues
    characters: list[Character]     # All characters with full detail
    objects: list[GameObject]       # All game objects
    scenes: list[SceneDesign]       # Scene-by-scene breakdown
    scene_transitions: list[Transition]  # How scenes connect
    dialogue_samples: list[Dialogue]     # Representative dialogue lines

class Character:
    name: str
    face_id: str                  # stable identity anchor (e.g. "char_01"), survives costume changes
    role: str                     # protagonist / antagonist / NPC / companion / boss / minion
    visual: str                   # PRIMARY appearance: face, body, hair, build (never changes)
    personality: str              # temperament, speech patterns, motivations
    behavior: str                 # AI behavior: patrol, attack, dialogue triggers
    abilities: list[str]          # special abilities, attacks
    relationships: str            # relations to other characters
    personas: list[Persona]       # alternate costumes/outfits of the same person

class Persona:
    outfit: str                   # costume name (e.g. "armored battle gear", "casual clothes")
    visual: str                   # full appearance in this outfit (face consistent, clothes change)
    context: str                  # when/where this look appears (e.g. "act 1", "final battle")

class GameObject:
    name: str
    role: str                     # player / enemy / obstacle / collectible / environment / trigger / UI
    visual: str                   # Detailed visual description
    behavior: str                 # Movement, interaction, state changes
    spatial: str                  # Position relative to other elements

class SceneDesign:
    name: str
    description: str              # What happens in this scene
    layout: str                   # Spatial layout: ground, platforms, walls, exits
    goals: list[str]              # Player objectives
    hazards: list[str]            # Dangers, enemies, traps
    triggers: list[str]           # Events that activate
    visual_theme: str             # Scene-specific colors, lighting

class Transition:
    source: str                   # Source scene name
    destination: str              # Destination scene name
    trigger: str                  # What causes the transition
    effect: str                   # Visual transition effect

class Dialogue:
    speaker: str
    line: str
    context: str                  # When/why this line is said
```

## Generated Godot Project

### Files produced

| File | Source | Description |
|---|---|---|
| `project.godot` | template | Engine config, window size, input map (move, jump, interact) |
| `main.tscn` | data-driven | Scene tree: Player + Enemies + Environment + Collectibles + Triggers + DialogueUI + Sprites |
| `player.gd` | LLM | Player controller matching character's behavior and abilities |
| `game_manager.gd` | LLM | Game state: score, goals, scene transitions, win/lose |
| `enemy.gd` | LLM | Enemy AI matching character patrol/attack patterns (if enemies exist) |
| `dialogue_ui.gd` | LLM | Dialogue system with speaker name + line display (if dialogue exists) |
| `*.gd` (extras) | LLM | Any additional scripts: inventory, combat, camera, UI, etc. |
| `assets/background.png` | ffmpeg | Representative frame from video midpoint as scene background |
| `assets/characters/*.png` | ffmpeg | Character sprite frames (cropped using spatial hints) |
| `assets/objects/*.png` | ffmpeg | Object sprite frames (collectibles, weapons, etc.) |
| `game_design.json` | analyzer | Full design document as project metadata |

### Asset extraction

After LLM analysis, the pipeline extracts visual assets from the source video:
- **Background**: A representative frame from the video midpoint → `assets/background.png`
- **Characters**: Frames at 3 timestamps per character, cropped using the LLM's spatial descriptions → `assets/characters/<name>.png`
- **Objects**: Frames for collectibles, weapons, decoration objects → `assets/objects/<name>.png`
- **Scenes**: Additional background frames for multi-scene videos → `assets/scene_<name>.png`

Spatial cropping uses the `spatial` field from `GameObject` analysis:
- "left side of frame" → crops to left 40%
- "center" → crops to center 60%
- "right third" → crops to right 40%
- No hint → full frame

### Image generation (future)

An abstract `ImageGenProvider` interface is reserved for future image generation APIs:
- **NullProvider** (default): no-op, assets come directly from video frames
- **Future**: DALL-E, Stable Diffusion, or local APIs can transform extracted assets to match `--instruct` style

Configuration (env vars / `.env`):

| Variable | Default | Description |
|---|---|---|
| `V2G_IMAGEGEN_API_KEY` | — | API key for image generation service |
| `V2G_IMAGEGEN_BASE_URL` | `https://api.openai.com/v1` | Image gen endpoint |
| `V2G_IMAGEGEN_MODEL` | `dall-e-3` | Image gen model |
| `V2G_IMAGEGEN_STYLE` | — | Global style prefix prepended to all prompts |

To implement a new provider: subclass `ImageGenProvider` in `v2g/llm/image_gen.py`,
implement `transform()` and `is_available()`, then update `get_provider()` factory.

### Scene tree (main.tscn)

The scene is generated programmatically from the design:
- **Player** — CharacterBody2D + CollisionShape2D + Camera2D
- **Enemy_\<Name\>** — CharacterBody2D for each enemy character/object (max 8)
- **Env_\<Name\>** — StaticBody2D for environment/obstacle objects (max 12)
- **Collectible_\<Name\>** — Area2D for collectible objects (max 16)
- **Trigger_\<Name\>** — Area2D for trigger objects (max 8)
- **GameManager** — Node with game_manager.gd
- **DialogueUI** — CanvasLayer + Panel + RichTextLabel (if dialogue exists)

### Input mapping

- `move_left`: A / Left arrow
- `move_right`: D / Right arrow
- `jump`: Space / Up arrow
- `interact`: E / Enter

### Extending

Generated scripts are starting points. Open in Godot 4.x and iterate:
1. Replace placeholder shapes with actual sprites
2. Build tilemaps from the `level.layout` descriptions
3. Wire up `game_manager.gd` signals to UI nodes
4. Add audio based on `style` and `mechanics` descriptions

## Dependencies

- Python 3.11+
- ffmpeg (PATH)
- yt-dlp (PATH, for URL downloads)
- Godot 4.x (to run generated projects)
- OpenAI-compatible API with vision/video support

## CLI

```bash
uv run v2g <source> [-o OUTPUT] [-d] [-i INSTRUCTION]

# Examples
uv run v2g ./gameplay.mp4                           # fast mode
uv run v2g ./gameplay.mp4 -d                        # detail mode
uv run v2g "https://youtube.com/..." -d             # URL + detail
uv run v2g ./clip.mp4 -o ./my_game                  # custom output dir
uv run v2g ./clip.mp4 -i "change to medieval"       # medieval reskin
uv run v2g ./clip.mp4 -i "vampire theme"            # vampire reskin
uv run v2g ./clip.mp4 -d -i "Lord of the Rings"     # detail + style override
```
