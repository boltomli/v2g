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
  └──────────────────────────────────────────── Godot Project Generator
                                                    │
                                                    ▼
                                              projects/<title>/
                                              ├── project.godot
                                              ├── main.tscn
                                              ├── player.gd
                                              ├── game_manager.gd
                                              └── game_design.json
```

## Modes

### Fast mode (default)
Extracts 16 evenly-spaced keyframes, sends them as images to the LLM.
- Works with any multimodal model (GPT-4o, Claude 3.5, Gemini, etc.)
- Lower token cost, faster analysis
- Good for short clips and clear visual concepts

### Detail mode (`--detail` / `-d`)
Sends the full video file directly to a video-capable LLM.
- Requires a model that supports `video_url` content type
- Produces richer design: object behaviors, physics rules, spatial layouts, progression
- Video is auto-trimmed to `V2G_MAX_DURATION` (default 120s)
- Video is auto-compressed if over `V2G_VIDEO_MAX_MB` (default 20MB)
- Higher token cost, slower, but significantly more detailed output

## Configuration

All via env vars or `.env`:

| Variable | Default | Description |
|---|---|---|
| `V2G_LLM_API_KEY` | — | API key (**required**) |
| `V2G_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `V2G_LLM_MODEL` | `gpt-4o` | Model name |
| `V2G_GODOT_PATH` | `godot` | Godot executable path |
| `V2G_MAX_DURATION` | `120` | Max video seconds (detail mode) |
| `V2G_FRAME_INTERVAL` | `2.0` | Seconds between extracted frames (fast mode) |
| `V2G_VIDEO_MAX_MB` | `20` | Max upload size in MB (detail mode) |

## GameDesign Schema

The LLM returns a JSON document validated into `GameDesign`:

```python
class GameDesign:
    title: str                    # Game title
    genre: str                    # platformer, puzzle, action, etc.
    mechanics: list[str]          # Core gameplay mechanics
    objects: list[GameObject]     # All game objects (player, enemies, etc.)
    levels: list[LevelDesign]     # Level definitions with goals
    controls: list[str]           # Control scheme descriptions
    style: str                    # Art/visual style
    summary: str                  # 2-3 sentence pitch
    physics: str                  # Physics rules (detail mode)
    progression: str              # Difficulty scaling (detail mode)

class GameObject:
    name: str
    role: str                     # player / enemy / obstacle / collectible / environment
    visual: str                   # Visual description
    behavior: str                 # Movement/interaction patterns (detail mode)

class LevelDesign:
    name: str
    description: str
    goals: list[str]
    layout: str                   # Spatial layout description (detail mode)
```

## Generated Godot Project

### Files produced

| File | Source | Description |
|---|---|---|
| `project.godot` | template | Engine config, window size, input map |
| `main.tscn` | template | Root scene with Player CharacterBody2D + Camera2D |
| `player.gd` | template | 2D platformer controller (gravity, jump, horizontal) |
| `enemy.gd` | template | Auto-walking enemy with wall bounce (if enemies exist) |
| `game_manager.gd` | template | Score tracking, goals, game-over signal |
| `*.gd` (extras) | LLM-generated | UI, collectibles, level-specific scripts |
| `game_design.json` | analyzer output | Full design document as project metadata |

### Input mapping

- `move_left`: A / Left arrow
- `move_right`: D / Right arrow
- `jump`: Space / Up arrow

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
uv run v2g <source> [-o OUTPUT] [-d]

# Examples
uv run v2g ./gameplay.mp4                # fast mode
uv run v2g ./gameplay.mp4 -d             # detail mode
uv run v2g "https://youtube.com/..." -d  # URL + detail
uv run v2g ./clip.mp4 -o ./my_game       # custom output dir
```
