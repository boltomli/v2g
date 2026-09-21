# v2g — Video to Godot Game

Transform any video into a playable Godot 4.x game using LLM-powered analysis.

## How It Works

```
Video (file/URL)
    │
    ├─ Fast mode: ffmpeg extracts keyframes ─┐
    │                                         ▼
    └─ Detail mode (-d): full video ───▶ LLM Analysis ──▶ Godot Project
                                         (multimodal)      (templates +
                                                           LLM scripts)
```

Two analysis modes:

| | Fast (default) | Detail (`-d`) |
|---|---|---|
| Input | 16 keyframe images | Full video file |
| Requires | Any vision model | Video-capable model |
| Output fields | Core design | + physics, behavior, layout, progression |
| Cost / speed | Lower / faster | Higher / slower |

1. **Extract** — Samples keyframes or prepares video for upload (yt-dlp + ffmpeg).
2. **Analyze** — Sends to multimodal LLM; returns a structured game design.
3. **Generate** — Scaffolds a Godot 4.x project: `project.godot`, scenes, player/enemy scripts, and LLM-generated extras.

## Prerequisites

- **Python 3.11+**
- **ffmpeg** on PATH
- **yt-dlp** on PATH (for URL downloads)
- **OpenAI-compatible API key** (GPT-4o, Claude via proxy, etc.)
- **Godot 4.x** to open the generated project

## Setup

```bash
git clone <repo> && cd v2g
uv sync
cp .env.example .env   # fill in V2G_LLM_API_KEY
```

## Usage

```bash
# Fast mode — extract keyframes
uv run v2g ./gameplay.mp4

# Detail mode — send full video to video-capable LLM
uv run v2g ./gameplay.mp4 -d

# From a URL
uv run v2g "https://www.youtube.com/watch?v=..."

# Custom output directory
uv run v2g ./clip.mp4 -o ./my_game
```

Then open the generated project in Godot:

```bash
godot --editor projects/<game-title>
```

## Configuration

All settings via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `V2G_LLM_API_KEY` | — | API key (required) |
| `V2G_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `V2G_LLM_MODEL` | `gpt-4o` | Model name |
| `V2G_GODOT_PATH` | `godot` | Godot executable |
| `V2G_MAX_DURATION` | `120` | Max video seconds (detail mode) |
| `V2G_FRAME_INTERVAL` | `2.0` | Seconds between extracted frames (fast mode) |
| `V2G_VIDEO_MAX_MB` | `20` | Max upload size in MB (detail mode) |

## Project Structure

```
v2g/
├── v2g/
│   ├── __main__.py         # CLI
│   ├── config.py            # settings
│   ├── pipeline.py          # orchestrator
│   ├── video/extractor.py   # frame extraction
│   ├── llm/
│   │   ├── client.py        # OpenAI API wrapper
│   │   └── analyzer.py      # frames → game design
│   └── godot/
│       ├── generator.py     # project scaffolding
│       └── templates.py     # GDScript + scene templates
└── projects/                # generated games
```
