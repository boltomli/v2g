# v2g — Video to Godot Game

Transform any video into a playable Godot 4.x game using LLM-powered analysis.

## How It Works

```
Video (file/URL)
    │
    ▼
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│  Frame        │───▶│  LLM Video    │───▶│  Godot Project   │
│  Extraction   │    │  Analysis     │    │  Generation      │
│  (ffmpeg/     │    │  (multimodal) │    │  (templates +    │
│   yt-dlp)     │    │               │    │   LLM scripts)   │
└──────────────┘    └───────────────┘    └──────────────────┘
                                                  │
                                                  ▼
                                          Playable .godot project
```

1. **Extract** — Samples keyframes from a local video or URL (yt-dlp + ffmpeg).
2. **Analyze** — Sends frames to a multimodal LLM; returns a structured game design (genre, mechanics, objects, levels, controls).
3. **Generate** — Scaffolds a Godot 4.x project: `project.godot`, scenes, player/enemy scripts, and LLM-generated extras.

## Prerequisites

- **Python 3.11+**
- **ffmpeg** on PATH
- **yt-dlp** on PATH (for URL downloads)
- **OpenAI-compatible API key** (GPT-4o, Claude via proxy, etc.)
- **Godot 4.x** to open the generated project

## Setup

```bash
pip install -e .
cp .env.example .env   # fill in V2G_LLM_API_KEY
```

## Usage

```bash
# From a local video
v2g ./gameplay.mp4

# From a URL
v2g "https://www.youtube.com/watch?v=..."

# Custom output directory
v2g ./clip.mp4 -o ./my_game
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
| `V2G_MAX_DURATION` | `120` | Max video seconds to process |
| `V2G_FRAME_COUNT` | `16` | Keyframes to extract |

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
