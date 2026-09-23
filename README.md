# v2g — Video to Godot Game

Transform any video into a playable Godot 4.x game using LLM-powered analysis.

## How It Works

```
Video (file/URL)
    │
    ├─ Subtitles → transcript (source-language dialogue, authoritative)
    ├─ Fast mode: ffmpeg extracts keyframes ─┐
    │                                         ▼
    └─ Detail mode (-d): full video ───▶ LLM Analysis ──▶ Visual Novel Project
                                         (multimodal)      (bilingual VN:
                                                            source line +
                                                            Chinese subtitle)
```

Key properties of generated games:

- **Visual novel**: dialogue flow, choice options, reputation score — input is
  Space/Enter/click only (`vn_manager.gd` runtime, template-owned).
- **Bilingual text**: every line shows the verbatim source-language transcript
  plus a Simplified Chinese subtitle; narration and UI are Chinese-only.
  Source-language text is extracted from the video's subtitles — never invented.

Two analysis modes:

| | Fast (default) | Detail (`-d`) |
|---|---|---|
| Input | Keyframe images (1 per 2s) | Full video file |
| Requires | Any vision model | Video-capable model |
| Output fields | Core design | + physics, behavior, layout, progression |
| Cost / speed | Lower / faster | Higher / slower |

1. **Extract** — Samples keyframes or prepares video for upload (yt-dlp + ffmpeg); pulls subtitles (sidecar or embedded) as the source-language transcript when available.
2. **Analyze** — Sends to multimodal LLM with the transcript injected; returns a structured game design with bilingual dialogue.
3. **Generate** — Scaffolds a visual-novel Godot 4.x project (`project.godot`, `main.tscn`, `vn_manager.gd` story runtime, `game_manager.gd`), compile-checks every script (one LLM repair pass, then fallback, on parse errors), then runs Godot headless to import assets and self-check.

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

# Custom run directory (default: projects/<timestamp>_<source>/ — new per run)
uv run v2g ./clip.mp4 -o ./my_game
```

Every run gets its own `projects/<timestamp>_<source>/` directory holding
`v2g.log`, `work/`, `llm/` and the game — delete it to remove the whole run.
Layered caching avoids pointless retries: `projects/.v2g_cache/` stores raw
LLM responses keyed by exact request content (reruns make zero API calls;
`V2G_LLM_CACHE=0` disables) **and** downloaded videos / re-encoded
intermediates keyed by URL or source file + settings (`V2G_MEDIA_CACHE=0`
disables — artifacts then go to the run's `work/`), so rerunning a source
skips both downloads and transcodes; `<run>/design.json` lets a rerun with
the same `-o` skip analysis entirely. Only a cached answer that fails
validation is refetched.

Then open the generated project in Godot:

```bash
godot --editor --path "projects/<run-id>"
```

## Configuration

All settings via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `V2G_LLM_API_KEY` | — | API key (required) |
| `V2G_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `V2G_LLM_MODEL` | `gpt-4o` | Model name |
| `V2G_LLM_CACHE` | `1` | Cache raw LLM responses across runs (`0` disables) |
| `V2G_MEDIA_CACHE` | `1` | Cache downloads/transcodes across runs (`0` disables) |
| `V2G_GODOT_PATH` | `godot` | Godot executable |
| `V2G_MAX_DURATION` | `120` | Max seconds per single analysis; longer detail-mode input is trimmed to this |
| `V2G_FRAME_INTERVAL` | `2.0` | Seconds between extracted frames (fast mode) |
| `V2G_VIDEO_MAX_MB` | `20` | Hard cap on upload size in MB — oversized files are split to fit (detail mode) |

## Project Structure

```
v2g/
├── v2g/
│   ├── __main__.py         # CLI
│   ├── config.py            # settings
│   ├── pipeline.py          # orchestrator
│   ├── video/
│   │   ├── extractor.py     # frame extraction (+ subtitle download for URLs)
│   │   ├── dialogue.py      # subtitle transcript extraction (source language)
│   │   └── asset_extractor.py  # seeded, hash-deduplicated asset frames
│   ├── llm/
│   │   ├── client.py        # OpenAI API wrapper
│   │   └── analyzer.py      # frames/video → bilingual game design
│   └── godot/
│       ├── generator.py     # project scaffolding + Godot self-check
│       └── templates.py     # VN scene, vn_manager story runtime, prompts
└── projects/                # generated games — one fresh directory per run
                              # (v2g.log, work/, llm/, and the game inside)
```
