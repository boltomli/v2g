# v2g — Video to Godot Game

Transform any video into a playable Godot 4.x game using LLM-powered analysis.

## How It Works

Three stages, each feeding the next — nothing from a later stage ever reaches
back into an earlier one:

```
Video (file/URL)
    │  Subtitles → transcript (source-language dialogue, authoritative;
    │               names in it are NOT characters unless shown)
    ▼
Stage 1 · analyze the ORIGINAL video + extract its assets
    Fast mode: ffmpeg keyframes ─┐   faithful GameDesign (characters, story,
    Detail mode (-d): full video ┴─▶ scenes, props) + the frames that show them:
                                   background/stills keep the video's aspect,
                                   char/obj sprites are 512×512 and model-checked
    ▼
Stage 2 · rewrite + redraw — with your theme, or none
    design re-skin (theme when given, otherwise "must differ from the source")
    assets redrawn beside their frames (local image gen; -i supplies the theme)
    ▼
Stage 3 · generate the game flow and copy
    Visual-novel Godot 4.x project (bilingual VN: source line + Chinese subtitle)
    voice-over / background music: planned, not implemented yet
```

Key properties of generated games:

- **Visual novel**: dialogue flow, choice options, reputation score — input is
  Space/Enter/click only (`vn_manager.gd` runtime, template-owned).
- **Bilingual text**: every line shows the verbatim source-language transcript
  plus a Simplified Chinese subtitle; narration and UI are Chinese-only.
  Source-language text is extracted from the video's subtitles — never invented.

Two analysis modes (both part of stage 1):

| | Fast (default) | Detail (`-d`) |
|---|---|---|
| Input | Keyframe images (1 per 2s) | Full video file |
| Requires | Any vision model | Video-capable model |
| Output fields | Core design | + physics, behavior, layout, progression |
| Cost / speed | Lower / faster | Higher / slower |

1. **Analyze + extract** — samples keyframes or prepares the video for upload
   (yt-dlp + ffmpeg), pulls subtitles (sidecar or embedded) as the
   source-language transcript, returns a faithful game design with bilingual
   dialogue, then cuts the assets out of the original video.
2. **Rewrite + redraw** — re-skins the design's presentation with the theme you
   passed (`-i`) or, with none, simply so it no longer looks like the source;
   characters, objects and scenes get new names and every background is a
   different place, while `face_id`, gameplay and the verbatim dialogue stay
   fixed — extracted sprites follow the renames (asset keys and speaker labels
   are re-keyed). If the re-skin fails, the run stops here instead of redrawing
   the art and generating a source-faithful game. The assets are then redrawn
   beside their frames when an image-gen backend is configured (skip is always
   reported); each brief opens and closes with the theme and carries the
   re-skinned design's art style, so the redraw follows `-i` instead of the
   source frame — and the text the source burned in (subtitles, watermark) is
   ruled out at every step: the re-skin drops it from the design, the frame
   caption ignores it, every brief forbids it — redrawn art ships without
   text. The source-referenced candidate is text-led: the frame enters as a
   caption and stays a reference, never a template.
3. **Generate** — scaffolds a visual-novel Godot 4.x project (`project.godot`,
   `main.tscn`, `vn_manager.gd` story runtime, `game_manager.gd`),
   compile-checks every script (one LLM repair pass, then fallback, on parse
   errors), then runs Godot headless to import assets and self-check.
   Voice-over and background music are planned but not implemented yet.

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
uv sync --extra imagegen   # optional: local image generation (Qwen-Image-2.1)
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

# Stage 2 theme — rewrite + redraw in a theme (omit it: the art only has to
# differ from the source video; extraction is the same either way)
uv run v2g ./clip.mp4 -i "vampire theme"
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
| `V2G_IMAGEGEN_PROVIDER` | — | Image generation: `qwen` = local Qwen-Image-2.1 (needs `--extra imagegen`); unset = off |
| `V2G_IMAGEGEN_STYLE` | — | Style prefix prepended to all image-gen prompts |
| `V2G_IMAGEGEN_STEPS` | `20` | Image-gen denoising steps |
| `V2G_IMAGEGEN_MAX_SIDE` | `1024` | Longest output edge for generated assets |
| `V2G_IMAGEGEN_AB` | — | Per asset, draw text-only and source-referenced candidates and keep the judge's pick (`1` = on; off = reference only) |
| `V2G_ASSET_VERIFY` | `1` | Vision model must confirm each extracted frame shows its asset (`0` = extract unchecked) |

## Development

Lint/format is `ruff` (line length 100, target py311) and the repo is kept
ruff-clean:

```bash
uv run ruff check .          # lint
uv run ruff format .         # format (also formats ```python blocks in AGENT.md)
uv run pytest                # tests
```

Git hooks run through [prek](https://github.com/brodul/prek) (a pre-commit
drop-in) — `.pre-commit-config.yaml` pins `ruff-check --fix` + `ruff-format`
to ruff v0.16.8:

```bash
prek install                 # wire the pre-commit hook (once per clone)
prek run --all-files         # run every hook over the tree
```

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
│   │   ├── asset_extractor.py  # shot-anchored, model-verified, 512×512 sprite frames
│   ├── llm/
│   │   ├── client.py        # OpenAI API wrapper
│   │   └── analyzer.py      # frames/video → bilingual game design
│   └── godot/
│       ├── generator.py     # project scaffolding + Godot self-check
│       └── templates.py     # VN scene, vn_manager story runtime, prompts
└── projects/                # generated games — one fresh directory per run
                              # (v2g.log, work/, llm/, and the game inside)
```
