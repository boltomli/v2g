# v2g Agent Document

## Overview

`v2g` is a pipeline that transforms a video into a playable Godot 4.x game project.

```
Video (file / URL)
  │
  ├─ URL: yt-dlp downloads video + best-effort .srt sidecars (media cache) ─┐
  │                                                            ▼
  │                                          Transcript extraction (ffmpeg)
  │                                           └─ source-language lines ONLY
  │                                              (sidecar / embedded subs)
  ├─ Fast mode (default): ffmpeg extracts N keyframes ──┐      │
  │                                                      ▼      │
  ├─ Detail mode (-d): ffmpeg trims/compresses ──▶ LLM analyzes ─┘ ──▶ GameDesign JSON
  │                                                      │
  │                                                      ▼
  │                                              Asset Extractor (ffmpeg)
  │                                               ├─ Background frames
  │                                               ├─ Character sprites (byte-distinct)
  │                                               └─ Object sprites
  │                                                      │
  │                              ┌────────────────────────┘
  │                              ▼
  │                      [Image Gen Provider] ← optional (V2G_IMAGEGEN_PROVIDER=qwen)
  │                       (style transfer)
  │                              │
  └──────────────────────────────┼───────────────────────────────┘
                                 ▼
                         Godot Project Generator  (visual novel)
                                 │  └─ Godot script check + repair, import, headless boot (self-check)
                                 ▼
                          projects/<run-id>/         (fresh directory per run)
                          ├── v2g.log               (full run log)
                          ├── work/                 (frames; downloads/segments when media cache off)
                          ├── llm/                  (raw LLM responses)
                          ├── project.godot        (advance input only)
                          ├── main.tscn            (Control root + GameManager)
                          ├── vn_manager.gd        (template-owned VN runtime,
                          │                         bilingual story embedded)
                          ├── game_manager.gd
                          ├── assets/              ← extracted video frames
                          │   ├── background.png
                          │   ├── char_*.png
                          │   ├── obj_*.png
                          │   └── scene_*.png
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
- Short video (≤ `V2G_CHUNK_DURATION` + 30 s — 90 s at defaults): single upload, trimmed to
  `V2G_MAX_DURATION` when longer, compressed at most once, then losslessly
  split by duration until every file fits `V2G_VIDEO_MAX_MB` (the cap is hard — compression
  output is verified, never assumed)
- **Long video (beyond that): automatic chunked analysis** — splits into `V2G_CHUNK_DURATION`-second
  segments (default 60 s each); each segment is compressed at most once, and any segment
  still over `V2G_VIDEO_MAX_MB` is halved losslessly (stream copy, no re-encode) until it fits.
  Analyzes each segment separately, then merges results.
  Characters are deduplicated (most detailed version kept), scenes concatenated, mechanics unioned.
- Produces richer design: object behaviors, physics rules, spatial layouts, progression
- Higher token cost, slower, but significantly more detailed output

## Configuration

All via env vars or `.env`:

Every CLI run creates a fresh `projects/<timestamp>_<source>/` directory before
any work starts; `v2g.log`, `work/` (downloads/frames/segments), `llm/` (raw
model responses) and the generated game all live inside it, so deleting that
one directory removes every artifact of the run. `-o` overrides the directory.

The pipeline is layered (extract → analyze → generate) with two caches so
retries are never pointless: `projects/.v2g_cache/` holds raw LLM responses
keyed by the exact request (model, prompts, file contents, sampling params) —
rerunning the same source makes zero API calls; its `media/` subtree holds
downloaded videos and re-encoded intermediates keyed by URL or source file +
settings, so reruns skip downloads and transcodes (`V2G_MEDIA_CACHE=0` sends
those artifacts to `work/` instead); `<run>/design.json` is an
analysis checkpoint, so rerunning with the same `-o` skips the LLM entirely.
Only a *cached* answer that fails parsing/refreshes once is invalidated and
refetched; a fresh malformed response is never re-requested.

| Variable | Default | Description |
|---|---|---|
| `V2G_LLM_API_KEY` | — | API key (**required**) |
| `V2G_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `V2G_LLM_MODEL` | `gpt-4o` | Model name |
| `V2G_LLM_CACHE` | `1` | Content-addressed raw-response cache across runs (`0` disables) |
| `V2G_MEDIA_CACHE` | `1` | Download/transcode artifact cache across runs (`0` disables) |
| `V2G_GODOT_PATH` | `godot` | Godot executable path |
| `V2G_MAX_DURATION` | `120` | Max seconds per single analysis; longer detail-mode input is trimmed to this |
| `V2G_FRAME_INTERVAL` | `2.0` | Seconds between frames (short video fast mode) |
| `V2G_FRAME_BUDGET` | `40` | Max keyframes to send to LLM (long video cap) |
| `V2G_SCENE_THRESHOLD` | `0.3` | ffmpeg scene-detect sensitivity (0.0–1.0) |
| `V2G_CHUNK_DURATION` | `60` | Seconds per analysis chunk (detail mode; must be ≤ `V2G_MAX_DURATION`) |
| `V2G_VIDEO_MAX_MB` | `20` | Max upload size in MB per chunk |

## Language & dialogue contract

The generated game is a **visual novel** with bilingual dialogue. Hard rules,
enforced in the analyzer prompts and the generator:

- **Target language: Simplified Chinese.** Every story entry carries `line_zh`
  (translation for dialogue, original Chinese narration otherwise). All
  template-authored UI strings (hints, score label, end card) are Chinese-only.
- **Source language comes from the video, never from the LLM.** `line` may only
  contain lines transcribed from the source material. The pipeline extracts
  subtitles (sidecar `.srt`/`.vtt`/`.ass` — bare or language-tagged like
  `.en.srt`, which is what yt-dlp writes — or the embedded subtitle stream) and
  injects the transcript into every analysis mode; LLM prompts mark it
  authoritative and require `line` to stay EMPTY when no transcript exists.
- Choice options are player-authored UI text: Chinese-only, unless the exact
  wording appears in the transcript.
- `dialogue_samples` **is the full playable script in order** — the generator
  embeds it (steps + choices + score) into `vn_manager.gd` at generation time.

## GameDesign Schema

The LLM returns a JSON document validated into `GameDesign`:

```python
class GameDesign:
    title: str  # Game title
    genre: str  # platformer, puzzle, action, adventure, RPG, etc.
    summary: str  # 2-3 sentence pitch
    narrative: str  # Full story arc with character motivations
    mechanics: list[str]  # Core gameplay mechanics
    controls: list[str]  # Control scheme descriptions
    style: str  # Art/visual style
    physics: str  # Physics rules (gravity, momentum, collisions)
    progression: str  # Difficulty scaling, area unlocking
    atmosphere: str  # Mood, tone, sound design cues
    characters: list[Character]  # All characters with full detail
    objects: list[GameObject]  # All game objects
    scenes: list[SceneDesign]  # Scene-by-scene breakdown
    scene_transitions: list[Transition]  # How scenes connect
    dialogue_samples: list[Dialogue]  # Representative dialogue lines


class Character:
    name: str
    face_id: str  # stable identity anchor (e.g. "char_01"), survives costume changes
    role: str  # protagonist / antagonist / NPC / companion / boss / minion
    visual: str  # PRIMARY appearance: face, body, hair, build (never changes)
    personality: str  # temperament, speech patterns, motivations
    behavior: str  # AI behavior: patrol, attack, dialogue triggers
    abilities: list[str]  # special abilities, attacks
    relationships: str  # relations to other characters
    personas: list[Persona]  # alternate costumes/outfits of the same person


class Persona:
    outfit: str  # costume name (e.g. "armored battle gear", "casual clothes")
    visual: str  # full appearance in this outfit (face consistent, clothes change)
    context: str  # when/where this look appears (e.g. "act 1", "final battle")


class GameObject:
    name: str
    role: str  # player / enemy / obstacle / collectible / environment / trigger / UI
    visual: str  # Detailed visual description
    behavior: str  # Movement, interaction, state changes
    spatial: str  # Position relative to other elements


class SceneDesign:
    name: str
    description: str  # What happens in this scene
    layout: str  # Spatial layout: ground, platforms, walls, exits
    goals: list[str]  # Player objectives
    hazards: list[str]  # Dangers, enemies, traps
    triggers: list[str]  # Events that activate
    visual_theme: str  # Scene-specific colors, lighting


class Transition:
    source: str  # Source scene name
    destination: str  # Destination scene name
    trigger: str  # What causes the transition
    effect: str  # Visual transition effect


class Dialogue:
    speaker: str  # empty for narration entries
    line: str  # VERBATIM source line from the transcript; empty = narration
    line_zh: str  # REQUIRED Simplified Chinese: translation / original narration
    context: str  # When/why this line is said
    choices: list[DialogueChoice]  # player options: {line, line_zh, score}


class DialogueChoice:
    line: str = ""  # source-language text ONLY if verbatim in transcript
    line_zh: str  # Simplified Chinese option text
    score: int = 1  # reputation/alliance points on selection
```

## Generated Godot Project

### Files produced

| File | Source | Description |
|---|---|---|
| `project.godot` | template | Engine config, window size matching the source video's aspect (ffprobe; 1280x720 fallback), `advance` input (Space/Enter) |
| `main.tscn` | template | Visual-novel root: `Control` (vn_manager.gd) + `GameManager` |
| `vn_manager.gd` | template | VN runtime: embedded bilingual story, dialogue box (source line + Chinese subtitle), choices, portraits, background flashes, restart |
| `game_manager.gd` | LLM | Game state honoring the `score_changed`/`add_score` contract (template fallback) |
| `*.gd` (extras) | LLM | Optional extras: alliance map, minigames, audio, save/load |
| `assets/background.png` | ffmpeg | Representative frame from video midpoint as scene background |
| `assets/char_*.png` | ffmpeg | Character sprite frames — seeded per entity and hash-checked byte-distinct |
| `assets/obj_*.png` | ffmpeg | Object sprite frames (collectibles, weapons, etc.) |
| `assets/scene_*.png` | ffmpeg | Additional scene backgrounds for multi-scene videos |
| `game_design.json` | analyzer | Full design document as project metadata |

After writing files, the generator compile-checks every script with
`godot --check-only` (the boot only parses scripts the main scene references),
repairs parse failures with one follow-up LLM pass, and falls back for what
still fails (`game_manager.gd` → template, unreferenced extras dropped), then
runs Godot headless twice (import scan, then a boot) — missing assets or
script errors surface at **generation time**.

### Asset extraction

After LLM analysis, the pipeline extracts visual assets from the source video:
- **Background**: A representative frame from the video midpoint → `assets/background.png`
- **Characters**: Frames from a per-entity seeded time window, cropped using the
  LLM's spatial descriptions → `assets/char_<name>.png`
- **Objects**: key-object roles match on any slash-separated token with the
  Chinese glosses folded in (`decoration / 身份标识`, `身份标识 / 互动道具`,
  `装饰 / 遮挡物` all hit the key set), UI-layer props (spatial/icons that
  live in a panel, never in a frame) are skipped, and each object is anchored
  to the scene that names it — bigram overlap between the object's name (or,
  for worn props, its `visual`/`behavior` naming the wearer) and the scene's
  text. Sampling then happens inside that scene's **establishing shot = the
  shot starting at the boundary (video start or detected cut) NEAREST the
  scene's timeline cell, sampled from its midpoint** (cells are equal-duration
  and misalign cuts — measured: "longest shot" picked a4.5 s dialogue beat
  over the2.5 s prop wide; "first segment of the cell" picked a leftover tail
  of the previous scene), cropped by the object's `spatial` hint. Worn props
  honor body部位 keywords too
  (胸前 → chest band, 腰侧 → waist band) so the sprite shows the wear region,
  not the wearer's face. Objects with no scene match keep the seeded window.
- **Scenes**: Additional background frames for multi-scene videos → `assets/scene_<name>.png`

**Distinctness guarantee**: each entity samples its own window of the timeline
(scene establishing shot for objects, name-seeded otherwise), and every accepted
sprite's SHA-256 is checked against everything already extracted — duplicates
are re-taken with jittered stamps. One entity can never receive another
entity's byte-identical frame.

Spatial cropping uses the `spatial` field from `GameObject` analysis — English
and Chinese framing words both parse (the LLM answers in the video's language):
- "left side of frame" / "画面左侧近景" → crops to left 40%
- "center" / "骑士胸口正中" → crops to center 60%
- "right third" / "右腰侧" → crops to right 40%
- "上方正中，顶部" → top-center 50% × top 45%
- No hint → full frame

### Image generation (optional, local)

Extracted assets can be re-drawn in a target style by a **local Qwen-Image-2.1**
model instead of shipping raw video frames. Two triggers:

- **`-i/--instruct <style>` — mandatory.** The run must restyle; if no image-gen
  backend can run (provider unset, CUDA/deps missing), it logs an explicit
  warning that assets were **NOT** restyled instead of silently skipping.
- **`V2G_IMAGEGEN_AUTORESTYLE=1` — automatic.** No `-i` needed: the prompt is
  the design's own `style` summary of the source video. A skipped run is
  reported at NOTICE level (not silently).

Providers:

- **NullProvider** (default, `V2G_IMAGEGEN_PROVIDER` unset): no-op — assets are
  the raw extracted frames, zero model involvement.
- **QwenImage21Provider** (`V2G_IMAGEGEN_PROVIDER=qwen`): in-process diffusers
  run of Qwen-Image-2.1 — one img2img restyle per asset; the design's `visual`
  description is passed as `reference` so subjects keep their identity.

Setup:

1. `uv sync --extra imagegen` — pulls a CUDA torch build plus git diffusers;
   base installs stay light (torch is ~2 GB).
2. First use downloads a ~23 GB managed bundle into
   `projects/.v2g_cache/imagegen/qwen-image-2.1/` — the official bf16 text
   encoder + VAE (ModelScope mirror) plus the unsloth GGUF Q4_K_M denoiser
   (the full bf16 stack is 33 GB and will not fit in RAM alongside the OS).
   Pre-warm it ahead of a run with:
   `.venv/Scripts/python -c "from v2g.llm.image_gen import prepare_model; prepare_model()"`
3. Placement picks itself from VRAM (the bf16 text encoder alone is 17.5 GB):
   ≥28 GB → fully on GPU, ≥20 GB → model CPU offload, else sequential
   offload with the GGUF denoiser kept resident. On a 6 GB RTX 4050 laptop
   the low-VRAM path also disables the prefix KV cache (~2 GB) and expands
   allocator segments — without that every step pages over PCIe at ~100 s/step.
   Measured: **~8 min per asset at the 1024 px / 40-step defaults**, ~4.5 min
   with `V2G_IMAGEGEN_STEPS=20`; model load ~35 s on first transform.

A failing asset keeps its original frame (logged) — image generation can
never fail a run. `V2G_IMAGEGEN_MODEL` overrides the model root (e.g. the
full bf16 repo on a big-GPU machine).

| Variable | Default | Description |
|---|---|---|
| `V2G_IMAGEGEN_PROVIDER` | — | `qwen` = local Qwen-Image-2.1; unset = off |
| `V2G_IMAGEGEN_MODEL` | — | Model root override (diffusers dir / HF id) |
| `V2G_IMAGEGEN_STYLE` | — | Global style prefix prepended to all prompts |
| `V2G_IMAGEGEN_STEPS` | `40` | Denoising steps |
| `V2G_IMAGEGEN_MAX_SIDE` | `1024` | Longest output edge (aspect kept, dims ÷32) |
| `V2G_IMAGEGEN_AUTORESTYLE` | — | Re-draw assets without `-i` (prompt = `design.style`) |

To implement a new provider: subclass `ImageGenProvider` in
`v2g/llm/image_gen.py`, implement `transform()` and `is_available()`, then
wire it into the `get_provider()` factory.

### Scene tree (main.tscn)

The scene is a minimal, template-owned visual-novel root:
- **Main** — `Control` (full rect) with `vn_manager.gd`: builds backgrounds,
  character portraits, the bilingual dialogue box (source line + Chinese
  subtitle), choice buttons, score/hint HUD, and scene-flash transitions at
  runtime from the embedded story JSON
- **GameManager** — Node with `game_manager.gd` (`score_changed`/`add_score` contract)

### Input mapping

- `advance`: Space / Enter (dialogue advance, restart on the end card)
- Left mouse click: advance, or pick a choice option (choice buttons intercept)

### Extending

Generated projects are starting points. Open in Godot 4.x and iterate:
1. Edit the story in `vn_manager.gd`'s `STORY_JSON` (or change `game_design.json` and regenerate)
2. Add choice branches / minigames as extra LLM scripts around the VN runtime
3. Wire `game_manager.gd` goals to win/lose and the alliance map
4. Add audio based on `style` and `atmosphere` descriptions

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
