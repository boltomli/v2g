"""Analyze video (frames or full video) and produce a structured game design document."""

from pathlib import Path

from pydantic import BaseModel

from v2g.llm.client import chat

# ── Shared schema ───────────────────────────────────────────────────────────

_JSON_SCHEMA = """\
{
  "title": "string - game title",
  "genre": "string - e.g. platformer, puzzle, action, endless-runner, adventure",
  "mechanics": ["string - core gameplay mechanics"],
  "objects": [
    {"name": "string", "role": "string - player/enemy/obstacle/collectible/environment",
     "visual": "string - brief visual description",
     "behavior": "string - how it moves/acts/interacts (optional)"}
  ],
  "levels": [
    {"name": "string", "description": "string", "goals": ["string"],
     "layout": "string - spatial layout description (optional)"}
  ],
  "controls": ["string - control descriptions"],
  "style": "string - art/visual style summary",
  "summary": "string - 2-3 sentence pitch",
  "physics": "string - physics rules (gravity, collisions, etc.) (optional)",
  "progression": "string - how difficulty or content scales (optional)"
}
"""

# ── System prompts ──────────────────────────────────────────────────────────

_SYSTEM_FRAMES = f"""\
You are a game designer. The user will provide key frames extracted from a video.
Analyze the visuals carefully and produce a game design document as JSON.

{_JSON_SCHEMA}

Return ONLY the JSON, no markdown fences, no commentary.
"""

_SYSTEM_VIDEO = f"""\
You are an expert game designer and video analyst. The user will provide a full video.
Watch it carefully — pay attention to:
- Camera movement, transitions, scene composition
- Character/object actions, timing, sequencing
- Spatial relationships and depth
- Color palette, lighting changes, mood shifts
- Any text, UI elements, or overlays visible
- Sound cues or rhythm patterns (infer from visual timing)

Then produce a detailed game design document as JSON:

{_JSON_SCHEMA}

Guidelines for detailed analysis:
- "objects" MUST include every distinct visual element with its behavior and spatial role
- "levels" MUST describe layout, progression triggers, and environmental hazards
- "mechanics" MUST capture timing-sensitive interactions (dodging, rhythm, combos)
- "physics" MUST describe movement constraints (gravity, momentum, collisions)
- "progression" MUST describe how the challenge evolves
- "style" MUST be specific enough for an artist to replicate the look

Return ONLY the JSON, no markdown fences, no commentary.
"""


# ── Pydantic models ────────────────────────────────────────────────────────

class GameObject(BaseModel):
    name: str
    role: str
    visual: str
    behavior: str = ""


class LevelDesign(BaseModel):
    name: str
    description: str
    goals: list[str]
    layout: str = ""


class GameDesign(BaseModel):
    title: str
    genre: str
    mechanics: list[str]
    objects: list[GameObject]
    levels: list[LevelDesign]
    controls: list[str]
    style: str
    summary: str
    physics: str = ""
    progression: str = ""


def _parse(raw: str) -> GameDesign:
    """Clean LLM output and parse into GameDesign."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return GameDesign.model_validate_json(cleaned.strip())


def analyze(frames: list[Path]) -> GameDesign:
    """Analyze video keyframes → GameDesign (fast mode)."""
    parts: list[str | Path] = [
        (
            "Here are key frames extracted from a video. "
            "Analyze the visuals and generate a game design document.\n"
        )
    ]
    parts.extend(frames)
    return _parse(chat(_SYSTEM_FRAMES, parts))


def analyze_video(video_path: Path) -> GameDesign:
    """Analyze a full video file → GameDesign (detailed mode).

    Sends the video directly to a model with video understanding support.
    Produces richer analysis: behavior, physics, progression, spatial layout.
    """
    parts: list[str | Path] = [
        (
            "Here is a video. Watch it carefully and produce a detailed game design document.\n"
        ),
        video_path,
    ]
    return _parse(chat(_SYSTEM_VIDEO, parts, max_tokens=16384, temperature=0.3))
