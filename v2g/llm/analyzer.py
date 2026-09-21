"""Analyze video frames and produce a structured game design document via LLM."""

from pathlib import Path

from pydantic import BaseModel

from v2g.llm.client import chat

_SYSTEM = """\
You are a game designer. The user will provide frames extracted from a video.
Analyze them and produce a game design document as JSON with these fields:

{
  "title": "string - game title",
  "genre": "string - e.g. platformer, puzzle, action, endless-runner, adventure",
  "mechanics": ["string - core gameplay mechanics"],
  "objects": [
    {"name": "string", "role": "string - player/enemy/obstacle/collectible/environment",
     "visual": "string - brief visual description"}
  ],
  "levels": [
    {"name": "string", "description": "string", "goals": ["string"]}
  ],
  "controls": ["string - control descriptions"],
  "style": "string - art/visual style summary",
  "summary": "string - 2-3 sentence pitch"
}

Return ONLY the JSON, no markdown fences, no commentary.
"""


class GameObject(BaseModel):
    name: str
    role: str
    visual: str


class LevelDesign(BaseModel):
    name: str
    description: str
    goals: list[str]


class GameDesign(BaseModel):
    title: str
    genre: str
    mechanics: list[str]
    objects: list[GameObject]
    levels: list[LevelDesign]
    controls: list[str]
    style: str
    summary: str


def analyze(frames: list[Path]) -> GameDesign:
    """Send extracted frames to the LLM and return a parsed GameDesign."""
    parts: list[str | Path] = [
        (
            "Here are the key frames extracted from a video. "
            "Analyze the visuals and generate a game design document.\n"
        )
    ]
    parts.extend(frames)

    raw = chat(_SYSTEM, parts)
    # Strip markdown fences if the LLM wraps them anyway
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return GameDesign.model_validate_json(cleaned.strip())
