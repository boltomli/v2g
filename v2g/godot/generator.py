"""Generate a complete Godot 4.x project from a GameDesign document."""

import json
from pathlib import Path

from v2g.config import settings
from v2g.godot import templates as T
from v2g.llm.analyzer import GameDesign
from v2g.llm.client import chat


def _safe_name(title: str) -> str:
    """Sanitize a game title into a filesystem-safe directory name."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title).strip("_").lower() or "game"


def _generate_extra_scripts(design: GameDesign) -> dict[str, str]:
    """Ask the LLM for additional scripts beyond the built-in templates."""
    design_json = design.model_dump_json(indent=2)
    raw = chat(T.LLM_SCRIPT_SYSTEM, [design_json])
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return json.loads(cleaned.strip())


def generate(design: GameDesign, project_root: Path | None = None) -> Path:
    """Create a full Godot project directory. Returns the project path.

    Generated structure:
        <project>/
            project.godot
            main.tscn
            player.gd
            enemy.gd          (if enemies exist)
            game_manager.gd
            ...extra LLM scripts...
    """
    if project_root is None:
        project_root = settings.output_root / _safe_name(design.title)
    project_root.mkdir(parents=True, exist_ok=True)

    has_enemies = any(o.role == "enemy" for o in design.objects)

    # 1. project.godot
    (project_root / "project.godot").write_text(T.project_dot_godot(design.title), encoding="utf-8")

    # 2. main scene
    (project_root / "main.tscn").write_text(T.main_scene(), encoding="utf-8")

    # 3. Built-in scripts
    (project_root / "player.gd").write_text(T.player_script(design), encoding="utf-8")
    (project_root / "game_manager.gd").write_text(T.game_manager_script(design), encoding="utf-8")
    if has_enemies:
        (project_root / "enemy.gd").write_text(T.enemy_script(), encoding="utf-8")

    # 4. LLM-generated extra scripts (UI, collectibles, level-specific logic, etc.)
    try:
        extra = _generate_extra_scripts(design)
        for fname, source in extra.items():
            safe_fname = fname if fname.endswith(".gd") else f"{fname}.gd"
            (project_root / safe_fname).write_text(source, encoding="utf-8")
    except (json.JSONDecodeError, KeyError):
        # Extra scripts are best-effort; the project is playable without them
        pass

    # 5. Write the game design as metadata
    (project_root / "game_design.json").write_text(design.model_dump_json(indent=2), encoding="utf-8")

    return project_root
