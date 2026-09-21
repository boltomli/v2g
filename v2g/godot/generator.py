"""Generate a complete Godot 4.x project from a GameDesign document.

Strategy:
1. Ask the LLM to generate ALL GDScript files from the full design context.
2. Validate LLM output; fall back to template scripts for any missing essentials.
3. Generate main.tscn programmatically from the design (objects, enemies, dialogue).
4. Write project.godot, game_design.json, and all scripts.
"""

import json
import logging
from pathlib import Path

from v2g.config import settings
from v2g.godot import templates as T
from v2g.llm.analyzer import GameDesign
from v2g.llm.client import chat

log = logging.getLogger(__name__)


def _safe_name(title: str) -> str:
    """Sanitize a game title into a filesystem-safe directory name."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title).strip("_").lower() or "game"


def _strip_md_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return cleaned.strip()


def _generate_scripts(design: GameDesign) -> dict[str, str]:
    """Ask the LLM to generate ALL GDScript scripts from the game design.

    Returns a dict of {filename: source_code}. On any failure, returns empty dict
    so the caller can fall back to templates.
    """
    design_json = design.model_dump_json(indent=2)
    raw = chat(T.LLM_SCRIPT_SYSTEM, [design_json], max_tokens=16384, temperature=0.3)
    cleaned = _strip_md_fences(raw)

    scripts: dict[str, str] = {}
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            log.warning("LLM returned non-dict for scripts: %s", type(parsed).__name__)
            return scripts
        for fname, source in parsed.items():
            if not isinstance(source, str):
                continue
            safe_fname = fname if fname.endswith(".gd") else f"{fname}.gd"
            # Basic sanity: must start with extends or @tool or @export or class_name
            stripped = source.lstrip()
            if not any(stripped.startswith(kw) for kw in ("extends ", "@tool", "@export", "class_name")):
                log.warning("Skipping %s: doesn't look like GDScript", safe_fname)
                continue
            scripts[safe_fname] = source
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning("Failed to parse LLM scripts: %s", e)

    return scripts


def _ensure_essentials(scripts: dict[str, str], design: GameDesign) -> dict[str, str]:
    """Ensure essential scripts exist; fill missing ones from templates."""
    if "player.gd" not in scripts:
        log.info("Fallback: generating player.gd from template")
        scripts["player.gd"] = T.player_script(design)

    if "game_manager.gd" not in scripts:
        log.info("Fallback: generating game_manager.gd from template")
        scripts["game_manager.gd"] = T.game_manager_script(design)

    has_enemies = any(
        c.role in ("antagonist", "boss", "minion", "enemy")
        for c in design.characters
    ) or any(o.role == "enemy" for o in design.objects)

    if has_enemies and "enemy.gd" not in scripts:
        log.info("Fallback: generating enemy.gd from template")
        scripts["enemy.gd"] = T.enemy_script()

    return scripts


def generate(
    design: GameDesign,
    project_root: Path | None = None,
    video_path: Path | None = None,
) -> Path:
    """Create a full Godot project directory. Returns the project path.

    Args:
        design: The game design document.
        project_root: Override project directory.
        video_path: Source video for asset extraction. If provided, frames are
            extracted as sprites/backgrounds and placed in assets/.

    Generated structure:
        <project>/
            project.godot
            main.tscn          (data-driven: player + enemies + environment + UI + sprites)
            player.gd
            game_manager.gd
            enemy.gd           (if enemies exist)
            dialogue_ui.gd     (if dialogue exists)
            ...additional LLM-generated scripts...
            assets/            (extracted video frames as sprites/backgrounds)
            game_design.json
    """
    if project_root is None:
        project_root = settings.output_root / _safe_name(design.title)
    project_root.mkdir(parents=True, exist_ok=True)

    # 1. Extract visual assets from video (if source video available)
    assets: dict[str, Path] = {}
    if video_path and video_path.is_file():
        log.info("Extracting visual assets from video...")
        from v2g.video.asset_extractor import extract_assets

        assets_dir = project_root / "assets"
        assets = extract_assets(video_path, design, assets_dir)
        log.info("Extracted %d asset(s)", len(assets))

        # Optional: transform assets via image gen provider
        from v2g.llm.image_gen import get_provider

        provider = get_provider()
        if provider.is_available() and assets:
            # Future: apply style transformation when instruct is provided
            pass

    # 2. Generate ALL scripts via LLM
    log.info("Generating GDScript files via LLM...")
    scripts = _generate_scripts(design)
    scripts = _ensure_essentials(scripts, design)
    log.info("Generated %d script(s): %s", len(scripts), ", ".join(sorted(scripts)))

    # 3. project.godot
    (project_root / "project.godot").write_text(
        T.project_dot_godot(design.title), encoding="utf-8"
    )

    # 4. main.tscn — data-driven from design + assets
    (project_root / "main.tscn").write_text(
        T.main_scene(design, scripts, assets), encoding="utf-8"
    )

    # 5. Write all scripts
    for fname, source in scripts.items():
        (project_root / fname).write_text(source, encoding="utf-8")

    # 6. Game design metadata
    (project_root / "game_design.json").write_text(
        design.model_dump_json(indent=2), encoding="utf-8"
    )

    return project_root
