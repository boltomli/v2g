"""Generate a complete Godot 4.x project from a GameDesign document.

Strategy:
1. Ask the LLM to generate optional GDScript extras from the design context.
2. Fill template-owned essentials: vn_manager.gd (VN runtime with the
   bilingual story) and a contract-compliant game_manager.gd.
3. Generate main.tscn from the VN template (Control root + GameManager).
4. Write project.godot (advance input only), game_design.json, and scripts.
"""

import json
import logging
import subprocess
from pathlib import Path

from v2g.config import settings
from v2g.godot import templates as T
from v2g.llm.analyzer import GameDesign
from v2g.llm.client import chat

log = logging.getLogger(__name__)


def _finalize_with_godot(project_root: Path) -> None:
    """Generation-stage self-check: import assets, then boot the game once.

    A fresh project has no ``.godot/imported`` cache — without this step the
    first launch reports missing textures. Booting headless afterwards surfaces
    scene/script errors at generation time instead of in front of the player.
    Best-effort: missing Godot only downgrades to a warning.
    """
    godot = settings.godot_path
    steps = [
        ([godot, "--headless", "--editor", "--quit-after", "20", "--path", str(project_root)],
         "asset import"),
        ([godot, "--headless", "--path", str(project_root), "--quit-after", "5"],
         "startup validation"),
    ]
    for cmd, what in steps:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("Godot %s skipped: %s", what, e)
            return
        output = (result.stdout or "") + (result.stderr or "")
        problems = [
            ln for ln in output.splitlines()
            if "SCRIPT ERROR" in ln or "Parse Error" in ln or "ERROR: Failed" in ln
        ]
        if problems:
            log.warning("Godot %s reported issues:\n%s", what, "\n".join(problems[:20]))
        else:
            log.info("Godot %s OK", what)


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
    from v2g import cache, runlog

    design_json = design.model_dump_json(indent=2)
    res = chat(T.LLM_SCRIPT_SYSTEM, [design_json], max_tokens=16384, temperature=0.3)
    parsed = _try_load_scripts(res.text)
    if parsed is None and res.cached:
        # Only a cached answer gets one refill; a fresh malformed response
        # is not worth re-requesting (template fallback is cheap).
        cache.invalidate(res.key)
        log.warning("Discarded invalid cached scripts response; refetching once")
        res = chat(
            T.LLM_SCRIPT_SYSTEM, [design_json],
            max_tokens=16384, temperature=0.3, refresh=True,
        )
        parsed = _try_load_scripts(res.text)
    dump = runlog.llm_dump("scripts", res.text)  # keep the raw response either way

    scripts: dict[str, str] = {}
    if parsed is None:
        log.warning(
            "Failed to parse LLM scripts%s",
            f" — raw saved to {dump}" if dump else "",
        )
        return scripts
    for fname, source in parsed.items():
        if not isinstance(source, str):
            continue
        safe_fname = fname if fname.endswith(".gd") else f"{fname}.gd"
        if safe_fname == "vn_manager.gd":
            log.info("Ignoring LLM vn_manager.gd — the VN runtime is template-owned")
            continue
        # Basic sanity: must start with extends or @tool or @export or class_name
        stripped = source.lstrip()
        if not any(stripped.startswith(kw) for kw in ("extends ", "@tool", "@export", "class_name")):
            log.warning("Skipping %s: doesn't look like GDScript", safe_fname)
            continue
        scripts[safe_fname] = source

    return scripts


def _try_load_scripts(text: str) -> dict | None:
    """Parse the scripts JSON envelope; None = unusable (parse layer)."""
    try:
        data = json.loads(_strip_md_fences(text))
    except json.JSONDecodeError as e:
        log.debug("scripts JSON decode failed: %s", e)
        return None
    if not isinstance(data, dict):
        log.warning("LLM returned non-dict for scripts: %s", type(data).__name__)
        return None
    return data


def _ensure_essentials(
    scripts: dict[str, str],
    design: GameDesign,
    assets: dict[str, Path] | None = None,
) -> dict[str, str]:
    """Ensure essential scripts exist; fill missing ones from templates.

    - vn_manager.gd is ALWAYS template-owned (it embeds the bilingual story).
    - game_manager.gd must keep the score_changed/add_score contract the VN
      runtime wires to; anything else falls back to the template.
    """
    gm = scripts.get("game_manager.gd")
    if gm is None or "score_changed" not in gm:
        if gm is not None:
            log.warning("game_manager.gd lacks the score_changed contract — using template")
        scripts["game_manager.gd"] = T.game_manager_script(design)

    scripts["vn_manager.gd"] = T.vn_manager_script(design, assets)
    return scripts


def generate(
    design: GameDesign,
    project_root: Path,
    video_path: Path | None = None,
) -> Path:
    """Create a full Godot project directory. Returns the project path.

    Args:
        design: The game design document.
        project_root: Directory to write the project into — the run directory
            created by ``runlog.start_run`` at pipeline start.
        video_path: Source video for asset extraction. If provided, frames are
            extracted as sprites/backgrounds and placed in assets/.

    Generated structure:
        <project>/
            project.godot        (advance input only — visual novel)
            main.tscn            (Control root: vn_manager + GameManager)
            vn_manager.gd        (template-owned VN runtime with bilingual story)
            game_manager.gd      (LLM or fallback; score_changed contract)
            ...additional LLM-generated scripts...
            assets/              (extracted video frames as backgrounds/portraits)
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
    scripts = _ensure_essentials(scripts, design, assets)
    log.info("Generated %d script(s): %s", len(scripts), ", ".join(sorted(scripts)))

    # 3. project.godot
    (project_root / "project.godot").write_text(
        T.project_dot_godot(design.title), encoding="utf-8"
    )

    # 4. main.tscn — visual-novel root (vn_manager + GameManager)
    (project_root / "main.tscn").write_text(
        T.main_scene(design, scripts), encoding="utf-8"
    )

    # 5. Write all scripts
    for fname, source in scripts.items():
        (project_root / fname).write_text(source, encoding="utf-8")

    # 6. Game design metadata
    (project_root / "game_design.json").write_text(
        design.model_dump_json(indent=2), encoding="utf-8"
    )

    # 7. Import assets and boot once — errors surface at generation time
    _finalize_with_godot(project_root)

    return project_root
