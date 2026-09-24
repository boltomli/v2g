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
from v2g.llm import jsonfix
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
        (
            [godot, "--headless", "--editor", "--quit-after", "20", "--path", str(project_root)],
            "asset import",
        ),
        (
            [godot, "--headless", "--path", str(project_root), "--quit-after", "5"],
            "startup validation",
        ),
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
            ln
            for ln in output.splitlines()
            if "SCRIPT ERROR" in ln or "Parse Error" in ln or "ERROR: Failed" in ln
        ]
        if problems:
            log.warning("Godot %s reported issues:\n%s", what, "\n".join(problems[:20]))
        else:
            log.info("Godot %s OK", what)


def _gd_source_ok(source: str) -> bool:
    """Sanity: the text must actually look like GDScript."""
    stripped = source.lstrip()
    return any(stripped.startswith(kw) for kw in ("extends ", "@tool", "@export", "class_name"))


def _check_script(project_root: Path, fname: str) -> list[str] | None:
    """Compile-check one script with Godot; [] = clean, None = Godot unavailable.

    The headless boot only parses scripts the main scene references, so LLM
    extras (alliance/save overlays) would otherwise ship with parse errors
    that only surface when someone opens the project in the editor.
    """
    try:
        result = subprocess.run(
            [
                settings.godot_path,
                "--headless",
                "--path",
                str(project_root),
                "--check-only",
                "--script",
                f"res://{fname}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("Script compile check skipped: %s", e)
        return None
    output = (result.stdout or "") + (result.stderr or "")
    return [
        ln
        for ln in output.splitlines()
        if "SCRIPT ERROR" in ln or "Parse Error" in ln or "ERROR: Failed" in ln
    ]


def _repair_scripts(
    design: GameDesign, scripts: dict[str, str], failures: dict[str, list[str]]
) -> dict[str, str]:
    """One follow-up LLM pass returning corrected source for failing files only."""
    from v2g import runlog

    report = "\n\n".join(
        f"## {fname}\nGodot compile check said:\n"
        + "\n".join(errs)
        + f"\n\nCurrent source:\n```gdscript\n{scripts[fname]}\n```"
        for fname, errs in failures.items()
    )
    ask = (
        "These files failed Godot's compile check. Return a JSON object mapping "
        "filename to its corrected full source, ONLY for the files below "
        "(same output format as before):\n\n" + report
    )
    res = chat(
        T.LLM_SCRIPT_SYSTEM,
        [design.model_dump_json(indent=2), ask],
        temperature=0.3,
    )
    runlog.llm_dump("scripts_repair", res.text)
    parsed = _try_load_scripts(res.text)
    if parsed is None:
        log.warning("Script repair response unusable — falling back")
        return {}
    repaired: dict[str, str] = {}
    for fname, source in parsed.items():
        if not isinstance(source, str):
            continue
        safe_fname = fname if fname.endswith(".gd") else f"{fname}.gd"
        if safe_fname in failures and _gd_source_ok(source):
            repaired[safe_fname] = source
    return repaired


def _validate_scripts(
    project_root: Path, design: GameDesign, scripts: dict[str, str]
) -> dict[str, str]:
    """Compile-check every script, repair once, then fall back — no parse
    error ships. Keeps files on disk in sync with the returned dict.
    """
    from v2g import runlog

    failures: dict[str, list[str]] = {}
    for fname in scripts:
        errs = _check_script(project_root, fname)
        if errs is None:
            return scripts  # Godot unavailable — nothing to enforce
        if errs:
            failures[fname] = errs
            log.warning("Parse errors in %s:\n%s", fname, "\n".join(errs))
    if not failures:
        return scripts

    repairable = {f: e for f, e in failures.items() if f != "vn_manager.gd"}
    if repairable:
        for fname, source in _repair_scripts(design, scripts, repairable).items():
            log.log(runlog.NOTICE, "LLM repair fixed %s", fname)
            (project_root / fname).write_text(source, encoding="utf-8")
            scripts[fname] = source

    for fname in list(failures):
        errs = _check_script(project_root, fname)
        if not errs:
            continue  # repaired cleanly
        if fname == "game_manager.gd":
            log.warning("game_manager.gd still fails compile — using template")
            scripts[fname] = T.game_manager_script(design)
            (project_root / fname).write_text(scripts[fname], encoding="utf-8")
        elif fname == "vn_manager.gd":
            log.error("Template-owned vn_manager.gd fails compile:\n%s", "\n".join(errs))
        elif any(fname in src for f, src in scripts.items() if f != fname):
            log.error(
                "Keeping %s despite parse errors — another script references it:\n%s",
                fname,
                "\n".join(errs),
            )
        else:
            log.warning("Dropping %s (parse errors, unreferenced):\n%s", fname, "\n".join(errs))
            (project_root / fname).unlink(missing_ok=True)
            del scripts[fname]
    return scripts


def _safe_name(title: str) -> str:
    """Sanitize a game title into a filesystem-safe directory name."""
    return (
        "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title).strip("_").lower()
        or "game"
    )


def _asset_reference(design: GameDesign, key: str) -> str:
    """Design-side visual context for an extracted asset key.

    Keys follow ``v2g.video.asset_extractor``: ``background``,
    ``characters/<safe>``, ``objects/<safe>``, ``scenes/<safe>`` — the safe
    name matches :func:`v2g.godot.templates._safe`.
    """
    kind, sep, name = key.partition("/")
    if kind == "background" and design.scenes:
        scene = design.scenes[0]
        return f"{scene.name}: {scene.description}"
    if not sep:
        return ""
    if kind == "characters":
        return next(
            (f"{c.name}, {c.visual}" for c in design.characters if T._safe(c.name) == name), ""
        )
    if kind == "objects":
        return next(
            (f"{o.name}, {o.visual}" for o in design.objects if T._safe(o.name) == name), ""
        )
    if kind == "scenes":
        return next(
            (f"{s.name}: {s.description}" for s in design.scenes if T._safe(s.name) == name), ""
        )
    return ""


def _restyle_assets(
    design: GameDesign,
    assets: dict[str, Path],
    instruct: str | None,
) -> None:
    """Re-draw each extracted asset in a target style, in place.

    Trigger: an explicit ``-i/--instruct`` (mandatory — a request that cannot
    run is reported, never silently dropped) or ``V2G_IMAGEGEN_AUTORESTYLE``
    (auto: the prompt is the design's own style summary of the source video).
    Without a style request the raw frames are the correct output. One failing
    asset keeps its original frame — image generation can never fail a run.
    """
    if not assets:
        return
    from v2g import runlog
    from v2g.llm.image_gen import NullProvider, get_provider

    request = (instruct or "").strip()
    if request:
        style = request
    elif settings.imagegen_autorestyle:
        style = design.style.strip()
        if not style:
            log.log(
                runlog.NOTICE,
                "Auto-restyle on but the design has no style summary — keeping raw frames",
            )
            return
    else:
        return

    provider = get_provider()
    unfulfilled: str | None = None
    if isinstance(provider, NullProvider):
        unfulfilled = "V2G_IMAGEGEN_PROVIDER is unset (raw frames are the configured default)"
    elif not provider.is_available():
        unfulfilled = "the configured imagegen provider is not ready (see its log lines above)"
    if unfulfilled is not None:
        if request:
            log.warning(
                "-i/--instruct given but assets were NOT restyled: %s — install with "
                "`uv sync --extra imagegen` and set V2G_IMAGEGEN_PROVIDER=qwen",
                unfulfilled,
            )
        else:
            log.log(runlog.NOTICE, "Auto-restyle skipped: %s", unfulfilled)
        return

    mode = "" if request else " (auto)"
    log.log(runlog.NOTICE, "Restyling %d asset(s) with imagegen%s", len(assets), mode)
    failures = 0
    for key, path in list(assets.items()):
        try:
            assets[key] = provider.transform(path, style, reference=_asset_reference(design, key))
        except Exception as e:  # noqa: BLE001 — any failure keeps the run's original frame
            failures += 1
            log.warning("Image generation failed for %s — keeping original: %s", key, e)
    if failures and request:
        log.warning(
            "Style instruction given but %d/%d asset(s) kept their original frames",
            failures,
            len(assets),
        )


def _generate_scripts(design: GameDesign) -> dict[str, str]:
    """Ask the LLM to generate ALL GDScript scripts from the game design.

    Returns a dict of {filename: source_code}. On any failure, returns empty dict
    so the caller can fall back to templates.
    """
    from v2g import cache, runlog

    design_json = design.model_dump_json(indent=2)
    res = chat(T.LLM_SCRIPT_SYSTEM, [design_json], temperature=0.3)
    parsed = _try_load_scripts(res.text)
    if parsed is None and res.cached:
        # Only a cached answer gets one refill; a fresh malformed response
        # is not worth re-requesting (template fallback is cheap).
        cache.invalidate(res.key)
        log.warning("Discarded invalid cached scripts response; refetching once")
        res = chat(
            T.LLM_SCRIPT_SYSTEM,
            [design_json],
            temperature=0.3,
            refresh=True,
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
        if not _gd_source_ok(source):
            log.warning("Skipping %s: doesn't look like GDScript", safe_fname)
            continue
        scripts[safe_fname] = source

    return scripts


def _try_load_scripts(text: str) -> dict | None:
    """Parse the scripts JSON envelope; None = unusable (parse layer).

    On decode failure falls back to the shared truncation repair: a
    max_tokens cut mid-envelope still yields every complete script before
    the cut (the cut file itself is caught later by the compile check).
    """
    for cand in jsonfix.json_candidates(text):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError as e:
            log.debug("scripts JSON decode failed: %s", e)
            continue
        if not isinstance(data, dict):
            log.warning("LLM returned non-dict for scripts: %s", type(data).__name__)
            return None
        return data
    return jsonfix.salvage(text)


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
    *,
    instruct: str | None = None,
) -> Path:
    """Create a full Godot project directory. Returns the project path.

    Args:
        design: The game design document.
        project_root: Directory to write the project into — the run directory
            created by ``runlog.start_run`` at pipeline start.
        video_path: Source video for asset extraction. If provided, frames are
            extracted as sprites/backgrounds and placed in assets/.
        instruct: Optional style instruction — when set (and an image-gen
            provider is configured), extracted assets are restyled first.

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
    video_size: tuple[int, int] | None = None
    if video_path and video_path.is_file():
        log.info("Extracting visual assets from video...")
        from v2g.video.asset_extractor import extract_assets, probe_video_size

        video_size = probe_video_size(video_path)
        assets_dir = project_root / "assets"
        assets = extract_assets(video_path, design, assets_dir)
        log.info("Extracted %d asset(s)", len(assets))

        # Optional: restyle extracted assets with the local image-gen provider
        _restyle_assets(design, assets, instruct)

    # 2. Generate ALL scripts via LLM
    log.info("Generating GDScript files via LLM...")
    scripts = _generate_scripts(design)
    scripts = _ensure_essentials(scripts, design, assets)
    log.info("Generated %d script(s): %s", len(scripts), ", ".join(sorted(scripts)))

    # 3. project.godot — window aspect follows the source video
    (project_root / "project.godot").write_text(
        T.project_dot_godot(design.title, *T.viewport_for(video_size)),
        encoding="utf-8",
    )

    # 4. main.tscn — visual-novel root (vn_manager + GameManager)
    (project_root / "main.tscn").write_text(T.main_scene(design, scripts), encoding="utf-8")

    # 5. Write all scripts
    for fname, source in scripts.items():
        (project_root / fname).write_text(source, encoding="utf-8")

    # 6. Game design metadata
    (project_root / "game_design.json").write_text(
        design.model_dump_json(indent=2), encoding="utf-8"
    )

    # 7. Compile-check every script — the boot below never parses scripts the
    # main scene doesn't reference — repair/fallback so parse errors never ship
    scripts = _validate_scripts(project_root, design, scripts)

    # 8. Import assets and boot once — errors surface at generation time
    _finalize_with_godot(project_root)

    return project_root
