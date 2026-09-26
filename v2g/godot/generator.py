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
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from v2g.config import settings
from v2g.godot import templates as T
from v2g.llm import jsonfix
from v2g.llm.analyzer import GameDesign, SceneDesign, safe_name
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
    repaired: dict[str, str] = {}
    if repairable:
        repaired = _repair_scripts(design, scripts, repairable)
        for fname, source in repaired.items():
            (project_root / fname).write_text(source, encoding="utf-8")
            scripts[fname] = source

    for fname in list(failures):
        errs = _check_script(project_root, fname)
        if not errs:
            if fname in repaired:
                # Only now is the repair proven — a source that still failed
                # the re-check below must never be announced as fixed.
                log.log(runlog.NOTICE, "LLM repair fixed %s", fname)
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


# Kind-specific redraw mandates — each asset kind must change in its own way:
# characters get a new look/clothes/pose, objects become standalone item art,
# scenes are re-laid-out as a whole. The SUBJECT line is authoritative for what
# the thing IS (the design is rewritten to the theme); the source frame only
# shows what must NOT be copied — presentation, framing, pixels.
_DIRECTIVES: dict[str, str] = {
    "characters": (
        "Redraw this CHARACTER as a square half-body portrait (head and torso, cut at the "
        "waist) in a dynamic pose, drawn fresh — never lifted from the source frame. Outfit, "
        "hairstyle and colors follow the SUBJECT below; framing, stance and background must "
        "differ from the source frame. Fill the square with the figure on a plain background."
    ),
    "objects": (
        "Draw this OBJECT alone as standalone item art: one item as described in the "
        "SUBJECT below, cut out and centered on a plain background — no scenery, no "
        "characters, no other props in frame. Draw it fresh to fit the theme instead of "
        "lifting it out of the source frame."
    ),
    "scenes": (
        "Redesign this LOCATION as a wide game background: a fresh layout, palette, "
        "lighting and props — never the source frame's composition. The SUBJECT below "
        "defines what the place is; its whole look changes around that role. No "
        "characters painted into the scene."
    ),
}
_DIRECTIVES["background"] = _DIRECTIVES["scenes"]  # the main background is a scene view

# Stage 2 without a user theme: the art only has to differ from the source.
_NO_THEME_STYLE = "a cohesive art style that differs from the source video"


def _scene_subject(scene: SceneDesign) -> str:
    parts = [f"{scene.name}: {scene.description}"]
    if scene.layout:
        parts.append(f"layout: {scene.layout}")
    if scene.visual_theme:
        parts.append(f"theme: {scene.visual_theme}")
    return " | ".join(parts)


def _asset_subject(design: GameDesign, key: str) -> str:
    """Design-side subject description for an extracted asset key.

    Keys follow ``v2g.video.asset_extractor``: ``background``,
    ``characters/<safe>``, ``objects/<safe>``, ``scenes/<safe>`` — the safe
    name matches :func:`v2g.godot.templates._safe`.
    """
    kind, sep, name = key.partition("/")
    if kind == "background" and design.scenes:
        return _scene_subject(design.scenes[0])
    if not sep:
        return ""
    if kind == "characters":
        return next(
            (
                f"{c.name} ({c.role}): {c.visual}"
                + (f"; actions: {c.behavior}" if c.behavior else "")
                for c in design.characters
                if safe_name(c.name) == name
            ),
            "",
        )
    if kind == "objects":
        return next(
            (
                f"{o.name} ({o.role}): {o.visual}"
                for o in design.objects
                if safe_name(o.name) == name
            ),
            "",
        )
    if kind == "scenes":
        return next((_scene_subject(s) for s in design.scenes if safe_name(s.name) == name), "")
    return ""


def _redraw_prompt(design: GameDesign, key: str, style: str) -> str:
    """Full redraw brief for one asset: theme, art style, kind directive, subject.

    The theme is stated twice — opening line and closing mandate — and the
    design's stage-2 ``style`` rides along: the reference candidate is fed the
    source frame as visual context, so a theme mentioned only once at the top
    loses to the video's own palette.
    """
    kind = key.partition("/")[0]
    blocks = [f"Redraw in this theme: {style}"]
    if design.style.strip():
        blocks.append(f"ART STYLE: {design.style}")
    blocks.append(_DIRECTIVES.get(kind, _DIRECTIVES["scenes"]))
    subject = _asset_subject(design, key)
    if subject:
        blocks.append(f"SUBJECT: {subject}")
    blocks.append(
        f"THEME MANDATE: {style}. Every color, material, garment, hairstyle, prop "
        f"and lighting choice must belong to this theme — where the source frame "
        f"disagrees with the theme, the theme wins; do not reproduce the source "
        f"frame's palette or costumes."
    )
    blocks.append(
        "NO BURNED-IN TEXT: no subtitles, captions, watermarks, logos or UI overlays "
        "anywhere in the image. Whatever text the source frame carries is not part of "
        "the artwork — the game renders its own text at runtime (in-world signage only "
        "when the SUBJECT asks for it)."
    )
    return "\n\n".join(blocks)


_CAPTION_SYSTEM = "You caption frames for a redraw brief. Answer with plain text only, no preamble."

# The caption and the judge are one-sentence answers, but a reasoning model
# spends the cap thinking before it answers: at max_tokens=200 it returned an
# empty body every time (finish=length) — no captions, no usable verdicts. Same
# floor asset_extractor._VERIFY_TOKENS proved sufficient.
_VISION_TOKENS = 6000


def _frame_caption(path: Path) -> str:
    """The source frame as text (i2t2i): what it shows, never how to draw it.

    The ref candidate is fed the frame as visual context, and pixels out-shout
    text — restating the frame as a sentence keeps the brief in charge. Any
    failure returns empty: the reference note then lacks the description,
    never the authority rules.
    """
    ask = (
        "Describe what this source frame shows: the subject and the identity cues that "
        "say WHAT it is (person, item, place). One or two plain sentences. No style, "
        "palette, composition or drawing advice of any kind. Ignore all burned-in text "
        "— subtitles, captions, watermarks, logos, on-screen titles: they are not part "
        "of the subject and must never appear in your description."
    )
    try:
        res = chat(_CAPTION_SYSTEM, [ask, path], temperature=0.0, max_tokens=_VISION_TOKENS)
    except Exception as e:  # noqa: BLE001 — a caption enriches the brief, it never gates it
        log.info("imagegen: no source-frame caption (%s) — brief alone", e)
        return ""
    return " ".join(res.text.split())[:400]


def _reference_prompt(brief: str, caption: str) -> str:
    """The ref candidate's brief: text binding, the attached frame advisory.

    The frame rides along as visual context, so this note re-asserts i2t2i —
    the drawing is driven by the text (the brief plus what the frame shows,
    captioned above), never by the frame's own pixels.
    """
    note = (
        "SOURCE FRAME — REFERENCE ONLY, NEVER A TEMPLATE: the attached image is the "
        "original frame this asset was cut from, shown only so you recognise what the "
        "SUBJECT is. Every line of the brief above is binding and outranks the frame — "
        "where they disagree, the frame loses."
    )
    if caption:
        note += f" What the frame shows: {caption}."
    note += (
        " Draw from the text: never copy the frame's composition, palette, costume, "
        "pose or lighting — and never reproduce any text the frame carries (subtitles, "
        "captions, watermark): the drawing contains no text at all."
    )
    return f"{brief}\n\n{note}"


_JUDGE_SYSTEM = (
    "You compare two AI redraws of one game asset against a brief and pick the better one. "
    "Answer only with the requested JSON."
)


def _judge_redraw(style: str, brief: str, original: Path, ref: Path, text: Path) -> str | None:
    """Let a vision model pick between the reference and the text-only redraw.

    Returns ``"ref"`` / ``"text"``, or None when the judge cannot answer — the
    caller then keeps whichever candidate changed the source most.
    """
    ask = (
        f"Brief:\n{brief}\n\n"
        f"Theme: {style}\n\n"
        "Three images: (0) the source frame, (1) candidate 'ref' drawn WITH the source "
        "frame as visual reference, (2) candidate 'text' drawn from the brief alone.\n"
        "Pick the better candidate. It must follow the theme, show the SUBJECT as "
        "described, and be drawn fresh — composed for the brief, not copied from the "
        "source frame (different pose for a character, the object standing alone, a "
        "redesigned scene). Reject a candidate that merely reproduces the source "
        "frame's pixels, or one that fails to show the subject.\n"
        'JSON only: {"pick": "ref" | "text", "reason": "one sentence"}'
    )
    try:
        res = chat(
            _JUDGE_SYSTEM, [ask, original, ref, text], temperature=0.0, max_tokens=_VISION_TOKENS
        )
        data = jsonfix.salvage(res.text)
    except Exception as e:  # noqa: BLE001 — a judge outage must never fail the run
        log.warning("Redraw judge unavailable (%s)", e)
        return None
    pick = data.get("pick") if isinstance(data, dict) else None
    if pick not in ("ref", "text"):
        log.warning("Redraw judge answered unusably (%r) — using visual difference", res.text[:120])
        return None
    log.info("Redraw judge picked %s: %s", pick, data.get("reason", ""))
    return pick


def _visual_difference(a: Path, b: Path) -> float:
    """0..1 mean channel difference of 32×32 thumbnails — how far a redraw moved."""
    from PIL import Image

    with Image.open(a) as ia, Image.open(b) as ib:
        pa = ia.convert("RGB").resize((32, 32)).tobytes()
        pb = ib.convert("RGB").resize((32, 32)).tobytes()
    return sum(abs(x - y) for x, y in zip(pa, pb)) / (len(pa) * 255)


@contextmanager
def _candidates_dir(enabled: bool) -> Iterator[Path | None]:
    """Where both A/B candidates are staged; None when disabled.

    Inside a run they land in ``<run>/work/imagegen/`` (kept for comparison);
    outside a run they go to a scratch directory removed on exit. Never raises:
    when staging fails, the caller falls back to a single candidate written
    beside the frame.
    """
    from v2g import runlog

    if not enabled:
        yield None
        return
    run = runlog.run_dir()
    if run is not None:
        try:
            kept = run / "work" / "imagegen"
            kept.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning(
                "imagegen: cannot stage A/B candidates (%s) — drawing one candidate beside "
                "the frame",
                e,
            )
            yield None
            return
        yield kept  # outside the try: consumer exceptions pass through untouched
        return
    try:
        scratch = Path(tempfile.mkdtemp(prefix="v2g-imagegen-"))
    except OSError as e:
        log.warning(
            "imagegen: cannot stage A/B candidates (%s) — drawing one candidate beside the frame", e
        )
        yield None
        return
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _replace(src: Path, dst: Path) -> None:
    """Overwrite *dst* with *src* atomically — a failed copy never truncates *dst*."""
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _redraw_path(path: Path) -> Path:
    """Where a redrawn asset lands: beside the frame, never on top of it."""
    return path.with_name(f"{path.stem}.redraw{path.suffix}")


def restyle_assets(
    design: GameDesign,
    assets: dict[str, Path],
    instruct: str | None,
) -> None:
    """Stage 2 (image half): re-draw every extracted asset, beside the original.

    Runs after the re-skin, with or without a theme: ``instruct`` supplies the
    theme when there is one, otherwise the art only has to differ from the
    source video (the briefs already forbid lifting its framing). The stage is
    skipped only when no image-gen backend can run — and that is always said
    out loud, never silently.

    Every asset is redrawn from a kind-specific brief (new look/clothes/pose for
    characters, the object alone, a redesigned scene). The result is written to
    a sibling ``*.redraw.png`` and the asset key is repointed at it, so the
    extracted frame survives for comparison and for a redo. With
    ``V2G_IMAGEGEN_AB`` each asset gets two candidates — one drawn with the
    source frame as visual context (its brief stays binding: the frame is
    captioned into text and demoted to a reference, never a template), one
    from the brief alone — and a vision judge keeps the better; both are
    staged under ``<run>/work/imagegen/`` during a run (a scratch directory
    otherwise) for comparison. One failing asset keeps its original frame —
    image generation can never fail a run.
    """
    if not assets:
        return
    from v2g import runlog
    from v2g.llm.image_gen import NullProvider, get_provider

    request = (instruct or "").strip()
    style = request or _NO_THEME_STYLE

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
            log.log(runlog.NOTICE, "Redraw skipped: %s — assets stay extracted frames", unfulfilled)
        return

    log.log(runlog.NOTICE, "Restyling %d asset(s) with imagegen", len(assets))
    failures = 0
    with _candidates_dir(settings.imagegen_ab) as cand_dir:
        modes = (("ref", True), ("text", False)) if cand_dir else (("ref", True),)
        for key, path in list(assets.items()):
            brief = _redraw_prompt(design, key, style)
            # ref candidate: the brief stays binding and the frame enters as
            # text first (caption) — i2t2i, with the image as a reference.
            ref_brief = _reference_prompt(brief, _frame_caption(path))
            redraw = _redraw_path(path)  # the extracted frame is never written to
            made: dict[str, Path] = {}
            for name, condition in modes:
                dest = cand_dir / f"{key.replace('/', '__')}.{name}.png" if cand_dir else redraw
                try:
                    prompt = ref_brief if condition else brief
                    made[name] = provider.transform(path, prompt, condition=condition, dest=dest)
                except Exception as e:  # noqa: BLE001 — a failed candidate just drops out
                    log.warning("Image generation failed for %s (%s): %s", key, name, e)
            if not made:
                failures += 1
                continue
            if len(made) == 1:
                pick = next(iter(made))
            else:
                pick = _judge_redraw(style, brief, path, made["ref"], made["text"])
                if pick is None:
                    try:
                        pick = max(made, key=lambda m: _visual_difference(path, made[m]))
                        log.info(
                            "imagegen: no judge pick for %s — kept the redraw that "
                            "changed the source most (%s)",
                            key,
                            pick,
                        )
                    except Exception as e:  # noqa: BLE001 — unreadable candidate
                        pick = next(iter(made))
                        log.warning(
                            "imagegen: cannot compare candidates for %s (%s) — keeping the %s one",
                            key,
                            e,
                            pick,
                        )
            out = made[pick]
            if out != redraw:
                try:
                    _replace(out, redraw)  # atomic: a failed copy leaves the frame intact
                except Exception as e:  # noqa: BLE001 — any failure keeps the run's original frame
                    failures += 1
                    log.warning("Image generation failed for %s — keeping original: %s", key, e)
                    continue
                out = redraw
            assets[key] = out  # the game reads the redraw; the frame stays on disk
            staged = cand_dir if runlog.run_dir() is not None and len(made) == 2 else ""
            log.info(
                "imagegen: %s ← %s redraw%s",
                key,
                pick,
                f" (candidates in {staged})" if staged else "",
            )
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
    *,
    assets: dict[str, Path] | None = None,
    video_size: tuple[int, int] | None = None,
) -> Path:
    """Stage 3: create the Godot project — game flow and copy — and return its path.

    Stages 1 (analyze the original + extract its assets) and 2 (re-skin and
    redraw, theme or not) already ran in the pipeline, so this function only
    turns the finished design plus the finished assets into a project.

    Args:
        design: The design as it stands **after** the stage-2 re-skin.
        project_root: Directory to write the project into — the run directory
            created by ``runlog.start_run`` at pipeline start.
        assets: Stage 1's extracted sprites/stills, repointed at stage 2's
            redraws when those ran.
        video_size: Source video's pixel size — the window follows its aspect.

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
    assets = dict(assets or {})

    # 1. Generate ALL scripts via LLM
    log.info("Generating GDScript files via LLM...")
    scripts = _generate_scripts(design)
    scripts = _ensure_essentials(scripts, design, assets)
    log.info("Generated %d script(s): %s", len(scripts), ", ".join(sorted(scripts)))

    # 2. project.godot — window aspect follows the source video
    (project_root / "project.godot").write_text(
        T.project_dot_godot(design.title, *T.viewport_for(video_size)),
        encoding="utf-8",
    )

    # 3. main.tscn — visual-novel root (vn_manager + GameManager)
    (project_root / "main.tscn").write_text(T.main_scene(design, scripts), encoding="utf-8")

    # 4. Write all scripts
    for fname, source in scripts.items():
        (project_root / fname).write_text(source, encoding="utf-8")

    # 5. Game design metadata
    (project_root / "game_design.json").write_text(
        design.model_dump_json(indent=2), encoding="utf-8"
    )

    # 6. Compile-check every script — the boot below never parses scripts the
    # main scene doesn't reference — repair/fallback so parse errors never ship
    scripts = _validate_scripts(project_root, design, scripts)

    # 7. Import assets and boot once — errors surface at generation time
    _finalize_with_godot(project_root)

    return project_root
