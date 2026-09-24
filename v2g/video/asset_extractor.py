"""Extract visual assets from video frames based on LLM analysis results.

Uses ffmpeg to:
- Extract scene background frames from representative timestamps
- Extract character/object sprite frames using spatial hints from the design
- Crop regions based on the LLM's spatial descriptions (English and Chinese
  framing words both: left/right/top/center, 左/右/上/正中 …)
- Anchor object sprites to the scene that names them and sample that scene's
  establishing shot — the shot starting at the boundary (video start / cut)
  nearest the scene's timeline cell; set dressing lives in that opening wide,
  not in the dialogue close-ups that follow it
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import zlib
from pathlib import Path

from v2g.config import settings
from v2g.llm.analyzer import GameDesign, GameObject

log = logging.getLogger(__name__)

# ── Crop region helpers ──────────────────────────────────────────────────────


def _parse_spatial_hint(hint: str) -> tuple[float, float, float, float]:
    """Convert a natural-language spatial hint into a (x_frac, y_frac, w_frac, h_frac) crop.

    Understands the English hints from the schema docs AND the Chinese hints
    LLMs actually emit for Chinese-language designs — English-only matching
    silently turned every Chinese hint into a full-frame (no-op) crop.

    Examples:
        "left side of frame"       → (0.0, 0.0, 0.4, 1.0)
        "画面左侧近景，前景层"        → (0.0, 0.0, 0.4, 1.0)
        "right third"              → (0.6, 0.0, 0.4, 1.0)
        "center of frame"          → (0.2, 0.1, 0.6, 0.8)
        "拱门上方正中，顶部前景层"     → (0.25, 0.0, 0.5, 0.45)
        "bottom half"              → (0.0, 0.55, 1.0, 0.45)
        "top-left corner"          → (0.0, 0.0, 0.4, 0.45)
    """
    hint_lower = hint.lower()
    has_left = "left" in hint_lower or "左" in hint
    has_right = "right" in hint_lower or "右" in hint
    has_top = "top" in hint_lower or "上" in hint or "顶" in hint
    has_bottom = "bottom" in hint_lower or "下" in hint or "底" in hint
    has_center = "center" in hint_lower or "middle" in hint_lower or "中" in hint
    has_chest = "胸" in hint
    has_waist = "腰" in hint

    # Start with full frame
    x, y, w, h = 0.0, 0.0, 1.0, 1.0

    # "center" alone (no directional modifier) → centered box
    if has_center and not any((has_left, has_right, has_top, has_bottom, has_chest, has_waist)):
        return (0.2, 0.1, 0.6, 0.8)

    # Horizontal — directional words win over a bare "center/中" (中景/中层 are
    # layer words, not framing words: only fall through to center when neither
    # left nor right was given).
    if has_left:
        x, w = 0.0, 0.4
    elif has_right:
        x, w = 0.6, 0.4
    elif has_center:
        x, w = 0.25, 0.5

    # Vertical — body部位 (胸前/腰侧) crop to the wear region, so a worn prop
    # sprite is dominated by the prop instead of the wearer's full figure.
    if has_top:
        y, h = 0.0, 0.45
    elif has_bottom:
        y, h = 0.55, 0.45
    elif has_chest:
        y, h = 0.25, 0.35
    elif has_waist:
        y, h = 0.45, 0.3
    elif has_center and w == 1.0:
        y, h = 0.15, 0.7

    return x, y, w, h


def _get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def probe_video_size(video_path: Path) -> tuple[int, int] | None:
    """First video stream's display (width, height); None when probing fails.

    Honors a rotation side-data entry so phone-shot portrait footage stored
    with landscape pixels still reports its displayed orientation.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,side_data_list",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        streams = json.loads(result.stdout).get("streams") or []
        if not streams:
            return None
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
        for side in streams[0].get("side_data_list") or []:
            if int(side.get("rotation", 0)) % 180 != 0:
                width, height = height, width
                break
        return width, height
    except (
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        KeyError,
        TypeError,
    ):
        return None


# ── Core extraction ──────────────────────────────────────────────────────────


def _extract_frame(video_path: Path, timestamp: float, out_path: Path) -> Path:
    """Extract a single frame at *timestamp* seconds."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )
    return out_path


def _crop_image(src: Path, dst: Path, region: tuple[float, float, float, float]) -> Path:
    """Crop an image using fractional coordinates (x, y, w, h) each in [0, 1]."""
    x, y, w, h = region
    # ffmpeg crop filter uses pixels, but we express as percentages
    crop_filter = f"crop=iw*{w:.2f}:ih*{h:.2f}:iw*{x:.2f}:ih*{y:.2f}"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", crop_filter, "-q:v", "2", str(dst)],
        capture_output=True,
        check=True,
    )
    return dst


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes — used to prove two assets are not identical."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamps_for_entity(duration: float, count: int = 3, seed: int = 0) -> list[float]:
    """Return *count* evenly-spaced timestamps inside a *seed*-selected window.

    Each entity gets a 40%-of-timeline window offset by its seed, so different
    entities sample different moments. A shared schedule made every character
    sprite a byte-identical frame (the duplicate-asset bug).
    """
    if duration <= 1.0:
        return [duration / 2]
    # Full working range: 10%..90% of the video
    full_start = duration * 0.1
    full = duration * 0.8
    window = full * 0.4
    offset = ((seed % 1000) / 1000.0) * (full - window)
    start = full_start + offset
    if count == 1:
        return [start + window / 2]
    step = window / (count - 1)
    return [start + i * step for i in range(count)]


# ── Scene anchoring (objects) ────────────────────────────────────────────────

_KEY_OBJECT_ROLES = frozenset({"collectible", "obstacle", "vehicle", "weapon", "decoration"})


_ROLE_ALIASES = {  # Chinese role glosses LLMs emit → schema role vocabulary
    "装饰": "decoration",
    "身份标识": "decoration",
    "遮挡物": "obstacle",
    "障碍": "obstacle",
    "道具": "collectible",
    "武器": "weapon",
    "载具": "vehicle",
    "旁白": "narrator",
}


def _role_tokens(role: str) -> set[str]:
    """Normalized role tokens: every "/"-separated token, glosses folded in.

    LLMs gloss roles in mixed languages and either order — "decoration /
    身份标识", "身份标识 / 互动道具", "装饰 / 遮挡物" — so filters test set
    membership instead of comparing one side of the slash.
    """
    tokens: set[str] = set()
    for raw in role.split("/"):
        token = raw.strip().lower()
        tokens.add(token)
        tokens.update(en for zh, en in _ROLE_ALIASES.items() if zh in token)
    return tokens


def _bigrams(text: str) -> set[str]:
    """Character bigrams — the overlap unit for matching names to scene prose."""
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _scene_window(duration: float, index: int, count: int) -> tuple[float, float]:
    """Timeline cell of scene *index*: the timeline split into *count* equal
    cells centered on each scene's representative stamp (``duration*index/count``
    — the same stamp its background frame is grabbed from), the first cell
    starting at 0 and the last ending at *duration*.
    """
    if count <= 1:
        return (0.0, duration)
    lo = 0.0 if index == 0 else duration * (index - 0.5) / count
    hi = duration if index == count - 1 else duration * (index + 0.5) / count
    return (lo, hi)


def _shot_boundaries(video_path: Path, threshold: float) -> list[float]:
    """Cut timestamps (seconds) from ffmpeg scene detection; [] when none/failing.

    Same scene-change metric as keyframe extraction; one decode pass, only run
    when at least one object anchors to a scene.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "info",
                "-i",
                str(video_path),
                "-vf",
                f"select='gt(scene,{threshold})',showinfo",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    stamps = re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr + result.stdout)
    return sorted({float(s) for s in stamps})


def _establishing_shot(window: tuple[float, float], cuts: list[float]) -> tuple[float, float]:
    """Shot starting at the boundary NEAREST the cell start — the scene's wide.

    Scene cells are equal-duration and misalign cuts (measured: the prop wide
    starts0.4 s before its cell while a4.5 s dialogue beat outlasted it; the
    synthetic red/blue fixture put a cell edge mid-tail). Snapping to the
    nearest boundary — video start or detected cut, ties → the later one —
    recovers the real establishing shot. Falls back to the cell when nothing
    overlaps it.
    """
    lo, hi = window
    boundaries = sorted({0.0, *cuts})
    best_key: tuple[float, float] | None = None
    best_seg: tuple[float, float] | None = None
    for i, start in enumerate(boundaries):
        if start >= hi:
            break
        end = boundaries[i + 1] if i + 1 < len(boundaries) else float("inf")
        if end <= lo:  # shot finished before the cell → no overlap
            continue
        key = (abs(start - lo), -start)
        if best_key is None or key < best_key:
            best_key, best_seg = key, (start, end)
    if best_seg is None:
        return (lo, hi)
    start, end = best_seg
    # Unbounded tail: clip to the cell so stamps stay inside the scene.
    return (max(start, lo), hi) if end == float("inf") else (start, end)


def _match_scene(design: GameDesign, obj: GameObject) -> int | None:
    """Index of the scene that owns *obj*, or None (falls back to seeded window).

    Scene prose names set dressing explicitly ("左侧陶盆龙舌兰，上方悬铁灯笼"):
    score every scene by shared character bigrams of the object's name, then —
    for props identified by their wearer — of name+visual+behavior; ≥2 shared
    bigrams are required to beat coincidence.
    """
    if not design.scenes:
        return None
    scene_grams = [
        _bigrams(f"{s.name}{s.description}{s.layout}{s.visual_theme}") for s in design.scenes
    ]

    def best_of(text: str) -> int | None:
        grams = _bigrams(text)
        if len(grams) < 2:
            return None
        scores = [len(grams & sg) for sg in scene_grams]
        idx = max(range(len(scores)), key=scores.__getitem__)
        return idx if scores[idx] >= 2 else None

    hit = best_of(obj.name)
    if hit is not None:
        return hit
    return best_of(f"{obj.name}{obj.visual}{obj.behavior}")


def _is_ui_only(obj: GameObject) -> bool:
    """True when the design puts the prop in the UI (icons/panels) — it never
    appears in a video frame, so there is nothing to extract."""
    text = f"{obj.role} {obj.spatial} {obj.visual}"
    return re.search(r"(?<![a-z])ui(?![a-z])|图标|面板|icon|panel", text, re.IGNORECASE) is not None


# ── Public API ───────────────────────────────────────────────────────────────


def _capture_distinct(
    video_path: Path,
    duration: float,
    out_dir: Path,
    final: Path,
    *,
    seed: int,
    used_hashes: set[str],
    region: tuple[float, float, float, float] | None,
    window: tuple[float, float] | None = None,
    attempts: int = 3,
) -> bool:
    """Extract a frame for one entity whose bytes are NOT already used.

    *window* pins sampling to [lo, hi] (objects: their scene's master shot);
    without it a seed-selected timeline window picks entity-specific moments
    (characters). The hash check retries with jittered stamps when a candidate
    duplicates an earlier asset. Returns False only when extraction itself
    fails everywhere.
    """
    tag = final.stem
    lo, hi = window if window is not None else (0.0, duration)
    span = max(hi - lo, 0.1)
    for attempt in range(attempts):
        if window is None:
            jitter = attempt * duration * 0.07
            stamps = _timestamps_for_entity(duration, count=3, seed=seed + attempt * 7919)
        else:
            jitter = attempt * span * 0.15
            # Midpoint first: within a shot it sits furthest from (possibly
            # undetected) shot boundaries, and it's the probe-validated moment;
            # hash dedup cannot judge content, so stamp order is the decision.
            stamps = [lo + span * f for f in (0.5, 0.25, 0.75)]
        for k, ts in enumerate(stamps):
            ts = min(max(ts + jitter, lo), max(hi - 0.05, lo))
            cand = out_dir / f"_{tag}_{attempt}_{k}.png"
            try:
                _extract_frame(video_path, ts, cand)
            except subprocess.CalledProcessError:
                continue
            if region is not None:
                cropped = out_dir / f"{tag}_{attempt}_{k}_c.png"
                try:
                    _crop_image(cand, cropped, region)
                    cand.unlink(missing_ok=True)
                    cand = cropped
                except subprocess.CalledProcessError:
                    cropped.unlink(missing_ok=True)
                    # keep uncropped
            digest = _file_hash(cand)
            is_new = digest not in used_hashes
            last_try = attempt == attempts - 1 and k == len(stamps) - 1
            if is_new or last_try:
                if not is_new:
                    log.warning(
                        "'%s' frame still duplicates an earlier asset — source video may be static there",
                        tag,
                    )
                cand.replace(final)
                used_hashes.add(digest)
                return True
            cand.unlink(missing_ok=True)
    return False


def extract_assets(video_path: Path, design: GameDesign, out_dir: Path) -> dict[str, Path]:
    """Extract visual assets from *video_path* based on *design* analysis.

    Returns a dict mapping asset names to file paths in *out_dir*:
        {"background": ..., "characters/Goblin": ..., "objects/Sword": ...}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}

    try:
        duration = _get_video_duration(video_path)
    except (subprocess.CalledProcessError, ValueError) as e:
        log.warning("Cannot get video duration: %s — skipping asset extraction", e)
        return assets

    # ── 1. Background: frame from the middle of the video ────────────────────
    used_hashes: set[str] = set()
    bg_path = out_dir / "background.png"
    try:
        _extract_frame(video_path, duration / 2, bg_path)
        assets["background"] = bg_path
        used_hashes.add(_file_hash(bg_path))
        log.info("Extracted background from t=%.1fs", duration / 2)
    except subprocess.CalledProcessError as e:
        log.warning("Failed to extract background: %s", e)

    # ── 2. Character sprites ─────────────────────────────────────────────────
    characters = [c for c in design.characters if "narrator" not in _role_tokens(c.role)]
    for char in characters:
        safe_name = (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in char.name).strip("_").lower()
        )
        seed = zlib.crc32(char.name.encode("utf-8"))
        spatial_hint = ""
        # Try to find spatial hints from related objects
        for obj in design.objects:
            if char.name.lower() in obj.name.lower() or obj.name.lower() in char.name.lower():
                spatial_hint = obj.spatial
                break
        region = _parse_spatial_hint(spatial_hint) if spatial_hint else None

        final = out_dir / f"char_{safe_name}.png"
        accepted = _capture_distinct(
            video_path,
            duration,
            out_dir,
            final,
            seed=seed,
            used_hashes=used_hashes,
            region=region,
        )
        if accepted:
            assets[f"characters/{safe_name}"] = final
            log.info("Extracted character '%s' sprite", char.name)
        else:
            log.warning("Character '%s': no frame could be extracted", char.name)

    # ── 3. Object sprites (collectibles, obstacles, key objects) ─────────────
    key_objects = [o for o in design.objects if _role_tokens(o.role) & _KEY_OBJECT_ROLES]
    cuts: list[float] | None = None  # detected once, only if some object anchors
    for obj in key_objects:
        safe_name = (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name).strip("_").lower()
        )
        if _is_ui_only(obj):
            log.info("Object '%s' lives in the UI, not in any video frame — skipping", obj.name)
            continue
        region = _parse_spatial_hint(obj.spatial) if obj.spatial else None
        window = None
        scene_idx = _match_scene(design, obj)
        if scene_idx is not None:
            if cuts is None:
                cuts = _shot_boundaries(video_path, settings.scene_threshold)
            window = _establishing_shot(
                _scene_window(duration, scene_idx, len(design.scenes)), cuts
            )
            log.info(
                "Object '%s': anchored to scene %d, establishing shot %.1f-%.1fs",
                obj.name,
                scene_idx + 1,
                window[0],
                window[1],
            )
        final = out_dir / f"obj_{safe_name}.png"
        accepted = _capture_distinct(
            video_path,
            duration,
            out_dir,
            final,
            seed=zlib.crc32(obj.name.encode("utf-8")),
            used_hashes=used_hashes,
            region=region,
            window=window,
            attempts=2,
        )
        if accepted:
            assets[f"objects/{safe_name}"] = final
            log.info("Extracted object '%s' sprite", obj.name)
        else:
            log.warning("Object '%s': no frame could be extracted", obj.name)

    # ── 4. Scene backgrounds (first scene gets the main background) ──────────
    for i, scene in enumerate(design.scenes[1:], start=1):  # skip first (already have background)
        safe_name = (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in scene.name).strip("_").lower()
        )
        ts_ratio = i / max(len(design.scenes), 1)
        ts = duration * ts_ratio
        bg_path = out_dir / f"scene_{safe_name}.png"
        try:
            _extract_frame(video_path, ts, bg_path)
            assets[f"scenes/{safe_name}"] = bg_path
            log.info("Extracted scene background '%s' from t=%.1fs", scene.name, ts)
        except subprocess.CalledProcessError as e:
            log.warning("Failed to extract scene '%s': %s", scene.name, e)

    log.info("Extracted %d assets total", len(assets))
    return assets
