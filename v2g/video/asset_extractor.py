"""Extract visual assets from video frames based on LLM analysis results.

Uses ffmpeg to:
- Extract scene background frames from representative timestamps
- Extract character/object sprite frames using spatial hints from the design
- Crop regions based on the LLM's spatial descriptions (left, right, center, etc.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import zlib
from pathlib import Path

from v2g.llm.analyzer import GameDesign

log = logging.getLogger(__name__)

# ── Crop region helpers ──────────────────────────────────────────────────────

def _parse_spatial_hint(hint: str) -> tuple[float, float, float, float]:
    """Convert a natural-language spatial hint into a (x_frac, y_frac, w_frac, h_frac) crop.

    Examples:
        "left side of frame"       → (0.0, 0.0, 0.4, 1.0)
        "right third"              → (0.66, 0.0, 0.34, 1.0)
        "center of frame"          → (0.2, 0.1, 0.6, 0.8)
        "bottom half"              → (0.0, 0.5, 1.0, 0.5)
        "top-left corner"          → (0.0, 0.0, 0.4, 0.4)
    """
    hint_lower = hint.lower()
    has_left = "left" in hint_lower
    has_right = "right" in hint_lower
    has_top = "top" in hint_lower
    has_bottom = "bottom" in hint_lower
    has_center = "center" in hint_lower or "middle" in hint_lower

    # Start with full frame
    x, y, w, h = 0.0, 0.0, 1.0, 1.0

    # "center" alone (no directional modifier) → centered box
    if has_center and not has_left and not has_right and not has_top and not has_bottom:
        return (0.2, 0.1, 0.6, 0.8)

    # Horizontal
    if has_left:
        x, w = 0.0, 0.4
    elif has_right:
        x, w = 0.6, 0.4
    elif has_center:
        x, w = 0.25, 0.5

    # Vertical
    if has_top:
        y, h = 0.0, 0.45
    elif has_bottom:
        y, h = 0.55, 0.45
    elif has_center and w == 1.0:
        y, h = 0.15, 0.7

    return x, y, w, h


def _get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def probe_video_size(video_path: Path) -> tuple[int, int] | None:
    """First video stream's display (width, height); None when probing fails.

    Honors a rotation side-data entry so phone-shot portrait footage stored
    with landscape pixels still reports its displayed orientation.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,side_data_list",
             "-of", "json", str(video_path)],
            capture_output=True, text=True, check=True,
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
        OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError,
    ):
        return None


# ── Core extraction ──────────────────────────────────────────────────────────

def _extract_frame(video_path: Path, timestamp: float, out_path: Path) -> Path:
    """Extract a single frame at *timestamp* seconds."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{timestamp:.2f}", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", str(out_path)],
        capture_output=True, check=True,
    )
    return out_path


def _crop_image(src: Path, dst: Path, region: tuple[float, float, float, float]) -> Path:
    """Crop an image using fractional coordinates (x, y, w, h) each in [0, 1]."""
    x, y, w, h = region
    # ffmpeg crop filter uses pixels, but we express as percentages
    crop_filter = f"crop=iw*{w:.2f}:ih*{h:.2f}:iw*{x:.2f}:ih*{y:.2f}"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", crop_filter, "-q:v", "2", str(dst)],
        capture_output=True, check=True,
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
    attempts: int = 3,
) -> bool:
    """Extract a frame for one entity whose bytes are NOT already used.

    Seeded windows pick entity-specific moments; the hash check retries with
    jittered seeds when the chosen frame duplicates an earlier asset. Returns
    False only when extraction itself fails everywhere.
    """
    tag = final.stem
    for attempt in range(attempts):
        jitter = attempt * duration * 0.07
        stamps = _timestamps_for_entity(duration, count=3, seed=seed + attempt * 7919)
        for k, ts in enumerate(stamps):
            ts = min(max(ts + jitter, 0.0), max(duration - 0.05, 0.0))
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
    characters = [c for c in design.characters if c.role != "narrator"]
    for char in characters:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in char.name).strip("_").lower()
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
            video_path, duration, out_dir, final,
            seed=seed, used_hashes=used_hashes, region=region,
        )
        if accepted:
            assets[f"characters/{safe_name}"] = final
            log.info("Extracted character '%s' sprite", char.name)
        else:
            log.warning("Character '%s': no frame could be extracted", char.name)

    # ── 3. Object sprites (collectibles, obstacles, key objects) ─────────────
    key_objects = [
        o for o in design.objects
        if o.role in ("collectible", "obstacle", "vehicle", "weapon", "decoration")
    ]
    for obj in key_objects:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name).strip("_").lower()
        region = _parse_spatial_hint(obj.spatial) if obj.spatial else None
        final = out_dir / f"obj_{safe_name}.png"
        accepted = _capture_distinct(
            video_path, duration, out_dir, final,
            seed=zlib.crc32(obj.name.encode("utf-8")),
            used_hashes=used_hashes,
            region=region,
            attempts=2,
        )
        if accepted:
            assets[f"objects/{safe_name}"] = final
            log.info("Extracted object '%s' sprite", obj.name)
        else:
            log.warning("Object '%s': no frame could be extracted", obj.name)

    # ── 4. Scene backgrounds (first scene gets the main background) ──────────
    for i, scene in enumerate(design.scenes[1:], start=1):  # skip first (already have background)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in scene.name).strip("_").lower()
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
