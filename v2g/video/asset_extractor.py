"""Extract visual assets from video frames based on LLM analysis results.

Uses ffmpeg to:
- Extract scene background frames from representative timestamps
- Extract character/object sprite frames using spatial hints from the design
- Crop regions based on the LLM's spatial descriptions (left, right, center, etc.)
"""

from __future__ import annotations

import logging
import subprocess
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


def _timestamps_for_entity(duration: float, count: int = 3) -> list[float]:
    """Return *count* evenly-spaced timestamps avoiding the very start/end."""
    if duration <= 1.0:
        return [duration / 2]
    # Spread across 10%..90% of the video
    start = duration * 0.1
    end = duration * 0.9
    if count == 1:
        return [(start + end) / 2]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


# ── Public API ───────────────────────────────────────────────────────────────

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
    bg_path = out_dir / "background.png"
    try:
        _extract_frame(video_path, duration / 2, bg_path)
        assets["background"] = bg_path
        log.info("Extracted background from t=%.1fs", duration / 2)
    except subprocess.CalledProcessError as e:
        log.warning("Failed to extract background: %s", e)

    # ── 2. Character sprites ─────────────────────────────────────────────────
    characters = [c for c in design.characters if c.role != "narrator"]
    for char in characters:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in char.name).strip("_").lower()
        timestamps = _timestamps_for_entity(duration, count=3)
        spatial_hint = ""
        # Try to find spatial hints from related objects
        for obj in design.objects:
            if char.name.lower() in obj.name.lower() or obj.name.lower() in char.name.lower():
                spatial_hint = obj.spatial
                break

        best_path: Path | None = None
        for j, ts in enumerate(timestamps):
            raw_path = out_dir / f"_char_{safe_name}_{j}.png"
            try:
                _extract_frame(video_path, ts, raw_path)
            except subprocess.CalledProcessError:
                continue

            if spatial_hint:
                cropped_path = out_dir / f"char_{safe_name}_{j}.png"
                region = _parse_spatial_hint(spatial_hint)
                try:
                    _crop_image(raw_path, cropped_path, region)
                    raw_path.unlink(missing_ok=True)
                    raw_path = cropped_path
                except subprocess.CalledProcessError:
                    pass  # keep uncropped

            if best_path is None:
                best_path = raw_path
            else:
                # Keep all variants; first is the default
                pass

        if best_path:
            final = out_dir / f"char_{safe_name}.png"
            if best_path != final:
                best_path.rename(final)
                # Clean up other variants
                for f in out_dir.glob(f"_char_{safe_name}_*.png"):
                    f.unlink(missing_ok=True)
                for f in out_dir.glob(f"char_{safe_name}_*.png"):
                    f.unlink(missing_ok=True)
            assets[f"characters/{safe_name}"] = final
            log.info("Extracted character '%s' sprite", char.name)

    # ── 3. Object sprites (collectibles, obstacles, key objects) ─────────────
    key_objects = [
        o for o in design.objects
        if o.role in ("collectible", "obstacle", "vehicle", "weapon", "decoration")
    ]
    for obj in key_objects:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name).strip("_").lower()
        ts = _timestamps_for_entity(duration, count=1)[0]
        raw_path = out_dir / f"_obj_{safe_name}.png"
        try:
            _extract_frame(video_path, ts, raw_path)
        except subprocess.CalledProcessError:
            continue

        final = out_dir / f"obj_{safe_name}.png"
        if obj.spatial:
            region = _parse_spatial_hint(obj.spatial)
            try:
                _crop_image(raw_path, final, region)
                raw_path.unlink(missing_ok=True)
            except subprocess.CalledProcessError:
                raw_path.rename(final)
        else:
            raw_path.rename(final)

        assets[f"objects/{safe_name}"] = final
        log.info("Extracted object '%s' sprite", obj.name)

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
