"""Extract visual assets from video frames based on LLM analysis results.

ffmpeg cuts candidate frames; each accepted one is then shaped for its kind:
- **characters and objects → 512×512 square sprites**, cropped around the
  subject (model-located box → the design's `spatial` hint → frame center), so
  a portrait slot shows the subject instead of a letterboxed landscape strip
- **backgrounds and scene stills → the untouched frame**, keeping the source
  video's own aspect ratio

Every candidate is gated before it ships: blank/black transition frames are
demoted to the back of the queue, and with verification on
(`V2G_ASSET_VERIFY`, default) all candidates go to the vision model as **one
contact sheet**, which ranks them; a sprite's pick is then located at full
size — that answer is the crop box. A refusal moves to the next ranked pick.
An unreachable model, an all-rejected entity or a duplicate hash falls back
rather than fails: extraction never raises past its own logging.

Candidate pools are shot midpoints (ffmpeg cut detection), never bare timeline
fractions: an object's or scene's timeline cell contributes its establishing
shot first — the shot starting at the boundary (video start / cut) nearest the
cell — and the rest of the video is then sampled evenly, so a mis-anchored
scene still finds its frame. Set dressing lives in that opening wide, not in
the dialogue close-ups that follow it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import zlib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from v2g.config import settings
from v2g.llm import jsonfix
from v2g.llm.analyzer import GameDesign, GameObject, SceneDesign
from v2g.llm.client import chat

log = logging.getLogger(__name__)

SPRITE_SIZE = 512  # edge of a character/object sprite; scenes keep the video's pixels
_BLANK_LUMA = 10.0  # mean brightness below this = transition/black card, never shipped first


@dataclass(frozen=True)
class _Target:
    """One asset to capture: what must appear, and how the frame is framed.

    ``square=True`` (characters/objects) crops a ``SPRITE_SIZE`` square around
    the subject; ``square=False`` (backgrounds/scene stills) keeps the whole
    frame so the asset follows the source video's aspect ratio. ``region`` is
    the design's spatial hint — the framing used when the model gives no box.
    """

    kind: str  # "characters" | "objects" | "scenes" | "background"
    subject: str  # design text naming what has to be visible
    square: bool = True
    region: tuple[float, float, float, float] | None = None


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
    """Extract a single frame at *timestamp* seconds.

    A seek that lands past the last frame makes ffmpeg exit 0 **without
    writing anything** — that is a failed candidate, so it raises instead of
    handing the caller a path to a file that does not exist.
    """
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
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise subprocess.CalledProcessError(
            1, "ffmpeg", stderr=f"no frame at t={timestamp:.2f}".encode()
        )
    return out_path


def _mean_luma(path: Path) -> float:
    """Mean brightness (0..255) of a 64×64 grayscale copy — the blank-frame probe.

    Transition frames and black cards score under ``_BLANK_LUMA``; any frame
    with real content (even a dim scene, a subtitle on black) scores above it.
    """
    from PIL import Image

    with Image.open(path) as im:
        hist = im.convert("L").resize((64, 64)).histogram()
    total = sum(hist)
    return sum(i * n for i, n in enumerate(hist)) / total if total else 0.0


def _content_rect(im) -> tuple[int, int, int, int] | None:
    """Picture area to crop sprites into, or None when the frame is full-bleed.

    Rows/columns are classified by how much of them is near-black, and the
    **largest contiguous run of picture** wins — not "everything outside the
    leading bar". That matters because a source can carry a black band *inside*
    the picture (inset cards, fades), and it tolerates a watermark sitting in
    the letterbox bar: a run-based scan neither stops early at a watermark nor
    crops a band in. Runs shorter than a third of the frame mean the frame is
    dark or full-bleed, so no bounds are imposed.
    """
    gray = im.convert("L")
    width, height = gray.size

    def run(vertical: bool, length: int) -> tuple[int, int]:
        best_start = best_len = start = size = 0
        for i in range(length):
            strip = (
                gray.crop((0, i, width, i + 1)) if vertical else gray.crop((i, 0, i + 1, height))
            )
            hist = strip.histogram()
            total = sum(hist) or 1
            picture = sum(hist[13:]) * 10 >= total  # under 10% near-black → picture
            if picture:
                if size == 0:
                    start = i
                size += 1
                if size > best_len:
                    best_start, best_len = start, size
            else:
                size = 0
        return (best_start, best_start + best_len)

    top, bottom = run(True, height)
    left, right = run(False, width)
    if bottom - top < height // 3 or right - left < width // 3:
        return None
    if top == 0 and left == 0 and bottom == height and right == width:
        return None
    return (left, top, right, bottom)


def _region_crop(
    region: tuple[float, float, float, float],
    width: int,
    height: int,
    bounds: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pixel box of the largest square inside *bounds* covering *region*.

    ``region`` is fractional (x, y, w, h); the square grows to the region's
    larger side, then is clamped into *bounds* (defaults to the whole frame) —
    the fallback framing when no model box exists (spatial hint, or the whole
    frame when there is none).
    """
    left_b, top_b, right_b, bottom_b = bounds or (0, 0, width, height)
    x, y, w, h = region
    rx = max(x * width, left_b)
    ry = max(y * height, top_b)
    rw = min(x * width + w * width, right_b) - rx
    rh = min(y * height + h * height, bottom_b) - ry
    side = round(min(max(rw, rh), right_b - left_b, bottom_b - top_b))
    left = round(min(max(rx + rw / 2 - side / 2, left_b), right_b - side))
    top = round(min(max(ry + rh / 2 - side / 2, top_b), bottom_b - side))
    return (left, top, left + side, top + side)


def _box_crop(
    box: list[float],
    width: int,
    height: int,
    kind: str,
    bounds: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pixel square framing the model-located subject, per asset kind.

    Characters are cut **half-body**: the square is at least half the subject's
    box tall (a full-figure box → head to waist) and hangs from the box top, so
    the head is never clipped. Objects are cut whole and centred, with a
    quarter-size margin. The subject box is clamped into *bounds* first, so a
    letterbox bar inside it never reaches the sprite.
    """
    left_b, top_b, right_b, bottom_b = bounds or (0, 0, width, height)
    x, y, w, h = box
    bx = max(x * width, left_b)
    by = max(y * height, top_b)
    bw = max(min(x * width + w * width, right_b) - bx, 1.0)
    bh = max(min(y * height + h * height, bottom_b) - by, 1.0)
    if kind == "characters":
        side = max(bw * 1.2, bh * 0.55)
        cy = by + side / 2
    else:
        side = max(bw, bh) * 1.25
        cy = by + bh / 2
    side = round(min(side, right_b - left_b, bottom_b - top_b))
    left = round(min(max(bx + bw / 2 - side / 2, left_b), right_b - side))
    top = round(min(max(cy - side / 2, top_b), bottom_b - side))
    return (left, top, left + side, top + side)


def _materialize(cand: Path, dst: Path, target: _Target, box: list[float] | None) -> Path:
    """Write the accepted frame to *dst* as the asset, consuming *cand*.

    Sprites become a ``SPRITE_SIZE`` square (model box → spatial hint → frame
    centre), kept inside the picture area so letterbox bars never reach them;
    backgrounds and scene stills move over untouched, keeping the source video's
    aspect ratio — bars included, they are part of the video.
    """
    from PIL import Image

    if not target.square:
        cand.replace(dst)
        return dst
    with Image.open(cand) as im:
        bounds = _content_rect(im)
        crop_box = (
            _box_crop(box, *im.size, target.kind, bounds)
            if box is not None
            else _region_crop(target.region or (0.0, 0.0, 1.0, 1.0), *im.size, bounds)
        )
        sprite = im.crop(crop_box).resize((SPRITE_SIZE, SPRITE_SIZE), Image.LANCZOS)
        tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
        try:
            sprite.save(tmp)
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)
    cand.unlink(missing_ok=True)
    return dst


def _scene_text(scene: SceneDesign) -> str:
    """Compact subject line for a scene — what a background frame must match."""
    return " | ".join(p for p in (f"{scene.name}: {scene.description}", scene.visual_theme) if p)


def _contact_sheet(frames: list[Path], dst: Path, cols: int = 3) -> tuple[int, int]:
    """Tile candidate frames into one grid image; returns ``(cols, rows)``.

    One vision call then ranks every candidate at once instead of asking the
    model about frames one by one — cheaper *and* able to compare shots from
    the whole timeline, which is what a scene whose cell missed its content
    needs. JPEG keeps the upload small.
    """
    from PIL import Image, ImageDraw

    with Image.open(frames[0]) as first:
        fw, fh = first.size
    tile_w = 448
    tile_h = max(1, round(tile_w * fh / fw))
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(frames):
        x, y = (i % cols) * tile_w, (i // cols) * tile_h
        with Image.open(path) as im:
            sheet.paste(im.convert("RGB").resize((tile_w, tile_h), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(255, 215, 0), width=3)
    sheet.save(dst, format="JPEG", quality=88)
    return cols, rows


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes — used to prove two assets are not identical."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shot_pool(
    duration: float,
    cuts: list[float],
    cell: tuple[float, float] | None,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Shots to sample for one asset — the contact sheet's cells, best first.

    With a *cell* (objects and scenes): the establishing shot and the cell's
    other shots come first, then the rest of the video spread evenly — an
    anchor that missed its content still gets whole-video coverage instead of
    a dozen stamps inside the same wrong region. Without a cell (characters):
    even coverage of the whole timeline, phase-shifted by *seed* so two
    entities don't sample the same shots.

    Capped at ``_SHEET_CELLS``: past that the frames would not fit the sheet.
    """
    segments = _shot_segments(duration, cuts)
    if not segments:
        return [(0.0, duration)]
    if cell is None:
        step = max(1, len(segments) // _SHEET_CELLS)
        phase = int(((seed % 997) / 997) * step)
        return segments[phase::step][:_SHEET_CELLS]
    ordered = _candidate_windows(cell, cuts, duration)
    head = ordered[:5]
    tail = ordered[5:]
    step = max(1, len(tail) // max(1, _SHEET_CELLS - len(head))) if tail else 1
    return (head + tail[::step])[:_SHEET_CELLS]


def _stamps(pool: list[tuple[float, float]]) -> list[float]:
    """Timestamps for *pool*, one contact sheet's worth, best guess first.

    Each shot contributes as many interior samples as it takes to fill the
    sheet, round-robin across shots: the sheet shows *every* pooled shot before
    any shot repeats, and a video with one unbroken take still yields a dozen
    distinct moments (six entities must never collapse onto one frame).
    """
    if not pool:
        return []
    slices = max(1, -(-_SHEET_CELLS // len(pool)))
    stamps = [
        min(max(a + (b - a) * (i + 0.5) / slices, a), max(b - 0.05, a))
        for i in range(slices)
        for a, b in pool
    ]
    return stamps[:_SHEET_CELLS]


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


def _shot_segments(duration: float, cuts: list[float]) -> list[tuple[float, float]]:
    """Every shot in the timeline as (start, end), split at the detected cuts."""
    bounds = sorted({0.0, *(c for c in cuts if 0.0 < c < duration), duration})
    return list(pairwise(bounds))


def _candidate_windows(
    cell: tuple[float, float], cuts: list[float], duration: float, limit: int | None = None
) -> list[tuple[float, float]]:
    """Shots to sample for one scene/object, best guess first:
    the establishing shot, then the cell's other shots (widest overlap first),
    then the nearest shots outside the cell.

    Scene cells are equal-duration while the LLM's scene list is neither evenly
    spaced nor reliably in timeline order, so a cell can miss its content
    completely — the model check gates every candidate, which lets a stubborn
    asset search outward instead of shipping a mis-timed frame.
    """
    lo, hi = cell
    segments = _shot_segments(duration, cuts)
    est = _establishing_shot(cell, cuts)

    def overlap(seg: tuple[float, float]) -> float:
        return max(0.0, min(hi, seg[1]) - max(lo, seg[0]))

    def distance(seg: tuple[float, float]) -> float:
        return 0.0 if overlap(seg) else min(abs(seg[0] - hi), abs(seg[1] - lo))

    inside = sorted((s for s in segments if overlap(s) > 0), key=lambda s: -overlap(s))
    outside = sorted((s for s in segments if overlap(s) <= 0), key=distance)
    ordered = [est]
    ordered += [s for s in (*inside, *outside) if abs(s[0] - est[0]) > 0.05]
    return ordered if limit is None else ordered[:limit]


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


# ── Model check (content gate) ───────────────────────────────────────────────

_VERIFY_SYSTEM = (
    "You select and verify frames cut from a video for game assets. "
    "Answer only with the requested JSON."
)

_SHEET_COLS = 3  # contact-sheet grid: _SHEET_COLS × ceil(count / _SHEET_COLS) cells
_SHEET_CELLS = 12  # most candidates shown to the model at once (3×4 grid)
_VERIFY_TOKENS = 6000  # reasoning models spent 2000 tokens thinking and answered nothing


def _verify_prompt(target: _Target) -> str:
    """What the model must confirm about one full-size candidate frame."""
    if target.kind in ("scenes", "background"):
        return (
            "This frame was cut from a video to serve as the game background for the scene "
            f'"{target.subject}".\n'
            "Is it a usable background for that scene — the described place, with real set and "
            "props visible? A black screen, a transition frame, a different location, or a "
            "card full of text/credits → found=false.\n"
            'JSON only: {"found": true, "reason": "one short sentence"}'
        )
    noun = "character" if target.kind == "characters" else "object"
    return (
        f'This frame was cut from a video to show the {noun} "{target.subject}".\n'
        f"Is that {noun} actually visible in it?\n"
        "Also return a tight bounding box of the subject as [x, y, w, h] in fractions of the "
        "image (x,y = top-left corner, w,h = size).\n"
        f"A different {noun}, an empty shot, or only the surroundings → found=false.\n"
        'JSON only: {"found": true, "box": [0.3, 0.1, 0.4, 0.8], "reason": "one short sentence"}'
    )


def _pick_prompt(target: _Target, count: int, cols: int) -> str:
    """What the model must answer about a contact sheet of candidate frames.

    Wording matters: asking "is it visible?" on thumbnails got every sprite
    pick answered with an empty list (a person in a wide shot is a few pixels
    in a cell), so the model ranks by plausibility instead of certifying.
    """
    rows = (count + cols - 1) // cols
    grid = (
        f"The image is one {rows}×{cols} grid of frames cut from the same video: cell 1 is "
        f"top-left, then left to right, top to bottom ({count} cells).\n"
    )
    if target.kind in ("scenes", "background"):
        task = (
            f"Rank the cell(s) that could serve as the game background for the scene "
            f'"{target.subject}", most suitable first — the described place, with real set and '
            "props visible.\n"
            "Judge plausibly: a wide or dim shot still counts if it could be that place; a card "
            "full of text or credits does not.\n"
            "Answer an empty list only when no cell could be it."
        )
    else:
        noun = "character" if target.kind == "characters" else "object"
        task = (
            f'Rank the cell(s) where the {noun} "{target.subject}" is visible or most likely to '
            "be, best first.\n"
            "Do not withhold a cell because the shot is wide or the subject small — rank by "
            "likelihood.\n"
            "Answer an empty list only when no cell could show it."
        )
    return grid + task + '\nJSON only: {"cells": [3, 7], "reason": "one short sentence"}'


def _parse_cells(value: object, count: int) -> list[int]:
    """0-based cell indices (order kept, in range) from a model answer."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    cells: list[int] = []
    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= count and (index - 1) not in cells:
            cells.append(index - 1)
    return cells


def _parse_box(value: object) -> list[float] | None:
    """Normalized ``[x, y, w, h]`` from a model answer, or None when unusable."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0 - x), min(max(h, 0.0), 1.0 - y)
    return [x, y, w, h] if w >= 0.02 and h >= 0.02 else None


class _Verifier:
    """Vision-model gate: which candidate shows the asset, and where it sits.

    :meth:`pick` ranks a contact sheet of every candidate in **one** call — the
    model compares shots side by side instead of judging them one at a time,
    which is what a scene whose timeline cell missed its content needs.
    :meth:`locate` then confirms the winning full-size frame for a sprite and
    returns the subject's box.

    ``enabled=False`` (``V2G_ASSET_VERIFY=0``) never calls the model, and a
    subject-less target (a design with no scenes) has nothing to check. Any
    failure — no key, unreachable endpoint, unusable answer — disables both for
    the rest of the run instead of retrying per asset, so a model outage costs
    one warning and leaves extraction exactly as it was.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._down = False

    def _ask(self, prompt: str, image: Path) -> dict | None:
        """One verified chat round; None = no verdict (off, failed, unusable)."""
        if not self.enabled or self._down:
            return None
        try:
            res = chat(
                _VERIFY_SYSTEM,
                [prompt, image],
                temperature=0.0,
                # Reasoning endpoints spend the budget thinking before they
                # answer: at 200 the body came back empty (finish=length), which
                # silently turned the gate off for that frame, and 1000 was
                # eaten whole on the longer prompts too.
                max_tokens=_VERIFY_TOKENS,
            )
            data = jsonfix.salvage(res.text)
        except Exception as e:  # noqa: BLE001 — an outage must never fail extraction
            self._down = True
            log.warning("Asset verification unavailable (%s) — extracting frames unchecked", e)
            return None
        if not isinstance(data, dict):
            log.warning("Asset verification answered unusably (%.120s) — no verdict", res.text)
            return None
        return data

    def pick(self, sheet: Path, target: _Target, count: int) -> list[int] | None:
        """Ranked 0-based cell indices of *sheet*, best first; None = no verdict.

        An empty list is a verdict ("no candidate qualifies") and reaches the
        caller as such — only a failed round returns None.
        """
        if not target.subject.strip():
            return None
        cols = min(_SHEET_COLS, count)
        data = self._ask(_pick_prompt(target, count, cols), sheet)
        if data is None:
            return None
        if "cells" not in data:
            log.warning("Asset verification listed no cells (%.120s) — no verdict", data)
            return None
        return _parse_cells(data["cells"], count)

    def locate(self, image: Path, target: _Target) -> tuple[bool, list[float] | None] | None:
        """``(found, subject box)`` for one full-size frame; None = no verdict."""
        if not target.subject.strip():
            return None
        data = self._ask(_verify_prompt(target), image)
        if data is None:
            return None
        if not isinstance(data.get("found"), bool):
            log.warning("Asset verification gave no verdict (%.120s)", data)
            return None
        return data["found"], _parse_box(data.get("box")) if target.square else None


# ── Public API ───────────────────────────────────────────────────────────────


def _capture_distinct(
    video_path: Path,
    out_dir: Path,
    final: Path,
    *,
    used_hashes: set[str],
    target: _Target,
    pool: list[tuple[float, float]],
    verifier: _Verifier | None = None,
) -> bool:
    """Extract one asset for *target* whose bytes are NOT already used.

    *pool* is the shot list to sample (see ``_shot_pool``: anchored shots
    first, then whole-video coverage). All of its midpoints are cut at once,
    blank/black frames demoted to the back of the queue, and — with
    verification on — shown to the model as **one contact sheet**, so it
    compares every candidate in a single call and answers with its picks
    ranked. Each pick is shaped for its kind (a ``SPRITE_SIZE`` square sprite
    for characters/objects, the untouched frame for backgrounds/scene stills),
    and a sprite's pick is re-checked on the full-size frame, which also
    yields the subject's box. Picks that duplicate an earlier asset move on to
    the next ranked one; when nothing survives — no match, all taken — the
    first candidate ships anyway, so an entity never silently disappears.
    Returns False only when frames could not be cut at all.
    """
    verifier = verifier or _Verifier(False)
    tag = final.stem
    stamps = _stamps(pool)

    fresh: list[Path] = []
    blank: list[Path] = []
    stamp_of: dict[Path, float] = {}
    for i, ts in enumerate(stamps):
        cand = out_dir / f"_{tag}_{i}.png"
        try:
            _extract_frame(video_path, ts, cand)
        except subprocess.CalledProcessError:
            continue
        stamp_of[cand] = ts
        if _mean_luma(cand) < _BLANK_LUMA:
            log.debug("Asset '%s': blank frame at t=%.1fs demoted", tag, ts)
            blank.append(cand)
        else:
            fresh.append(cand)
    candidates = (fresh + blank)[:_SHEET_CELLS]  # blanks may show for a dark scene, never lead
    if not candidates:
        return False

    # Every target is chosen from a contact sheet (one call comparing every
    # shot); sprites are then re-checked full size, because a thumbnail cell
    # can rank the right shot but cannot resolve the subject's box.
    sheet_ranked: list[int] | None = None
    if verifier.enabled and target.subject.strip():
        cols = min(_SHEET_COLS, len(candidates))
        sheet = out_dir / f"_{tag}_sheet.jpg"
        _contact_sheet(candidates, sheet, cols)
        sheet_ranked = verifier.pick(sheet, target, len(candidates))
        sheet.unlink(missing_ok=True)
    if sheet_ranked is None:
        ranked = list(range(len(candidates)))  # no verdict (off / failed) → stamp order
    elif sheet_ranked:
        # Validated picks first, then the rest: a pick that turns out to
        # duplicate an earlier asset falls through to a distinct cell instead
        # of forcing two scenes onto one byte-identical still.
        ranked = list(dict.fromkeys([*sheet_ranked, *range(len(candidates))]))
    else:
        ranked = []  # the model vouched for nothing
    if not ranked:
        log.info("Asset '%s': the model found no matching frame", tag)
    confirm = target.square or sheet_ranked is None  # judge each frame full size?

    for rank, idx in enumerate(ranked):
        frame = candidates[idx]
        box: list[float] | None = None
        if confirm:
            verdict = verifier.locate(frame, target)
            if verdict is not None:
                if not verdict[0]:
                    log.info("Asset '%s': model rejected candidate %d", tag, idx + 1)
                    continue
                box = verdict[1]
        sprite = out_dir / f"_{tag}_{idx}_sprite.png"
        try:
            _materialize(frame, sprite, target, box)
        except OSError as e:
            log.warning("Asset '%s': cannot shape a frame (%s)", tag, e)
            frame.unlink(missing_ok=True)
            continue
        digest = _file_hash(sprite)
        is_new = digest not in used_hashes
        if is_new or rank == len(ranked) - 1:
            if not is_new:
                log.warning(
                    "'%s' frame still duplicates an earlier asset — source video may be static there",
                    tag,
                )
            sprite.replace(final)
            used_hashes.add(digest)
            log.info("Asset '%s': kept frame at t=%.1fs", tag, stamp_of.get(frame, -1.0))
            _discard(*fresh, *blank)
            return True
        sprite.unlink(missing_ok=True)

    keep = next((path for path in candidates if path.exists()), None)
    if keep is None:
        return False
    log.warning(
        "'%s': no candidate passed the checks — keeping the least-bad frame (t=%.1fs)",
        tag,
        stamp_of.get(keep, -1.0),
    )
    sprite = out_dir / f"_{tag}_fallback.png"
    try:
        _materialize(keep, sprite, target, None)
    except OSError as e:
        log.warning("Asset '%s': cannot shape the fallback frame (%s)", tag, e)
        _discard(*fresh, *blank)
        return False
    sprite.replace(final)
    used_hashes.add(_file_hash(final))
    _discard(*fresh, *blank)
    return True


def _discard(*paths: Path | None) -> None:
    """Drop leftover candidate frames (a gated one that never shipped)."""
    for path in paths:
        if path is not None:
            path.unlink(missing_ok=True)


def extract_assets(
    video_path: Path,
    design: GameDesign,
    out_dir: Path,
    *,
    verify: bool | None = None,
) -> dict[str, Path]:
    """Extract visual assets from *video_path* based on *design* analysis.

    Every asset is shaped for its kind: characters and objects come back as
    ``SPRITE_SIZE``×``SPRITE_SIZE`` square sprites cropped around the subject,
    backgrounds and scene stills as untouched frames following the source
    video's aspect ratio. *verify* overrides ``V2G_ASSET_VERIFY``: with
    verification on, the vision model ranks every candidate on one contact
    sheet, and a sprite's pick is then located at full size — that answer
    carries the box the crop is taken from.

    Returns a dict mapping asset names to file paths in *out_dir*:
        {"background": ..., "characters/Goblin": ..., "objects/Sword": ...}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}
    verifier = _Verifier(settings.asset_verify if verify is None else verify)

    try:
        duration = _get_video_duration(video_path)
    except (subprocess.CalledProcessError, ValueError) as e:
        log.warning("Cannot get video duration: %s — skipping asset extraction", e)
        return assets

    used_hashes: set[str] = set()
    cuts: list[float] | None = None  # detected once: every asset samples shots

    def pool_for(cell: tuple[float, float] | None, seed: int = 0) -> list[tuple[float, float]]:
        """Shots for one asset: *cell*'s establishing shot first, then — so an
        anchor that missed its content still has somewhere to look — the rest
        of the video spread evenly. ``cell=None`` covers the whole timeline."""
        nonlocal cuts
        if cuts is None:
            cuts = _shot_boundaries(video_path, settings.scene_threshold)
        return _shot_pool(duration, cuts, cell, seed)

    # ── 1. Background: frame from the middle of the video ────────────────────
    bg_target = _Target(
        "background",
        _scene_text(design.scenes[0]) if design.scenes else "",
        square=False,
    )
    bg_path = out_dir / "background.png"
    if _capture_distinct(
        video_path,
        out_dir,
        bg_path,
        used_hashes=used_hashes,
        target=bg_target,
        pool=pool_for((duration * 0.4, duration * 0.6)),
        verifier=verifier,
    ):
        assets["background"] = bg_path
        log.info("Extracted background (kept the video's aspect ratio)")
    else:
        log.warning("Failed to extract background")

    # ── 2. Character sprites (512×512 half-body portraits) ───────────────────
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
        target = _Target(
            "characters",
            f"{char.name} ({char.role}): {char.visual}",
            region=region,
        )
        final = out_dir / f"char_{safe_name}.png"
        accepted = _capture_distinct(
            video_path,
            out_dir,
            final,
            used_hashes=used_hashes,
            target=target,
            pool=pool_for(None, seed),
            verifier=verifier,
        )
        if accepted:
            assets[f"characters/{safe_name}"] = final
            log.info("Extracted character '%s' sprite", char.name)
        else:
            log.warning("Character '%s': no frame could be extracted", char.name)

    # ── 3. Object sprites (512×512 squares) ──────────────────────────────────
    key_objects = [o for o in design.objects if _role_tokens(o.role) & _KEY_OBJECT_ROLES]
    for obj in key_objects:
        safe_name = (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name).strip("_").lower()
        )
        if _is_ui_only(obj):
            log.info("Object '%s' lives in the UI, not in any video frame — skipping", obj.name)
            continue
        region = _parse_spatial_hint(obj.spatial) if obj.spatial else None
        seed = zlib.crc32(obj.name.encode("utf-8"))
        cell: tuple[float, float] | None = None
        scene_idx = _match_scene(design, obj)
        if scene_idx is not None:
            cell = _scene_window(duration, scene_idx, len(design.scenes))
        pool = pool_for(cell, seed)
        if cell is not None:
            log.info(
                "Object '%s': anchored to scene %d (cell %.1f-%.1fs), %d shots to search",
                obj.name,
                scene_idx + 1,
                cell[0],
                cell[1],
                len(pool),
            )
        target = _Target(
            "objects",
            f"{obj.name} ({obj.role}): {obj.visual}",
            region=region,
        )
        final = out_dir / f"obj_{safe_name}.png"
        accepted = _capture_distinct(
            video_path,
            out_dir,
            final,
            used_hashes=used_hashes,
            target=target,
            pool=pool,
            verifier=verifier,
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
        bg_path = out_dir / f"scene_{safe_name}.png"
        accepted = _capture_distinct(
            video_path,
            out_dir,
            bg_path,
            used_hashes=used_hashes,
            target=_Target("scenes", _scene_text(scene), square=False),
            pool=pool_for(
                _scene_window(duration, i, len(design.scenes)),
                zlib.crc32(scene.name.encode("utf-8")),
            ),
            verifier=verifier,
        )
        if accepted:
            assets[f"scenes/{safe_name}"] = bg_path
            log.info("Extracted scene background '%s'", scene.name)
        else:
            log.warning("Failed to extract scene '%s'", scene.name)

    log.info("Extracted %d assets total", len(assets))
    return assets
