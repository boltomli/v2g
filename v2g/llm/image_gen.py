"""Image generation/transformation providers.

The default is :class:`NullProvider` — a no-op, so extracted assets come
straight from video frames with no model involved and no cost.

Setting ``V2G_IMAGEGEN_PROVIDER=qwen`` switches to :class:`QwenImage21Provider`,
which restyles images with a **local Qwen-Image-2.1** run through diffusers
(install the optional extra first: ``uv sync --extra imagegen``).

Managed model bundle — downloaded once into
``<output_root>/.v2g_cache/imagegen/qwen-image-2.1/`` (~23 GB):

- text encoder + VAE + configs: official ``Qwen/Qwen-Image-2.1`` weights via
  the ModelScope mirror (bf16; that mirror is several times faster than HF
  from here);
- denoiser: the unsloth GGUF Q4_K_M quant (~4.2 GB) — the full bf16 stack is
  33 GB on disk and cannot fit this machine's RAM next to the OS, while the
  GGUF keeps the resident set around 22 GB.

Device policy at load time, from available VRAM (the bf16 text encoder alone
is ~17.5 GB): >= 28 GB → weights fully on GPU; >= 20 GB → model CPU offload;
below that → sequential CPU offload (one module resident at a time — what a
6 GB card needs).

Prepare the bundle ahead of a run::

    .venv/Scripts/python -c "from v2g.llm.image_gen import prepare_model; prepare_model()"

Failures raise; :func:`v2g.godot.generator._restyle_assets` keeps the original
frame and logs, so image generation can never fail a run.
"""

from __future__ import annotations

import abc
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

from v2g.config import settings

log = logging.getLogger(__name__)


class ImageGenProvider(abc.ABC):
    """Abstract interface for image generation/transformation."""

    @abc.abstractmethod
    def transform(
        self,
        image_path: Path,
        prompt: str,
        *,
        reference: str = "",
        size: str = "",
    ) -> Path:
        """Restyle *image_path* according to *prompt*; return the result path.

        Args:
            image_path: Source image (PNG) — may be overwritten in place.
            prompt: Style/content instruction (e.g. the ``--instruct`` text).
            reference: Subject context (e.g. the character's visual
                description from the design) to keep identity stable.
            size: Explicit ``WxH`` output size; empty keeps the input aspect
                ratio within the provider's long-side budget.

        Returns:
            The resulting image path (possibly the input itself).

        Raises:
            Exception: on any generation failure — callers keep the original.
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured and ready."""
        ...


class NullProvider(ImageGenProvider):
    """No-op provider — returns the original image unchanged.

    Used when no image gen provider is configured (the default).
    """

    def transform(self, image_path: Path, prompt: str, **_: object) -> Path:
        return image_path

    def is_available(self) -> bool:
        return True  # always "available" (just does nothing)


# ── Qwen-Image-2.1 (local, diffusers) ───────────────────────────────────────

# Official diffusers-layout files needed at runtime — everything except the
# 14.2 GB bf16 denoiser shards, which the GGUF quant replaces.
_MIRROR = "https://modelscope.cn/models/Qwen/Qwen-Image-2.1/resolve/master/"
_OFFICIAL_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
_GGUF_URL = (
    "https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/"
    "qwen-image-2.1-Q4_K_M.gguf"
)
_GGUF_NAME = "qwen-image-2.1-Q4_K_M.gguf"
_MARKER = ".v2g_complete"


def model_root() -> Path:
    """Effective model root: env override, else the managed quantized bundle."""
    override = settings.imagegen_model.strip()
    if override:
        return Path(override).expanduser()
    return settings.output_root / ".v2g_cache" / "imagegen" / "qwen-image-2.1"


def prepare_model() -> Path:
    """Download the managed bundle once; return the model root.

    No-op when the completion marker exists or an override root is configured.
    Raises on network failure — never leaves a half-trusted bundle behind
    (every file is size-verified before the marker is written).
    """
    root = model_root()
    if settings.imagegen_model.strip():
        log.info("imagegen: using model root override %s", root)
        return root
    if (root / _MARKER).is_file():
        return root

    root.mkdir(parents=True, exist_ok=True)
    log.info("imagegen: fetching Qwen-Image-2.1 bundle into %s (~23 GB, first use)", root)
    for rel in _OFFICIAL_FILES:
        _fetch(_MIRROR + rel, root / rel)
    _fetch(_GGUF_URL, root / "transformer" / _GGUF_NAME)
    (root / _MARKER).write_text("ok\n", encoding="utf-8")
    log.info("imagegen: bundle ready at %s", root)
    return root


# ── Downloader (stdlib, ranged-parallel, resumable) ─────────────────────────

def _content_length(url: str) -> int | None:
    """Total size in bytes, or None when the server does not say."""
    try:
        with urlopen(Request(url, method="HEAD", headers={"User-Agent": "v2g/1.0"}), timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    except OSError as e:
        log.debug("imagegen: HEAD size probe failed for %s: %s", url, e)
    # Mirrors that reject HEAD: ask for the first byte and read the total from
    # Content-Range (a 200 answer to a ranged request still carries the full
    # Content-Length).
    try:
        req = Request(url, headers={"Range": "bytes=0-0", "User-Agent": "v2g/1.0"})
        with urlopen(req, timeout=30) as resp:
            cr = resp.headers.get("Content-Range", "")
            if "/" in cr:
                tail = cr.rsplit("/", 1)[1]
                if tail.isdigit():
                    return int(tail)
            if getattr(resp, "status", 200) == 200:
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return int(cl)
    except OSError as e:
        log.debug("imagegen: ranged size probe failed for %s: %s", url, e)
    return None


def _fetch(url: str, dst: Path, *, streams: int = 6) -> None:
    """Download *url* to *dst* with parallel ranged chunks and resume.

    An exactly-sized file already present is kept. Ranges unknown → single
    stream through a ``.part`` file with an atomic rename. Sizes are checked
    against the server's total — a wrong-sized file is refetched, never trusted.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = _content_length(url)
    if total is None:
        _fetch_stream(url, dst)
        return
    if dst.is_file() and dst.stat().st_size == total:
        return
    if dst.is_file():
        log.info("imagegen: %s has the wrong size — refetching", dst.name)
        dst.unlink()

    parts = dst.parent / f"{dst.name}.parts"
    parts.mkdir(exist_ok=True)
    chunk = max(32 << 20, -(-total // 96))  # ≥32 MB pieces, ≤96 chunks
    ranges = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]
    todo = [
        (i, start, end)
        for i, (start, end) in enumerate(ranges)
        if not _part_done(parts / f"{i}.part", end - start)
    ]
    log.info(
        "imagegen: %s (%.2f GB) — %d/%d chunks to fetch",
        dst.name, total / 1e9, len(todo), len(ranges),
    )
    with ThreadPoolExecutor(max_workers=streams) as pool:
        list(pool.map(lambda t: _fetch_part(url, parts / f"{t[0]}.part", t[1], t[2]), todo))

    tmp = dst.parent / f"{dst.name}.assembling"
    with open(tmp, "wb") as out:
        for i in range(len(ranges)):
            with open(parts / f"{i}.part", "rb") as piece:
                shutil.copyfileobj(piece, out, 1 << 20)
    shutil.rmtree(parts, ignore_errors=True)
    got = tmp.stat().st_size
    if got != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"assembled {dst.name} is {got} bytes, expected {total}")
    os.replace(tmp, dst)
    log.info("imagegen: got %s", dst.name)


def _part_done(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size > expected:  # corrupt/overrun chunk — start it over
        path.unlink()
        return False
    return size == expected


def _fetch_part(url: str, part: Path, start: int, end: int, retries: int = 4) -> None:
    """Fetch bytes ``[start, end)`` into *part*, resuming a partial file."""
    expected = end - start
    for attempt in range(retries):
        have = part.stat().st_size if part.is_file() else 0
        if have > expected:
            part.unlink()
            have = 0
        if have == expected:
            return
        req = Request(
            url,
            headers={"Range": f"bytes={start + have}-{end - 1}", "User-Agent": "v2g/1.0"},
        )
        try:
            with urlopen(req, timeout=60) as resp, open(part, "ab") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
        except (OSError, HTTPException) as e:
            log.debug("imagegen: chunk %d-%d attempt %d failed: %s", start, end, attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
            continue
        if part.is_file() and part.stat().st_size == expected:
            return
        # Short or oversized read — the next attempt resumes or restarts.
        time.sleep(1.0)
    raise RuntimeError(f"download failed for {url} bytes {start}-{end}")


def _fetch_stream(url: str, dst: Path) -> None:
    """Single-stream copy for endpoints that do not report a size."""
    tmp = dst.parent / f"{dst.name}.part"
    complete = False
    try:
        with urlopen(Request(url, headers={"User-Agent": "v2g/1.0"}), timeout=60) as resp, \
                open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        complete = True
    finally:
        if not complete:  # failed mid-copy — never leave a partial behind
            tmp.unlink(missing_ok=True)
    os.replace(tmp, dst)


class QwenImage21Provider(ImageGenProvider):
    """Local Qwen-Image-2.1 via diffusers — restyles images on-device."""

    def __init__(self, *, steps: int | None = None, max_side: int | None = None) -> None:
        self._steps = steps if steps is not None else settings.imagegen_steps
        self._max_side = max_side if max_side is not None else settings.imagegen_max_side
        self._pipe = None  # lazy — loading costs tens of seconds and ~22 GB RAM

    # ── availability ────────────────────────────────────────────────────────

    @staticmethod
    def _import_error() -> str | None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from diffusers import QwenImage21Pipeline  # noqa: F401
        except Exception as e:  # noqa: BLE001 — any import-stage failure means "not ready"
            return f"{type(e).__name__}: {e}"
        return None

    def is_available(self) -> bool:
        err = self._import_error()
        if err:
            log.warning(
                "imagegen: optional dependencies missing (%s) — run `uv sync --extra imagegen`", err
            )
            return False
        return True

    # ── prompt / size (pure helpers) ────────────────────────────────────────

    def _prompt(self, prompt: str, reference: str) -> str:
        pieces = (settings.imagegen_style.strip(), prompt.strip(), reference.strip())
        return ", ".join(p for p in pieces if p)

    def _target_size(self, src: tuple[int, int], size: str) -> tuple[int, int]:
        """Output size: explicit ``WxH`` or the input aspect, capped and /32."""
        w, h = src
        if "x" in size.lower():
            try:
                w, h = (int(v) for v in size.lower().split("x", 1))
            except ValueError:
                pass
        w, h = max(int(w), 1), max(int(h), 1)
        scale = min(1.0, self._max_side / max(w, h))
        tw = max(32, round(w * scale / 32) * 32)
        th = max(32, round(h * (tw / w) / 32) * 32)  # follow the rounded width
        return tw, th

    # ── model loading ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._pipe is not None:
            return
        err = self._import_error()
        if err:
            raise RuntimeError(
                f"imagegen dependencies unavailable ({err}) — run `uv sync --extra imagegen`"
            )
        import torch
        from diffusers import (
            GGUFQuantizationConfig,
            QwenImage21Pipeline,
            QwenImage21Transformer2DModel,
        )
        from transformers import Qwen3VLForConditionalGeneration

        root = prepare_model()
        dtype = torch.bfloat16
        gguf = root / "transformer" / _GGUF_NAME
        if gguf.is_file():
            log.info("imagegen: loading Qwen-Image-2.1 (GGUF Q4 denoiser + bf16 text encoder)")
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                str(root), subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True
            )
            transformer = QwenImage21Transformer2DModel.from_single_file(
                str(gguf),
                config=str(root),
                subfolder="transformer",
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            pipe = QwenImage21Pipeline.from_pretrained(
                str(root), text_encoder=text_encoder, transformer=transformer, torch_dtype=dtype
            )
        else:
            # Override root without the managed GGUF (full bf16 repo, remote id) —
            # let diffusers load the whole pipeline as-is.
            log.info("imagegen: loading Qwen-Image-2.1 from %s", root)
            pipe = QwenImage21Pipeline.from_pretrained(str(root), torch_dtype=dtype)
        self._pipe = self._place(pipe)

    @staticmethod
    def _place(pipe):
        """Pick the heaviest placement the GPU can honestly hold."""
        import torch

        if not torch.cuda.is_available():
            log.info("imagegen: no CUDA GPU — running on CPU (very slow)")
            return pipe
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        if vram_gb >= 28:
            log.info("imagegen: %.0f GB VRAM — weights fully on GPU", vram_gb)
            return pipe.to("cuda")
        if vram_gb >= 20:
            log.info("imagegen: %.0f GB VRAM — model CPU offload", vram_gb)
            pipe.enable_model_cpu_offload()
            return pipe
        log.info("imagegen: %.0f GB VRAM — sequential CPU offload (module-at-a-time)", vram_gb)
        pipe.enable_sequential_cpu_offload()
        return pipe

    # ── generation ──────────────────────────────────────────────────────────

    def transform(
        self,
        image_path: Path,
        prompt: str,
        *,
        reference: str = "",
        size: str = "",
    ) -> Path:
        from PIL import Image

        self._load()

        with Image.open(image_path) as src:
            img = src.copy()  # detach before overwriting the file below
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        w, h = self._target_size(img.size, size)
        full = self._prompt(prompt, reference)
        log.info("imagegen: %s → %dx%d @ %d steps: %s", image_path.name, w, h, self._steps, full)
        out = self._pipe(
            prompt=full,
            image=img,
            width=w,
            height=h,
            num_inference_steps=self._steps,
        ).images[0]
        out.save(image_path)
        return image_path


# ── Provider factory ────────────────────────────────────────────────────────


def get_provider() -> ImageGenProvider:
    """Return the configured image generation provider.

    ``V2G_IMAGEGEN_PROVIDER`` selects it: empty (default) → :class:`NullProvider`,
    ``qwen`` → :class:`QwenImage21Provider` (local Qwen-Image-2.1). Anything
    else is a configuration error and raises.
    """
    name = settings.imagegen_provider.strip().lower()
    if not name:
        return NullProvider()
    if name == "qwen":
        return QwenImage21Provider()
    raise ValueError(
        f"V2G_IMAGEGEN_PROVIDER={settings.imagegen_provider!r} is unknown — use '' (off) or 'qwen'"
    )
