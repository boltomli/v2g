"""Throwaway: prove local Qwen-Image-2.1 generation once the bundle lands.

Waits for the managed bundle's completion marker (written by prepare_model),
then runs one real img2img transform on a pipeline asset and reports
wall-clock split (load vs generation), output validity, and peak VRAM.
Delete after use.
"""

import hashlib
import shutil
import time
from pathlib import Path

from v2g.llm.image_gen import QwenImage21Provider

ROOT = Path("projects/.v2g_cache/imagegen/qwen-image-2.1")
ASSETS = Path("projects/20260922-163646_video__10/assets")


def main() -> None:
    deadline = time.time() + 3 * 3600
    while not (ROOT / ".v2g_complete").exists():
        if time.time() > deadline:
            raise SystemExit("PROOF FAILED: timed out waiting for the model bundle")
        time.sleep(20)
    print(f"bundle ready: {ROOT}", flush=True)

    src = ASSETS / "background.png"
    if not src.exists():
        src = next(ASSETS.glob("*.png"))
    tmp = ROOT.parent / "_proof_input.png"
    shutil.copy(src, tmp)

    import torch
    from PIL import Image

    with Image.open(src) as im:
        src_size = im.size

    provider = QwenImage21Provider()
    assert provider.is_available(), "provider not available"
    expected = provider._target_size(src_size, "")  # portrait 1080x1922 → (576, 1024)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    provider._load()
    t1 = time.time()
    print(f"model load: {t1 - t0:.1f}s", flush=True)

    out = provider.transform(
        tmp,
        "medieval oil painting style, warm candlelight",
        reference="stone courtyard, white dress lady",
    )
    t2 = time.time()

    img = Image.open(out)
    changed = hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(out.read_bytes()).digest()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"PROOF OK: {out} size={img.size} mode={img.mode} bytes={out.stat().st_size}\n"
        f"changed={changed} generation={t2 - t1:.1f}s total={t2 - t0:.1f}s "
        f"peak_vram={peak:.2f} GiB",
        flush=True,
    )
    assert changed, "output is byte-identical to input — nothing was generated"
    assert img.size == expected, f"expected {expected}, got {img.size}"


if __name__ == "__main__":
    main()
