"""Provider selection, prompt/size helpers, and the model downloader.

No model and no real network: availability is injected, the fetch tests run
against a local Range-capable HTTP server.
"""

import threading
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from v2g.config import settings
from v2g.llm.image_gen import (
    NullProvider,
    QwenImage21Provider,
    _fetch,
    get_provider,
    model_root,
    prepare_model,
)


def test_default_is_noop_provider(monkeypatch):
    monkeypatch.setattr(settings, "imagegen_provider", "")
    provider = get_provider()
    assert isinstance(provider, NullProvider)
    assert provider.is_available()
    src = Path("anything.png")
    assert provider.transform(src, "restyle me") == src


def test_unknown_provider_is_a_config_error(monkeypatch):
    monkeypatch.setattr(settings, "imagegen_provider", "dall-e-3")
    with pytest.raises(ValueError, match="V2G_IMAGEGEN_PROVIDER"):
        get_provider()


def test_qwen_provider_selected(monkeypatch):
    monkeypatch.setattr(settings, "imagegen_provider", "qwen")
    assert isinstance(get_provider(), QwenImage21Provider)


def test_qwen_availability_tracks_optional_dependencies(monkeypatch):
    provider = QwenImage21Provider()
    monkeypatch.setattr(
        QwenImage21Provider, "_import_error", staticmethod(lambda: "ImportError: gone")
    )
    assert provider.is_available() is False
    monkeypatch.setattr(QwenImage21Provider, "_import_error", staticmethod(lambda: None))
    assert provider.is_available() is True


def test_failed_model_load_is_not_retried(monkeypatch):
    """A load that failed is remembered — later candidates must not reload ~22 GB."""
    provider = QwenImage21Provider()
    attempts: list[int] = []

    def boom() -> None:
        attempts.append(1)
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(provider, "_build_pipe", boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        provider._load()
    with pytest.raises(RuntimeError, match="already failed: RuntimeError: CUDA error") as second:
        provider._load()
    assert attempts == [1]  # the bundle was built once, not once per candidate
    assert "out of memory" in str(second.value)  # the original cause stays visible


def test_prompt_prepends_the_global_style_prefix(monkeypatch):
    monkeypatch.setattr(settings, "imagegen_style", "oil painting")
    provider = QwenImage21Provider()
    assert provider._prompt("medieval castle") == "oil painting, medieval castle"
    monkeypatch.setattr(settings, "imagegen_style", "")
    assert provider._prompt("castle") == "castle"
    assert provider._prompt("") == ""


def test_target_size_caps_aspect_and_stays_on_grid():
    provider = QwenImage21Provider(steps=1, max_side=1024)
    assert provider._target_size((1920, 1080), "") == (1024, 576)  # 16:9, capped
    assert provider._target_size((400, 900), "") == (384, 864)  # aspect kept
    assert provider._target_size((64, 64), "") == (64, 64)
    assert provider._target_size((10, 10), "") == (32, 32)  # never below the grid
    assert provider._target_size((1920, 1080), "512x512") == (512, 512)
    for w, h in ((1024, 576), (384, 864), (32, 32), (512, 512)):
        assert w % 32 == 0 and h % 32 == 0


def test_plain_non_linear_weights_dequantizes_direct_read_quants():
    """Norm-style weights leave the GGUF as plain tensors; linears stay quantized.

    The imagegen extra carries diffusers — skip cleanly when it is not installed.
    """
    torch = pytest.importorskip("torch")
    gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

    class _Norm(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self._parameters["weight"] = weight

    # BF16-in-GGUF: 4096 values stored as 8192 raw bytes (quant type 30).
    raw = gguf_utils.GGUFParameter(torch.zeros(8192, dtype=torch.uint8), quant_type=30)
    norm = _Norm(raw)
    plain_id = gguf_utils.GGUFParameter(torch.zeros(4, dtype=torch.uint8), quant_type=30)
    linear = gguf_utils.GGUFLinear(4, 4)
    linear.weight = torch.nn.Parameter(plain_id, requires_grad=False)

    holder = torch.nn.Module()
    holder.norm = norm
    holder.proj = linear

    QwenImage21Provider._plain_non_linear_weights(holder)

    fixed = holder.norm.weight
    assert not isinstance(fixed, gguf_utils.GGUFParameter)
    assert fixed.dtype == torch.bfloat16 and tuple(fixed.shape) == (4096,)
    # The linear keeps its quantized weight — GGUFLinear dequantizes per forward.
    assert isinstance(holder.proj.weight, gguf_utils.GGUFParameter)


def test_transform_resizes_condition_and_disables_kv_cache_on_low_vram(tmp_path):
    """Low-VRAM placement: condition at output size, no prefix KV cache."""
    from types import SimpleNamespace

    from PIL import Image

    calls: list[dict] = []

    class FakePipe:
        _v2g_low_vram = True

        def __call__(self, **kwargs):
            calls.append(kwargs)
            result = SimpleNamespace(
                images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "red")]
            )
            return result

    src = tmp_path / "a.png"
    Image.new("RGB", (1080, 1922), "blue").save(src)
    provider = QwenImage21Provider(steps=7, max_side=1024)
    provider._pipe = FakePipe()  # _load short-circuits on a preset pipe

    out = provider.transform(src, "oil style")

    assert out == src
    kw = calls[0]
    assert (kw["width"], kw["height"]) == (576, 1024)
    assert kw["image"].size == (576, 1024)  # condition resized to the canvas
    assert kw["use_kv_cache"] is False
    assert kw["num_inference_steps"] == 7
    with Image.open(src) as check:
        assert check.size == (576, 1024)  # saved in place at the new size

    # Plenty of VRAM: KV cache stays on, nothing extra is passed.
    class BigPipe(FakePipe):
        _v2g_low_vram = False

    calls.clear()
    provider._pipe = BigPipe()
    Image.new("RGB", (576, 1024), "blue").save(src)
    provider.transform(src, "oil style")
    assert "use_kv_cache" not in calls[0]


def test_transform_text_only_skips_the_condition_and_writes_to_dest(tmp_path):
    """condition=False is the t2i candidate: no source pixels, result to dest."""
    from types import SimpleNamespace

    from PIL import Image

    calls: list[dict] = []

    class FakePipe:
        _v2g_low_vram = False

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "red")]
            )

    src = tmp_path / "a.png"
    Image.new("RGB", (1080, 1922), "blue").save(src)
    dest = tmp_path / "text.png"
    provider = QwenImage21Provider(steps=7, max_side=1024)
    provider._pipe = FakePipe()

    out = provider.transform(src, "redraw brief", condition=False, dest=dest)

    assert out == dest
    assert calls[0]["image"] is None  # generated from the brief alone
    assert (calls[0]["width"], calls[0]["height"]) == (576, 1024)  # aspect still from source
    assert dest.is_file() and src.read_bytes() != dest.read_bytes()


def test_place_strategy_depends_on_vram_and_gguf(monkeypatch):
    """6 GB + GGUF keeps the denoiser resident; bigger cards offload heavier."""
    import types

    torch = pytest.importorskip("torch")

    from v2g.llm.image_gen import QwenImage21Provider

    class FakePipe:
        def __init__(self):
            self.actions: list = []
            self.vae = self._FakeVae(self)

        class _FakeVae:
            def __init__(self, owner):
                self._owner = owner

            def enable_tiling(self):
                self._owner.actions.append("vae_tiling")

        def to(self, device):
            self.actions.append(("to", device))
            return self

        def enable_model_cpu_offload(self):
            self.actions.append("model_offload")

        def enable_sequential_cpu_offload(self):
            self.actions.append("sequential_offload")

    def with_vram(gb: int) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda i: types.SimpleNamespace(total_memory=int(gb * 2**30)),
        )

    with_vram(6)
    pipe = FakePipe()
    QwenImage21Provider._place(pipe, gguf=True)
    # The GGUF tensor cannot survive meta-device offload — it must stay resident.
    assert pipe._exclude_from_cpu_offload == ["transformer"]
    assert pipe.actions == ["vae_tiling", "sequential_offload"]  # tiles tame the decode spike

    pipe = FakePipe()
    QwenImage21Provider._place(pipe, gguf=False)
    assert getattr(pipe, "_exclude_from_cpu_offload", []) == []  # plain bf16: meta-safe
    assert pipe.actions == ["vae_tiling", "sequential_offload"]

    with_vram(24)
    pipe = FakePipe()
    QwenImage21Provider._place(pipe, gguf=True)
    assert pipe.actions == ["model_offload"]  # roomy cards skip tiling

    with_vram(32)
    pipe = FakePipe()
    QwenImage21Provider._place(pipe, gguf=True)
    assert pipe.actions == [("to", "cuda")]


def test_prepare_model_short_circuits_on_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "imagegen_model", "")
    root = model_root()
    root.mkdir(parents=True)
    (root / ".v2g_complete").write_text("ok", encoding="utf-8")
    assert prepare_model() == root  # marker hit — never touches the network


def test_prepare_model_prefers_override(tmp_path, monkeypatch):
    override = tmp_path / "custom-root"
    monkeypatch.setattr(settings, "imagegen_model", str(override))
    assert prepare_model() == override  # user-managed root — no download


# ── downloader (local server speaking the mirrors' contract) ────────────────


class _RangeHandler(BaseHTTPRequestHandler):
    """Static file server with HEAD + Range support."""

    data = b""

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.data)))
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        start, end = 0, len(self.data) - 1
        if rng and rng.startswith("bytes="):
            first, _, last = rng[len("bytes=") :].partition("-")
            start = int(first)
            if last:
                end = min(int(last), len(self.data) - 1)
        body = self.data[start : end + 1]
        if rng:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _digest(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def test_fetch_assembles_chunks_resumes_and_skips_complete(tmp_path):
    src = tmp_path / "src.bin"
    with open(src, "wb") as f:  # 65.6 MB → two 32 MB chunks
        f.writelines(b"0123456789abcdef" * 10_000 for _ in range(410))
    _RangeHandler.data = src.read_bytes()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/blob.bin"
        dst = tmp_path / "out" / "blob.bin"
        dst.parent.mkdir()
        dst.write_bytes(b"stale, wrong size")  # wrong-sized leftover → refetched
        parts = dst.parent / "blob.bin.parts"  # a partial chunk → resumed
        parts.mkdir()
        (parts / "0.part").write_bytes(_RangeHandler.data[:1000])

        _fetch(url, dst, streams=3)

        assert _digest(dst) == _digest(src)
        assert not parts.exists()  # parts folded into the final file

        before = dst.stat().st_mtime_ns
        _fetch(url, dst, streams=3)  # exact size present → untouched
        assert dst.stat().st_mtime_ns == before
    finally:
        server.shutdown()
