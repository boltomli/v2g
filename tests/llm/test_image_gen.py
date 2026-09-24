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


def test_prompt_joins_style_instruction_reference(monkeypatch):
    monkeypatch.setattr(settings, "imagegen_style", "oil painting")
    provider = QwenImage21Provider()
    assert (
        provider._prompt("medieval castle", "tall knight")
        == "oil painting, medieval castle, tall knight"
    )
    monkeypatch.setattr(settings, "imagegen_style", "")
    assert provider._prompt("castle", "") == "castle"
    assert provider._prompt("", "") == ""


def test_target_size_caps_aspect_and_stays_on_grid():
    provider = QwenImage21Provider(steps=1, max_side=1024)
    assert provider._target_size((1920, 1080), "") == (1024, 576)  # 16:9, capped
    assert provider._target_size((400, 900), "") == (384, 864)  # aspect kept
    assert provider._target_size((64, 64), "") == (64, 64)
    assert provider._target_size((10, 10), "") == (32, 32)  # never below the grid
    assert provider._target_size((1920, 1080), "512x512") == (512, 512)
    for w, h in ((1024, 576), (384, 864), (32, 32), (512, 512)):
        assert w % 32 == 0 and h % 32 == 0


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
            first, _, last = rng[len("bytes="):].partition("-")
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
