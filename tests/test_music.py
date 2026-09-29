"""Tests for background music — fake trunk (OpenAI format) and local ACE-Step servers."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from v2g import music
from v2g.config import settings
from v2g.llm.analyzer import GameDesign
from v2g.llm.client import ChatResult

FAKE_MP3 = b"ID3-fake-mp3-bytes"


def _design() -> GameDesign:
    return GameDesign(
        title="Music Test",
        genre="visual novel",
        summary="s",
        mechanics=[],
        controls=["Space: advance"],
        style="cinematic, piano",
        atmosphere="mysterious, quiet tension",
        objects=[],
    )


def _serve(handler_cls) -> tuple[str, ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server, thread


class _Recorder:
    """Per-fixture state shared with the handler class."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.no_audio = False
        self.fail_task = False


@pytest.fixture
def chat_music_server():
    """Trunk stand-in speaking OpenAI chat-audio format."""
    state = _Recorder()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep pytest output clean
            pass

        def _json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state.requests.append(
                {
                    "path": self.path,
                    "body": json.loads(raw),
                    "auth": self.headers.get("Authorization"),
                }
            )
            if self.path != "/v1/chat/completions":
                self._json({"error": {"message": "no such route"}}, status=404)
                return
            if state.no_audio:
                self._json({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
                return
            self._json(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                                "audio": {
                                    "id": "audio-1",
                                    "data": base64.b64encode(FAKE_MP3).decode(),
                                    "transcript": "",
                                },
                            }
                        }
                    ]
                }
            )

    url, _server, _thread = _serve(Handler)
    try:
        yield url, state
    finally:
        _server.shutdown()
        _server.server_close()


@pytest.fixture
def acestep_server():
    """Local acestep-api stand-in (release_task / query_result / audio)."""
    state = _Recorder()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep pytest output clean
            pass

        def _json(self, obj) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/audio"):
                self.send_response(200)
                self.send_header("Content-Length", str(len(FAKE_MP3)))
                self.end_headers()
                self.wfile.write(FAKE_MP3)
            else:
                self._json({"data": [], "code": 200, "error": None})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state.requests.append({"path": self.path, "body": json.loads(raw)})
            if self.path == "/release_task":
                self._json({"data": {"task_id": "t1", "status": "queued"}, "code": 200})
            elif self.path == "/query_result":
                status = 2 if state.fail_task else 1
                result = json.dumps([{"file": "/v1/audio?path=x", "status": status}])
                self._json(
                    {
                        "data": [{"task_id": "t1", "status": status, "result": result}],
                        "code": 200,
                    }
                )
            else:
                self._json({"data": None, "code": 404, "error": "no such route"})

    url, _server, _thread = _serve(Handler)
    try:
        yield url, state
    finally:
        _server.shutdown()
        _server.server_close()


def _settings(monkeypatch, tmp_path, base: str, **over) -> None:
    """Point the trunk endpoint at *base*, enable music, default to `api`."""
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "llm_base_url", f"{base}/v1")
    monkeypatch.setattr(settings, "llm_api_key", "sk-trunk")
    monkeypatch.setattr(settings, "music_provider", "api")
    monkeypatch.setattr(settings, "music_model", "music-xl")
    monkeypatch.setattr(settings, "music_duration", 10)
    for key, value in over.items():
        monkeypatch.setattr(settings, key, value)


def test_api_provider_sends_openai_format_and_caches(chat_music_server, tmp_path, monkeypatch):
    url, state = chat_music_server
    _settings(monkeypatch, tmp_path, url)
    run_dir = tmp_path / "run"

    path = music.generate_music(_design(), "vampire theme", run_dir)

    assert path == "res://assets/bgm/bgm.mp3"
    assert (run_dir / "assets" / "bgm" / "bgm.mp3").read_bytes() == FAKE_MP3
    req = state.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sk-trunk"  # the trunk key, nothing else
    body = req["body"]
    assert body["model"] == "music-xl"
    assert body["modalities"] == ["text", "audio"]
    assert body["audio"] == {"format": "mp3", "voice": "alloy"}
    # Strict OpenAI body: no ACE-Step-specific fields...
    assert "duration" not in body
    assert "instrumental" not in body
    assert "lyrics" not in body
    # ...the guidance travels in the prompt text instead.
    content = body["messages"][0]["content"]
    assert "vampire theme" in content
    assert "10 seconds" in content
    assert "no vocals" in content

    # Second run: cache hit, endpoint untouched.
    assert music.generate_music(_design(), "vampire theme", run_dir) == path
    assert len(state.requests) == 1


def test_acestep_local_provider_drives_the_rest_protocol(acestep_server, tmp_path, monkeypatch):
    url, state = acestep_server
    _settings(
        monkeypatch,
        tmp_path,
        url,
        music_provider="acestep",
        music_acestep_url=url,
        music_model="acestep-v15-turbo",
    )
    monkeypatch.setattr(music, "_POLL_INTERVAL", 0.01)
    run_dir = tmp_path / "run"

    path = music.generate_music(_design(), None, run_dir)

    assert path == "res://assets/bgm/bgm.mp3"
    assert (run_dir / "assets" / "bgm" / "bgm.mp3").read_bytes() == FAKE_MP3
    release = state.requests[0]
    assert release["path"] == "/release_task"
    assert release["body"]["model"] == "acestep-v15-turbo"
    assert release["body"]["audio_duration"] == 10
    assert release["body"]["lyrics"] == ""
    assert release["body"]["audio_format"] == "mp3"
    assert any(r["path"] == "/query_result" for r in state.requests)

    # Second run: cache hit — no new REST traffic.
    assert music.generate_music(_design(), None, run_dir) == path
    assert sum(1 for r in state.requests if r["path"] == "/release_task") == 1


def test_acestep_without_a_running_server_returns_none(tmp_path, monkeypatch):
    # Port 9 (discard) on localhost: refused — no server, no launch attempt.
    _settings(
        monkeypatch,
        tmp_path,
        "http://127.0.0.1:9",
        music_provider="acestep",
        music_acestep_url="http://127.0.0.1:9",
    )

    assert music.generate_music(_design(), None, tmp_path / "run") is None


def test_music_is_off_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "music_model", "")

    assert music.generate_music(_design(), None, tmp_path / "run") is None


def test_response_without_audio_is_skipped_without_raising(
    chat_music_server, tmp_path, monkeypatch
):
    url, state = chat_music_server
    state.no_audio = True
    _settings(monkeypatch, tmp_path, url)

    assert music.generate_music(_design(), None, tmp_path / "run") is None
    # Nothing was written into the project or cached — the next run may retry.
    assert not (tmp_path / "run" / "assets" / "bgm" / "bgm.mp3").exists()


def test_unknown_provider_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "music_provider", "nope")
    monkeypatch.setattr(settings, "music_model", "music-xl")

    assert music.generate_music(_design(), None, tmp_path / "run") is None


def test_music_prompt_carries_atmosphere_style_theme_and_instrumental_directive():
    prompt = music.music_prompt(_design(), "cyberpunk")

    assert "mysterious, quiet tension" in prompt
    assert "cinematic, piano" in prompt
    assert "cyberpunk" in prompt
    assert "no vocals" in prompt


# ── `llm` provider: the text model writes the synth script ──────────────────


_WAV_SCRIPT = """\
import sys, wave
w = wave.open(sys.argv[1], "wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(8000)
w.writeframes(b"\\x00\\x00" * 800)
w.close()
"""


def _settings_llm(monkeypatch, tmp_path, code: str) -> None:
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "music_provider", "llm")
    monkeypatch.setattr(settings, "music_model", "mimo-v2.6-flash")
    monkeypatch.setattr(settings, "music_duration", 1)
    monkeypatch.setattr(music, "chat", lambda *a, **k: ChatResult(code, False, "k"))


def test_llm_provider_executes_the_generated_script(tmp_path, monkeypatch):
    _settings_llm(monkeypatch, tmp_path, _WAV_SCRIPT)
    run_dir = tmp_path / "run"

    path = music.generate_music(_design(), None, run_dir)

    assert path == "res://assets/bgm/bgm.mp3"
    data = (run_dir / "assets" / "bgm" / "bgm.mp3").read_bytes()
    assert data[:3] == b"ID3"  # ffmpeg turned the script's wav into an mp3
    # The script itself is kept for inspection.
    assert (run_dir / "work" / "bgm_llm" / "bgm_llm.py").read_text(encoding="utf-8") == (
        _WAV_SCRIPT.strip()
    )

    # Second run: served from the audio cache — chat is not called again.
    def _boom(*_a, **_k):
        raise AssertionError("chat must not run on a cache hit")

    monkeypatch.setattr(music, "chat", _boom)
    assert music.generate_music(_design(), None, run_dir) == path


def test_llm_provider_failing_script_is_skipped_without_raising(tmp_path, monkeypatch, caplog):
    _settings_llm(monkeypatch, tmp_path, "import sys\nsys.exit(3)\n")
    run_dir = tmp_path / "run"

    with caplog.at_level(music.runlog.NOTICE):
        assert music.generate_music(_design(), None, run_dir) is None

    assert not (run_dir / "assets" / "bgm" / "bgm.mp3").exists()
    assert any("BGM: skipped" in m for m in caplog.messages)
