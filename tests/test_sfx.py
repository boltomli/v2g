"""Tests for sound effects — fake LLM answers and a fake trunk audio endpoint."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from v2g import sfx
from v2g.config import settings
from v2g.llm.analyzer import DialogueSample, GameDesign
from v2g.llm.client import ChatResult

FAKE_MP3 = b"ID3-fake-sfx-bytes"


@pytest.fixture
def audio_server():
    """Local stand-in for the trunk chat-completions audio endpoint."""
    state = {"requests": [], "fail": None}

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
            state["requests"].append(
                {
                    "path": self.path,
                    "body": json.loads(raw),
                    "auth": self.headers.get("Authorization"),
                }
            )
            if state["fail"]:
                self._json({"detail": "nope"}, status=state["fail"])
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()


def _design(samples: int = 2) -> GameDesign:
    return GameDesign(
        title="Sfx Test",
        genre="visual novel",
        summary="s",
        mechanics=[],
        controls=["Space: advance"],
        style="cinematic",
        objects=[],
        atmosphere="mysterious",
        dialogue_samples=[
            DialogueSample(speaker="", line="", line_zh=f"第{i}句") for i in range(samples)
        ],
    )


def _settings(monkeypatch, tmp_path, base: str, **over) -> None:
    """Point the trunk endpoint at *base* and enable SFX."""
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "llm_base_url", f"{base}/v1")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test")
    monkeypatch.setattr(settings, "sfx_model", "sfx-xl")
    for key, value in over.items():
        monkeypatch.setattr(settings, key, value)


def test_event_and_cue_clips_land_in_project_and_cache(audio_server, tmp_path, monkeypatch):
    url, state = audio_server
    _settings(monkeypatch, tmp_path, url)
    run_sfx = sfx.Sfx(tmp_path / "run" / "assets" / "sfx")
    run_sfx.cues = ["door knock", ""]

    select = run_sfx.event("select")
    cue = run_sfx.step_cue(0)

    assert select == "res://assets/sfx/select.mp3"
    assert cue == "res://assets/sfx/cue_0.mp3"
    assert (tmp_path / "run" / "assets" / "sfx" / "select.mp3").read_bytes() == FAKE_MP3
    assert run_sfx.step_cue(1) is None  # empty cue stays silent
    assert run_sfx.generated == 2
    req = state["requests"][0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sk-test"
    body = req["body"]
    assert body["model"] == "sfx-xl"
    assert body["modalities"] == ["text", "audio"]
    assert body["audio"] == {"format": "mp3", "voice": "alloy"}
    assert "duration" not in body  # the TTS model reads text verbatim — no spoken hints
    assert body["messages"][0]["content"] == "叮——"  # vocal event word, exactly
    assert len(state["requests"]) == 2

    # A fresh builder (rerun): the audio cache answers — no new requests.
    rerun = sfx.Sfx(tmp_path / "run2" / "assets" / "sfx")
    rerun.cues = ["door knock", ""]
    assert rerun.event("select") == "res://assets/sfx/select.mp3"
    assert rerun.step_cue(0) == "res://assets/sfx/cue_0.mp3"
    assert len(state["requests"]) == 2
    assert rerun.generated == 2  # counted per run, served from cache


def test_unknown_event_and_out_of_range_cues_are_none(tmp_path):
    builder = sfx.Sfx(tmp_path / "assets" / "sfx")
    builder.cues = ["rain"]

    assert builder.event("nope") is None
    assert builder.step_cue(1) is None
    assert builder.step_cue(-1) is None
    assert builder.generated == 0


def test_endpoint_failure_disables_sfx_with_one_warning(
    audio_server, tmp_path, monkeypatch, caplog
):
    url, state = audio_server
    state["fail"] = 401
    _settings(monkeypatch, tmp_path, url)
    builder = sfx.Sfx(tmp_path / "assets" / "sfx")

    with caplog.at_level("WARNING"):
        assert builder.event("select") is None
        assert builder.enabled is False
        assert builder.event("transition") is None

    assert len(state["requests"]) == 1
    assert sum("Sound effects disabled" in m for m in caplog.messages) == 1


def test_build_sfx_follows_model_switches_with_tts_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "sfx_model", "")
    monkeypatch.setattr(settings, "tts_model", "")

    # No model at all → off, no builder, no LLM call.
    assert sfx.build_sfx(tmp_path, _design()) is None

    # The voice-over model backs SFX when no dedicated model is set.
    monkeypatch.setattr(settings, "tts_model", "mimo-v2.5-tts")
    monkeypatch.setattr(sfx, "derive_cues", lambda design: ["", "咚咚咚"])
    built = sfx.build_sfx(tmp_path, _design())
    assert built is not None
    assert built.model == "mimo-v2.5-tts"
    assert built.cues == ["", "咚咚咚"]

    # A dedicated SFX model wins over the fallback.
    monkeypatch.setattr(settings, "sfx_model", "sfx-xl")
    assert sfx.build_sfx(tmp_path, _design()).model == "sfx-xl"


def test_derive_cues_parses_normalizes_and_caps(monkeypatch):
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult('["", "door knock"]', False, "k"))
    assert sfx.derive_cues(_design(2)) == ["", "door knock"]

    # Length mismatch pads with silence.
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult('["only one"]', False, "k"))
    assert sfx.derive_cues(_design(3)) == ["only one", "", ""]

    # Object envelope is accepted too.
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult('{"cues": ["a", "b"]}', False, "k"))
    assert sfx.derive_cues(_design(2)) == ["a", "b"]

    # Unusable output → None, never an exception.
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult("no json here", False, "k"))
    assert sfx.derive_cues(_design(2)) is None

    # Chat raising → None.
    def _boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(sfx, "chat", _boom)
    assert sfx.derive_cues(_design(2)) is None


def test_derive_cues_caps_non_empty_entries(monkeypatch):
    n = sfx._MAX_CUES + 3
    answers = ",".join(f'"cue {i}"' for i in range(n))
    monkeypatch.setattr(sfx, "chat", lambda *a, **k: ChatResult(f"[{answers}]", False, "k"))

    cues = sfx.derive_cues(_design(n))

    assert cues is not None
    assert len(cues) == n
    assert sum(1 for c in cues if c) == sfx._MAX_CUES
