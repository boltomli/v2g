"""Tests for the OpenAI chat-audio client: voice config, assistant retry,
and the /audio/speech → chat-TTS fallback — against a fake gateway."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from v2g import audio_api, tts
from v2g.config import settings

FAKE_MP3 = b"ID3-fake-tts-bytes"


@pytest.fixture
def gateway_server():
    """Fake gateway: /v1/audio/speech missing (404), chat-TTS needs an
    assistant message, answers with OpenAI-shaped audio."""
    state = {"requests": [], "speech_status": 404, "hard_fail": False}

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
            state["requests"].append({"path": self.path, "body": json.loads(raw)})
            if self.path == "/v1/audio/speech":
                if state["speech_status"] == 200:
                    self._bytes(FAKE_MP3)
                else:
                    self._json({"error": "no such route"}, status=state["speech_status"])
                return
            if self.path != "/v1/chat/completions":
                self._json({"error": "no such route"}, status=404)
                return
            if state["hard_fail"]:
                self._json(
                    {
                        "error": {
                            "code": "400",
                            "message": "Unknown voice: alloy. Available voices: [mimo_default]",
                        }
                    },
                    status=400,
                )
                return
            messages = state["requests"][-1]["body"].get("messages", [])
            if not any(m.get("role") == "assistant" for m in messages):
                self._json(
                    {
                        "error": {
                            "code": "400",
                            "message": "messages must contain an assistant role for TTS model",
                        }
                    },
                    status=400,
                )
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

        def _bytes(self, data: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()


def _settings(monkeypatch, tmp_path, base: str, **over) -> None:
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "llm_base_url", f"{base}/v1")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test")
    for key, value in over.items():
        monkeypatch.setattr(settings, key, value)


def test_chat_audio_retries_once_with_an_assistant_message(gateway_server, tmp_path, monkeypatch):
    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url)

    data = audio_api.chat_audio("tts-model", "你好，测试。")

    assert data == FAKE_MP3
    assert len(state["requests"]) == 2
    first = state["requests"][0]["body"]["messages"]
    second = state["requests"][1]["body"]["messages"]
    assert all(m["role"] == "user" for m in first)  # strict OpenAI shape first
    assert second[-1] == {"role": "assistant", "content": "你好，测试。"}  # their protocol


def test_chat_audio_voice_comes_from_tts_voices_with_override(
    gateway_server, tmp_path, monkeypatch
):
    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url, tts_voices="mimo_default,冰糖")

    audio_api.chat_audio("tts-model", "第一句")
    assert state["requests"][0]["body"]["audio"]["voice"] == "mimo_default"

    audio_api.chat_audio("tts-model", "第二句", voice="茉莉")
    assert state["requests"][2]["body"]["audio"]["voice"] == "茉莉"


def test_chat_audio_http_error_surfaces_the_error_body(gateway_server, tmp_path, monkeypatch):
    url, state = gateway_server
    state["hard_fail"] = True
    _settings(monkeypatch, tmp_path, url)

    with pytest.raises(audio_api.AudioAPIError) as exc:
        audio_api.chat_audio("tts-model", "你好")

    # The gateway's voice list reaches the message — visible in the run log.
    assert "Unknown voice: alloy" in str(exc.value)
    assert "mimo_default" in str(exc.value)
    assert len(state["requests"]) == 1  # body has no "assistant" → no retry


def test_opentts_falls_back_to_chat_when_speech_route_is_missing(
    gateway_server, tmp_path, monkeypatch
):
    from v2g.llm import client as llm_client

    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url, tts_model="mimo-v2.5-tts")
    monkeypatch.setattr(llm_client, "_client", None)  # rebuild against the fake URL

    data = tts.OpenAITTS().synthesize("配音测试", "alloy")

    assert data == FAKE_MP3
    paths = [r["path"] for r in state["requests"]]
    assert paths[0] == "/v1/audio/speech"  # tried the OpenAI route first
    assert paths[-1] == "/v1/chat/completions"  # fell back to chat-TTS
    assert state["requests"][-1]["body"]["model"] == "mimo-v2.5-tts"


def test_opentts_non_404_speech_error_does_not_fall_back(gateway_server, tmp_path, monkeypatch):
    from v2g.llm import client as llm_client

    url, state = gateway_server
    state["speech_status"] = 500
    _settings(monkeypatch, tmp_path, url, tts_model="mimo-v2.5-tts")
    monkeypatch.setattr(llm_client, "_client", None)

    with pytest.raises(tts.TTSUnavailable):
        tts.OpenAITTS().synthesize("配音测试", "alloy")

    # No chat request: a broken (non-404) speech route never falls back.
    assert all(r["path"] == "/v1/audio/speech" for r in state["requests"])
    assert not any(r["path"] == "/v1/chat/completions" for r in state["requests"])


# ── Official TTS chat protocol: text on `assistant`, style on `user` ────────


def test_chat_speech_puts_text_on_the_assistant_turn(gateway_server, tmp_path, monkeypatch):
    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url, tts_voices="mimo_default,冰糖")

    data = audio_api.chat_speech("mimo-v2.5-tts", "夜色落下。")

    assert data == FAKE_MP3
    assert len(state["requests"]) == 1  # no protocol retry — the shape is the documented one
    body = state["requests"][0]["body"]
    assert body["messages"] == [{"role": "assistant", "content": "夜色落下。"}]
    assert body["audio"] == {"format": "mp3", "voice": "mimo_default"}


def test_chat_speech_style_rides_the_user_turn(gateway_server, tmp_path, monkeypatch):
    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url)

    audio_api.chat_speech("mimo-v2.5-tts", "太好了！", voice="茉莉", style="用轻快上扬的语调")

    body = state["requests"][0]["body"]
    assert body["messages"] == [
        {"role": "user", "content": "用轻快上扬的语调"},
        {"role": "assistant", "content": "太好了！"},
    ]
    assert body["audio"]["voice"] == "茉莉"


def test_chat_speech_voice_design_takes_a_description_not_an_id(
    gateway_server, tmp_path, monkeypatch
):
    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url)

    audio_api.chat_speech(
        "mimo-v2.5-tts-voicedesign",
        "夜色落下。",
        voice="alloy",  # must NOT reach the endpoint
        style="中年男性，纪录片旁白风格，嗓音低沉带磁性。",
    )

    body = state["requests"][0]["body"]
    assert body["messages"] == [
        {"role": "user", "content": "中年男性，纪录片旁白风格，嗓音低沉带磁性。"},
        {"role": "assistant", "content": "夜色落下。"},
    ]
    assert "voice" not in body["audio"]  # the gateway 400s on audio.voice here


def test_chat_speech_voice_design_without_style_still_sends_a_user_turn(
    gateway_server, tmp_path, monkeypatch
):
    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url)

    # An empty user turn is a 400 ("user message content must not be empty").
    audio_api.chat_speech("mimo-v2.5-tts-voicedesign", "夜色落下。")

    messages = state["requests"][0]["body"]["messages"]
    assert messages[0]["role"] == "user" and messages[0]["content"].strip()
    assert messages[-1] == {"role": "assistant", "content": "夜色落下。"}


def test_is_voice_design_matches_the_official_model_ids():
    assert audio_api.is_voice_design("mimo-v2.5-tts-voicedesign")
    assert audio_api.is_voice_design("mimo-v2.5-tts-VoiceDesign")
    assert not audio_api.is_voice_design("mimo-v2.5-tts")
    assert not audio_api.is_voice_design("mimo-v2.5-tts-voiceclone")
    assert not audio_api.is_voice_design("gpt-4o-mini-tts")


def test_opentts_voice_design_goes_straight_to_chat(gateway_server, tmp_path, monkeypatch):
    from v2g.llm import client as llm_client

    url, state = gateway_server
    _settings(monkeypatch, tmp_path, url, tts_model="mimo-v2.5-tts-voicedesign")
    monkeypatch.setattr(llm_client, "_client", None)

    data = tts.OpenAITTS().synthesize("夜色落下。", "中年男性，纪录片旁白风格。")

    assert data == FAKE_MP3
    # /audio/speech cannot carry a description — the voice-design protocol is chat-only.
    assert [r["path"] for r in state["requests"]] == ["/v1/chat/completions"]
    body = state["requests"][0]["body"]
    assert body["messages"][0] == {
        "role": "user",
        "content": "中年男性，纪录片旁白风格。",
    }
    assert "voice" not in body["audio"]
