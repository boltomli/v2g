"""Tests for voice-over synthesis — providers are faked, nothing hits the network."""

from pathlib import Path

from v2g import tts
from v2g.config import settings


class FakeProvider(tts.TTSProvider):
    """In-memory provider: audio bytes are ``<voice>:<text>``."""

    name = "fake"
    ext = "mp3"

    def __init__(self, *, voices: list[str] | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self._voices = voices or ["v0", "v1"]
        self.fail = fail

    def default_voices(self) -> list[str]:
        return list(self._voices)

    def synthesize(self, text: str, voice: str) -> bytes | None:
        self.calls.append((text, voice))
        if self.fail:
            raise tts.TTSUnavailable("boom")
        return f"{voice}:{text}".encode()


def _synth(tmp_path: Path, provider: FakeProvider) -> tts.Synthesizer:
    return tts.Synthesizer(provider, tmp_path / "run" / "assets" / "voice")


def test_speak_writes_project_clip_and_serves_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    provider = FakeProvider()
    synth = _synth(tmp_path, provider)

    path = synth.speak("你好世界", "")

    assert path is not None and path.startswith("res://assets/voice/")
    clip = tmp_path / "run" / "assets" / "voice" / path.rsplit("/", 1)[1]
    assert clip.read_bytes() == "v0:你好世界".encode()
    assert synth.voiced == 1

    # Same line again: the cache answers, the provider is not called twice.
    again = synth.speak("你好世界", "")
    assert again == path
    assert len(provider.calls) == 1
    assert synth.voiced == 2
    assert (tmp_path / ".v2g_cache" / "tts").is_dir()


def test_narrator_keeps_first_voice_and_speakers_rotate(tmp_path):
    provider = FakeProvider(voices=["v0", "v1", "v2"])
    synth = _synth(tmp_path, provider)

    assert synth.voice_for("") == "v0"  # narration always voice 0
    assert synth.voice_for("") == "v0"
    assert synth.voice_for("alice") == "v1"
    assert synth.voice_for("bob") == "v2"
    assert synth.voice_for("carol") == "v1"  # wraps around
    assert synth.voice_for("alice") == "v1"  # stable once assigned


def test_provider_failure_disables_voice_over_with_one_warning(tmp_path, caplog):
    provider = FakeProvider(fail=True)
    synth = _synth(tmp_path, provider)

    with caplog.at_level("WARNING"):
        assert synth.speak("第一句", "") is None
        assert synth.enabled is False
        # Disabled: later lines are skipped without touching the provider.
        assert synth.speak("第二句", "") is None

    assert len(provider.calls) == 1
    assert sum("Voice-over disabled" in m for m in caplog.messages) == 1


def test_overlong_line_is_skipped(tmp_path):
    provider = FakeProvider()
    synth = _synth(tmp_path, provider)

    assert synth.speak("x" * (tts._MAX_CHARS + 1), "") is None
    assert provider.calls == []
    assert synth.voiced == 0


def test_empty_text_and_disabled_state_do_not_synthesize(tmp_path):
    provider = FakeProvider()
    synth = _synth(tmp_path, provider)

    assert synth.speak("   ", "") is None
    assert provider.calls == []


def test_build_synth_follows_the_model_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)

    # Off by default: no model → no synthesizer, nothing touched.
    monkeypatch.setattr(settings, "tts_model", "")
    assert tts.build_synth(tmp_path) is None

    # Model set but no trunk key → refused up front, not on the first line.
    monkeypatch.setattr(settings, "tts_model", "gpt-4o-mini-tts")
    monkeypatch.setattr(settings, "llm_api_key", "")
    assert tts.build_synth(tmp_path) is None

    # Model + key → ready (construction makes no network calls).
    monkeypatch.setattr(settings, "llm_api_key", "sk")
    built = tts.build_synth(tmp_path)
    assert built is not None
    assert built.provider.name == "api"
