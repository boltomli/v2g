"""Shared fixtures: neutralize ambient .env audio knobs for every test.

The audio layers read ``V2G_*`` settings directly; a developer's local
``.env`` (loaded by ``Settings``) must not leak into test expectations —
each test sets exactly the knobs it needs.
"""

import pytest

from v2g.config import settings

_AUDIO_DEFAULTS = {
    "tts_model": "",
    "tts_voices": "",
    "tts_text": "zh",
    "music_model": "",
    "music_provider": "api",
    "music_acestep_url": "http://127.0.0.1:8001",
    "music_duration": 60,
    "sfx_model": "",
}


@pytest.fixture(autouse=True)
def _isolated_audio_settings(monkeypatch):
    for attr, value in _AUDIO_DEFAULTS.items():
        monkeypatch.setattr(settings, attr, value)
