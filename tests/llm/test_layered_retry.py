"""Layered retry policy + analysis checkpoint: no useless retries.

Policy: a *cached* answer that fails parsing is invalidated and refetched
once; a fresh answer that fails parsing is never re-requested.
"""

import json

import pytest

from v2g.config import settings
from v2g.godot import generator
from v2g.llm import analyzer
from v2g.llm.client import ChatResult
from v2g.llm.errors import LLMOutputError

_VALID = (
    '{"title":"T","genre":"g","summary":"s","mechanics":["m"],"controls":["c"],'
    '"style":"st","objects":[]}'
)


class _FakeChat:
    """Stands in for client.chat; returns canned ChatResults in order."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.kwds = []

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.kwds.append(kwargs)
        return self._results.pop(0)


def test_fresh_broken_response_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    fake = _FakeChat([ChatResult("garbage", cached=False, key="fresh")])
    monkeypatch.setattr(analyzer, "chat", fake)

    with pytest.raises(LLMOutputError):
        analyzer._request_design("sys", ["msg"])

    assert fake.calls == 1  # never re-pay for the same input


def test_cached_broken_response_is_refreshed_once(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    fake = _FakeChat(
        [
            ChatResult("garbage", cached=True, key="stale"),
            ChatResult(_VALID, cached=False, key="fresh"),
        ]
    )
    monkeypatch.setattr(analyzer, "chat", fake)

    design = analyzer._request_design("sys", ["msg"])

    assert design.title == "T"
    assert fake.calls == 2
    assert fake.kwds[1].get("refresh") is True


def test_valid_cached_response_is_used_as_is(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    fake = _FakeChat([ChatResult(_VALID, cached=True, key="hit")])
    monkeypatch.setattr(analyzer, "chat", fake)

    design = analyzer._request_design("sys", ["msg"])

    assert design.title == "T"
    assert fake.calls == 1


def test_content_filtered_response_fails_clearly_without_refetch(monkeypatch, tmp_path):
    """A filtered response must be named as such, not as a JSON/truncation problem."""
    monkeypatch.setattr(settings, "output_root", tmp_path)
    fake = _FakeChat(
        [
            ChatResult(
                "The request was rejected because it was considered high risk",
                cached=False,
                key="fresh",
                finish="content_filter",
            )
        ]
    )
    monkeypatch.setattr(analyzer, "chat", fake)

    with pytest.raises(LLMOutputError, match="content-filtered"):
        analyzer._request_design("sys", ["msg"])

    assert fake.calls == 1  # the same input would be filtered again — never re-pay


def test_generator_refreshes_cached_bad_scripts(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    good = json.dumps({"game_manager.gd": "extends Node\nfunc score_changed(_s): pass\n"})
    fake = _FakeChat(
        [
            ChatResult("not json", cached=True, key="stale"),
            ChatResult(good, cached=False, key="fresh"),
        ]
    )
    monkeypatch.setattr(generator, "chat", fake)

    scripts = generator._generate_scripts(analyzer._parse(_VALID))

    assert "game_manager.gd" in scripts
    assert fake.calls == 2
    assert fake.kwds[1].get("refresh") is True


def test_generator_does_not_retry_fresh_bad_scripts(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    fake = _FakeChat([ChatResult("not json", cached=False, key="fresh")])
    monkeypatch.setattr(generator, "chat", fake)

    assert generator._generate_scripts(analyzer._parse(_VALID)) == {}
    assert fake.calls == 1


def test_checkpoint_roundtrip_and_staleness(tmp_path):
    design = analyzer._parse(_VALID)
    analyzer.save_checkpoint(tmp_path, design, "key-1")

    reloaded = analyzer.load_checkpoint(tmp_path, "key-1")
    assert reloaded is not None and reloaded.title == "T"
    # inputs changed → stale checkpoint must be ignored
    assert analyzer.load_checkpoint(tmp_path, "key-2") is None


def test_missing_or_corrupt_checkpoint_is_none(tmp_path):
    assert analyzer.load_checkpoint(tmp_path, "k") is None
    (tmp_path / "design.key").write_text("k", encoding="utf-8")
    (tmp_path / "design.json").write_text("{broken", encoding="utf-8")
    assert analyzer.load_checkpoint(tmp_path, "k") is None
