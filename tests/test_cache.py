"""Tests for the cross-run caches (LLM responses + media artifacts)."""

import pytest

from v2g import cache
from v2g.config import settings


def test_put_get_invalidate_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    assert cache.get("k1") is None
    cache.put("k1", "hello")
    assert cache.get("k1") == "hello"
    cache.invalidate("k1")
    assert cache.get("k1") is None


def test_disabled_cache_ignores_reads_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    monkeypatch.setattr(settings, "llm_cache", False)
    cache.put("k", "x")
    assert cache.get("k") is None


def test_corrupt_entry_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    d = tmp_path / ".v2g_cache"
    d.mkdir()
    (d / "bad.json").write_text("{not json", encoding="utf-8")
    assert cache.get("bad") is None


def _key(parts, **over):
    kw = {"model": "m", "system": "sys", "parts": parts, "max_tokens": 100, "temperature": 0.3}
    kw.update(over)
    return cache.key_for(**kw)


def test_key_is_content_addressed_across_run_dirs(tmp_path):
    a = tmp_path / "run-1" / "frames" / "frame.png"
    b = tmp_path / "run-2" / "frames" / "frame.png"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"same-bytes")
    b.write_bytes(b"same-bytes")
    # identical content at different run paths → same key (cache survives reruns)
    assert _key([a]) == _key([b])
    # any input change → different key
    assert _key([a]) != _key([a], temperature=0.4)
    assert _key([a]) != _key([a], system="other-prompt")
    assert _key([a]) != _key([a, "extra text part"])
    b.write_bytes(b"other-bytes")
    assert _key([a]) != _key([b])


def test_media_entry_is_content_addressed_and_respects_toggle(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    work = tmp_path / "work"

    a = cache.media_entry("dl", {"url": "u"}, fallback=work)
    assert a == cache.media_entry("dl", {"url": "u"}, fallback=work)
    assert a != cache.media_entry("dl", {"url": "v"}, fallback=work)
    assert a.parent == tmp_path / ".v2g_cache" / "media"

    monkeypatch.setattr(settings, "media_cache", False)
    disabled = cache.media_entry("dl", {"url": "u"}, fallback=work)
    assert disabled.parent == work
    assert disabled.name == a.name  # same key, just relocated into the run


def test_media_stage_commits_only_on_success(tmp_path):
    entry = tmp_path / "entry"

    with cache.media_stage(entry) as stage:
        (stage / "artifact.bin").write_bytes(b"data")
    assert (entry / "artifact.bin").read_bytes() == b"data"
    assert not (tmp_path / "entry.staging").exists()

    failing = tmp_path / "failing"
    with pytest.raises(RuntimeError, match="build failed"):
        with cache.media_stage(failing) as stage:
            (stage / "artifact.bin").write_bytes(b"half")
            raise RuntimeError("build failed")
    assert not failing.exists()
    assert not (tmp_path / "failing.staging").exists()


def test_media_find_returns_probe_result_and_drops_unusable_entry(tmp_path):
    entry = tmp_path / "entry"
    entry.mkdir()

    def probe(directory):
        candidate = directory / "artifact.bin"
        return candidate if candidate.is_file() else None

    assert cache.media_find(entry, probe) is None
    assert not entry.exists()  # unusable entry is removed for a rebuild
    assert cache.media_find(tmp_path / "missing", probe) is None

    entry.mkdir()
    (entry / "artifact.bin").write_bytes(b"data")
    assert cache.media_find(entry, probe) == entry / "artifact.bin"
