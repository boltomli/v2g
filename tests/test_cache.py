"""Tests for the cross-run LLM response cache."""

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
