"""chat() serves repeats from cache; only misses (and refreshes) hit the API."""

from types import SimpleNamespace

from v2g.config import settings
from v2g.llm import client


def _install_client(monkeypatch, responses):
    """Fake OpenAI client whose create() pops canned responses and counts calls."""
    counter = {"n": 0}

    def _create(**_kw):
        counter["n"] += 1
        return next(responses)

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    monkeypatch.setattr(client, "_get_client", lambda: fake)
    return counter


def _resp(text, finish="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(content=text),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )


def test_repeat_request_is_served_from_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    counter = _install_client(monkeypatch, iter([_resp("pong")]))

    first = client.chat("sys", ["hi"])
    second = client.chat("sys", ["hi"])

    assert counter["n"] == 1  # second call made no API request
    assert first.cached is False
    assert second.cached is True
    assert second.text == "pong"
    assert second.key == first.key


def test_refresh_bypasses_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    counter = _install_client(monkeypatch, iter([_resp("one"), _resp("two")]))

    client.chat("sys", ["hi"])
    refreshed = client.chat("sys", ["hi"], refresh=True)

    assert counter["n"] == 2
    assert refreshed.cached is False
    assert refreshed.text == "two"


def test_truncated_response_is_never_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    counter = _install_client(
        monkeypatch, iter([_resp("partial", finish="length"), _resp("full")])
    )

    first = client.chat("sys", ["hi"])
    second = client.chat("sys", ["hi"])

    assert first.text == "partial"
    assert counter["n"] == 2  # truncated answer was not stored
    assert second.text == "full"


def test_content_filtered_response_is_never_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_root", tmp_path)
    counter = _install_client(
        monkeypatch, iter([_resp("blocked", finish="content_filter"), _resp("clean")])
    )

    first = client.chat("sys", ["hi"])
    second = client.chat("sys", ["hi"])

    assert first.text == "blocked"
    assert counter["n"] == 2  # content-filtered answer was not stored
    assert second.text == "clean"
