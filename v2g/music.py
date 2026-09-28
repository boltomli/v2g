"""Background music: one loopable mp3, two backends behind one switch.

- ``V2G_MUSIC_PROVIDER=api`` (default): **OpenAI-format chat audio** on the
  trunk endpoint (``V2G_LLM_BASE_URL`` / ``V2G_LLM_API_KEY``) with the model
  from ``V2G_MUSIC_MODEL`` — which backend serves the model is the user's
  routing concern (see :mod:`v2g.audio_api`).
- ``V2G_MUSIC_PROVIDER=acestep``: **local备选** — a running ACE-Step 1.5
  ``acestep-api`` server (its own ``release_task`` / ``query_result`` REST
  protocol) at ``V2G_MUSIC_ACESTEP_URL``. The server is the user's to start
  and stop; v2g only speaks to it.

``V2G_MUSIC_MODEL`` unset means no music at all.

Contract (same as every audio layer): a BGM failure never fails a run — the
project simply ships without music and says so.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

from v2g import audio_api, cache, runlog
from v2g.audio_api import AudioAPIError
from v2g.config import settings
from v2g.llm.analyzer import GameDesign

log = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0  # seconds between acestep /query_result calls
_TASK_TIMEOUT = 600.0  # seconds for one acestep task (first run may load models)


def music_prompt(design: GameDesign, instruct: str | None) -> str:
    """One caption describing the game's BGM: atmosphere + style + theme.

    Closes with an instrumental mandate — the VN speaks over this track, so
    vocals would fight the voice-over.
    """
    base = " ".join(p for p in (design.atmosphere.strip(), design.style.strip()) if p)
    if not base:
        base = "ambient background music"
    theme = (instruct or "").strip()
    themed = f", themed to {theme}" if theme else ""
    return (
        f"{base}{themed}, instrumental background music for a visual novel, "
        "no vocals, no lyrics, no singing"
    )


# ── acestep local REST (the `acestep` provider) ─────────────────────────────


def _post(base: str, route: str, payload: dict) -> object:
    """POST JSON to the local server; unwrap ``{data, code, error}``."""
    req = urllib.request.Request(
        f"{base.rstrip('/')}{route}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise AudioAPIError(f"{route} → HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise AudioAPIError(
            f"no ACE-Step server at {base} ({e.reason}) — start acestep-api "
            "or use V2G_MUSIC_PROVIDER=api"
        ) from e
    try:
        body = json.loads(raw)
    except ValueError as e:
        raise AudioAPIError(f"non-JSON response from {base}") from e
    if not isinstance(body, dict) or body.get("code") != 200:
        err = body.get("error") if isinstance(body, dict) else None
        raise AudioAPIError(f"{route}: {err or 'unexpected response'}")
    return body.get("data")


def _acestep_track(base: str, model: str, prompt: str) -> bytes:
    """One track from a running local acestep-api: submit → poll → download."""
    payload: dict = {
        "prompt": prompt,
        "lyrics": "",  # empty lyrics = instrumental
        "audio_duration": settings.music_duration,
        "audio_format": "mp3",
        "batch_size": 1,
        "use_random_seed": True,
    }
    if model:
        payload["model"] = model

    data = _post(base, "/release_task", payload)
    task_id = data.get("task_id") if isinstance(data, dict) else None
    if not task_id:
        raise AudioAPIError(f"release_task returned no task_id: {str(data)[:160]}")

    entry = _poll(base, str(task_id))
    try:
        results = json.loads(entry.get("result") or "[]")
    except (TypeError, ValueError) as e:
        raise AudioAPIError(f"unparsable task result: {e}") from e
    if not isinstance(results, list) or not results:
        raise AudioAPIError("task succeeded but produced no files")
    chosen = next((r for r in results if isinstance(r, dict) and r.get("status") == 1), results[0])
    file_url = chosen.get("file") if isinstance(chosen, dict) else None
    if not file_url:
        raise AudioAPIError("task result carries no audio file")
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base.rstrip('/')}{file_url}"), timeout=60.0
        ) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError) as e:
        raise AudioAPIError(f"audio download failed: {e}") from e


def _poll(base: str, task_id: str) -> dict:
    """Poll /query_result until the task succeeds; raise on failure/timeout."""
    end = time.monotonic() + _TASK_TIMEOUT
    while True:
        time.sleep(_POLL_INTERVAL)
        data = _post(base, "/query_result", {"task_id_list": [task_id]})
        entry = None
        if isinstance(data, list):
            entry = next(
                (e for e in data if isinstance(e, dict) and e.get("task_id") == task_id),
                data[0] if data else None,
            )
        if not isinstance(entry, dict):
            raise AudioAPIError("query_result returned no entry for the task")
        try:
            status = int(entry.get("status", 0))
        except (TypeError, ValueError):
            status = 0
        if status == 1:
            return entry
        if status == 2:
            raise AudioAPIError(f"generation task failed: {str(entry)[:200]}")
        if time.monotonic() > end:
            raise AudioAPIError(f"generation timed out after {_TASK_TIMEOUT:.0f}s")


# ── Pipeline entry point ─────────────────────────────────────────────────────


def generate_music(
    design: GameDesign,
    instruct: str | None,
    run_dir: Path,
) -> str | None:
    """Stage 3 background music: one loopable mp3 inside the project.

    Returns the ``res://`` path for the VN runtime, or None — model unset,
    provider misconfigured, or generation failed. Never raises.
    """
    provider = settings.music_provider.strip().lower() or "api"
    if provider not in ("api", "acestep"):
        log.log(
            runlog.NOTICE,
            "BGM: unknown provider %r (expected api|acestep) — skipped",
            provider,
        )
        return None
    model = settings.music_model.strip()
    if not model:
        log.log(runlog.NOTICE, "BGM: off (V2G_MUSIC_MODEL unset)")
        return None

    prompt = f"{music_prompt(design, instruct)}, {settings.music_duration} seconds"
    material = {
        "provider": provider,
        "url": settings.music_acestep_url if provider == "acestep" else "",
        "model": model,
        "prompt": prompt,
        "duration": settings.music_duration,
        "format": "mp3",
    }
    try:
        data = cache.audio_find("music", material, "mp3")
        cached = data is not None
        if data is None:
            if provider == "acestep":
                data = _acestep_track(settings.music_acestep_url, model, prompt)
            else:
                data = audio_api.chat_audio(model, prompt)
            cache.audio_put("music", material, data, "mp3")
        target = run_dir / "assets" / "bgm" / "bgm.mp3"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        where = settings.music_acestep_url if provider == "acestep" else settings.llm_base_url
        log.log(
            runlog.NOTICE,
            "BGM: 1 track → %s (%s, %d bytes)",
            target,
            "cache hit" if cached else f"generated via {where}",
            len(data),
        )
        return f"res://assets/bgm/{target.name}"
    except AudioAPIError as e:
        log.log(runlog.NOTICE, "BGM: skipped — %s", e)
        return None
    except Exception:
        log.exception("BGM generation failed — project ships without music")
        return None
