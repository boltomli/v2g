"""Background music: one loopable mp3, three backends behind one switch.

- ``V2G_MUSIC_PROVIDER=api`` (default): **OpenAI-format chat audio** on the
  trunk endpoint (``V2G_LLM_BASE_URL`` / ``V2G_LLM_API_KEY``) with the model
  from ``V2G_MUSIC_MODEL`` — which backend serves the model is the user's
  routing concern (see :mod:`v2g.audio_api`).
- ``V2G_MUSIC_PROVIDER=llm``: the **text model writes the music itself** —
  one cached chat call produces a self-contained Python synthesis script
  (``V2G_MUSIC_MODEL`` = the code-writing model, e.g. the trunk LLM), run
  isolated with a timeout and converted through ffmpeg. Works on any trunk
  that has no music model at all. Note: this executes model-written code —
  trust the endpoint you point at.
- ``V2G_MUSIC_PROVIDER=acestep``: **local alternative** — a running ACE-Step
  1.5 ``acestep-api`` server (its own ``release_task`` / ``query_result``
  REST protocol) at ``V2G_MUSIC_ACESTEP_URL``. The server is the user's to
  start and stop; v2g only speaks to it.

``V2G_MUSIC_MODEL`` unset means no music at all.

Contract (same as every audio layer): a BGM failure never fails a run — the
project simply ships without music and says so.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from v2g import audio_api, cache, runlog
from v2g.audio_api import AudioAPIError
from v2g.config import settings
from v2g.llm.analyzer import GameDesign
from v2g.llm.client import chat

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


# ── LLM codegen (the `llm` provider) ────────────────────────────────────────


def _codegen_system() -> str:
    """Spec for the text model: one self-contained Python script → WAV."""
    has_numpy = importlib.util.find_spec("numpy") is not None
    libs = (
        "numpy is installed — use it for synthesis"
        if has_numpy
        else "STANDARD LIBRARY ONLY (math/array/wave/struct/random) — numpy is NOT installed"
    )
    return f"""\
You are an expert Python audio programmer. Write ONE self-contained Python
script that renders instrumental background music for a visual novel and
writes it to disk.

Contract:
- The script receives the OUTPUT WAV PATH as sys.argv[1] and writes exactly
  {settings.music_duration} seconds of 16-bit PCM stereo audio at 44100 Hz there.
- Runtime: Python {sys.version_info.major}.{sys.version_info.minor}; {libs}.
- Deterministic: fixed random seed. No network, no reading of any file, no
  input(), no GUI, no packages beyond what is stated above.
- Musically: a chord pad, a simple melody and a light rhythm fitting the
  style request; instrumental only (no vocals, no speech, no singing).
- Loop-friendly: the last 0.1 s must return to the opening level so the wrap
  does not click.

Output ONLY the raw Python source — no markdown fences, no commentary.
"""


def _extract_code(text: str) -> str:
    """Raw source from a model answer; strips markdown fences when present."""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (fenced[-1] if fenced else text).strip()


def _llm_track(prompt: str, run_dir: Path) -> bytes:
    """Text model writes a synthesis script → run it isolated → wav → mp3.

    The script runs as ``python -I`` (isolated: env/user hooks ignored, venv
    site kept) with a timeout inside ``<run>/work/bgm_llm/``; the source is
    kept there for inspection. Any failure becomes :class:`AudioAPIError`, so
    the layer degrades to "no music" instead of failing the run.
    """
    work = (run_dir / "work" / "bgm_llm").resolve()
    work.mkdir(parents=True, exist_ok=True)
    try:
        res = chat(_codegen_system(), [prompt], temperature=0.4)
    except Exception as e:
        raise AudioAPIError(f"music codegen chat failed: {e}") from e
    code = _extract_code(res.text)
    if not code:
        raise AudioAPIError("music codegen returned no code")
    (work / "bgm_llm.py").write_text(code, encoding="utf-8")

    wav = work / "bgm.wav"
    wav.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code, str(wav)],
            cwd=str(work),
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise AudioAPIError(f"music codegen script could not run: {e}") from e
    if proc.returncode != 0 or not wav.is_file() or wav.stat().st_size == 0:
        # stdout first: a wrong-exit or wrong output path explains itself
        # there; stderr only carries a traceback for a real crash.
        tail = (
            proc.stdout.decode("utf-8", errors="replace")[-300:]
            or proc.stderr.decode("utf-8", errors="replace")[-300:]
            or "(no output)"
        )
        raise AudioAPIError(
            f"music codegen script failed (exit {proc.returncode}): {tail.replace(chr(10), ' ')}"
        )

    mp3 = work / "bgm.mp3"
    mp3.unlink(missing_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-f", "mp3", "-q:a", "4", str(mp3)],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not mp3.is_file() or mp3.stat().st_size == 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-400:].replace("\n", " ")
        raise AudioAPIError(f"ffmpeg wav→mp3 failed: {err}")
    return mp3.read_bytes()


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
    if provider not in ("api", "acestep", "llm"):
        log.log(
            runlog.NOTICE,
            "BGM: unknown provider %r (expected api|acestep|llm) — skipped",
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
        found = cache.audio_find("music", material, "mp3")
        cached = found is not None
        if found is not None:
            data = found.read_bytes()
        elif provider == "acestep":
            data = _acestep_track(settings.music_acestep_url, model, prompt)
            cache.audio_put("music", material, data, "mp3")
        elif provider == "llm":
            data = _llm_track(prompt, run_dir)
            cache.audio_put("music", material, data, "mp3")
        else:
            data = audio_api.chat_audio(model, prompt)
            cache.audio_put("music", material, data, "mp3")
        target = run_dir / "assets" / "bgm" / "bgm.mp3"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if cached:
            via = "cache hit"
        elif provider == "llm":
            via = "generated by LLM codegen"
        elif provider == "acestep":
            via = f"generated via {settings.music_acestep_url}"
        else:
            via = f"generated via {settings.llm_base_url}"
        log.log(
            runlog.NOTICE,
            "BGM: 1 track → %s (%s, %d bytes)",
            target,
            via,
            len(data),
        )
        return f"res://assets/bgm/{target.name}"
    except AudioAPIError as e:
        log.log(runlog.NOTICE, "BGM: skipped — %s", e)
        return None
    except Exception:
        log.exception("BGM generation failed — project ships without music")
        return None
