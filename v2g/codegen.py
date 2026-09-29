"""Run LLM-written audio synthesis scripts: source → isolated python → wav → mp3.

Shared by the background-music ``llm`` provider and the sound-effects layer:
the chat model writes one self-contained Python script, this module runs it
with ``python -I`` (isolated: env/user hooks ignored, venv site kept) under a
timeout inside the caller's work directory — the source is kept next to the
output for inspection — and normalizes the WAV to MP3 through ffmpeg.

Failures raise :class:`v2g.audio_api.AudioAPIError` so every caller keeps its
own contract: a BGM failure skips the track, a bad sound-effect clip drops
that clip, neither ever fails the run.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

from v2g.audio_api import AudioAPIError


def libs_note() -> str:
    """What a generated script may import: numpy when installed, stdlib else."""
    if importlib.util.find_spec("numpy") is not None:
        return "numpy is installed — use it for synthesis"
    return "STANDARD LIBRARY ONLY (math/array/wave/struct/random) — numpy is NOT installed"


def extract_code(text: str) -> str:
    """Raw source from a model answer; strips markdown fences when present."""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (fenced[-1] if fenced else text).strip()


def render(code: str, work: Path | str, *, stem: str, timeout: float) -> bytes:
    """Run *code* (``sys.argv[1]`` = output WAV path) → WAV → MP3 bytes.

    The source lands at ``<work>/<stem>.py`` for inspection; the WAV and MP3
    stay beside it. Any failure — crash, non-zero exit, empty/missing WAV,
    ffmpeg error — raises :class:`AudioAPIError`.
    """
    # Absolute: the script runs with cwd=work and opens argv[1] directly, so
    # a relative path would resolve against the work dir twice.
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{stem}.py").write_text(code, encoding="utf-8")

    wav = work / f"{stem}.wav"
    wav.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code, str(wav)],
            cwd=str(work),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise AudioAPIError(f"synth script could not run: {e}") from e
    if proc.returncode != 0 or not wav.is_file() or wav.stat().st_size == 0:
        # stdout first: a wrong-exit or wrong output path explains itself
        # there; stderr only carries a traceback for a real crash.
        tail = (
            proc.stdout.decode("utf-8", errors="replace")[-300:]
            or proc.stderr.decode("utf-8", errors="replace")[-300:]
            or "(no output)"
        )
        raise AudioAPIError(
            f"synth script failed (exit {proc.returncode}): {tail.replace(chr(10), ' ')}"
        )

    mp3 = work / f"{stem}.mp3"
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
