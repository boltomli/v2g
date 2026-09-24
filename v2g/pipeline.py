"""End-to-end pipeline: video source → LLM analysis → Godot project."""

import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from v2g import runlog
from v2g.config import settings
from v2g.godot.generator import generate
from v2g.llm.analyzer import (
    GameDesign,
    analysis_key,
    analyze,
    analyze_video_chunked,
    load_checkpoint,
    save_checkpoint,
)
from v2g.video.dialogue import TranscriptLine, extract_dialogue, format_transcript
from v2g.video.extractor import extract, prepare_video_for_upload, resolve_source, split_video

console = Console()
log = logging.getLogger(__name__)


def _print_design(design: GameDesign, instruct: str | None = None) -> None:
    physics = f"\nPhysics: {design.physics}" if design.physics else ""
    progression = f"\nProgression: {design.progression}" if design.progression else ""
    instruct_note = f"\nStyle: {instruct}" if instruct else ""
    narrative_brief = ""
    if design.narrative:
        # Show first 200 chars of narrative
        n = design.narrative
        narrative_brief = f"\nNarrative: {n[:200]}{'…' if len(n) > 200 else ''}"
    console.print(Panel(
        f"[bold]{design.title}[/]\n"
        f"Genre: {design.genre}\n"
        f"Mechanics: {', '.join(design.mechanics)}\n"
        f"Objects: {len(design.objects)}  |  Characters: {len(design.characters)}  |  Scenes: {len(design.scenes)}"
        f"{physics}{progression}{instruct_note}{narrative_brief}\n\n"
        f"{design.summary}",
        title="🎮 Game Design",
        border_style="green",
    ))


def _analyze(
    source_video: Path,
    work: Path,
    transcript_lines: list[TranscriptLine],
    *,
    detailed: bool,
    instruct: str | None,
) -> GameDesign:
    """Analyze layer: prepared source → GameDesign (checkpoint handled by caller)."""
    if detailed:
        # ── Detailed mode: send full video to LLM ────────────────────────
        from v2g.video.extractor import _get_duration

        duration = _get_duration(source_video)
        chunk_dur = settings.chunk_duration

        if duration > chunk_dur + 30:
            # Long video: split into chunks and analyze each separately
            n_segments = int(duration / chunk_dur) + 1
            console.print(f"[bold cyan]▶ Long video ({duration:.0f}s) — splitting into ~{n_segments} segments...[/]")
            segments = split_video(source_video, work, segment_duration=chunk_dur)
            console.print(f"  Split into {len(segments)} segments")

            console.print("[bold cyan]▶ Analyzing video segments with LLM (detailed)...[/]")
            design = analyze_video_chunked(
                segments, instruct=instruct, transcript=transcript_lines
            )
        else:
            # Short enough: one analysis; split further only if a piece still
            # exceeds the size cap after its single compression pass.
            console.print("[bold cyan]▶ Preparing video for detailed analysis...[/]")
            uploads = prepare_video_for_upload(source_video, work, settings.video_max_mb)
            total_mb = sum(p.stat().st_size for p in uploads) / (1024 * 1024)
            console.print(
                f"  Video ready ({len(uploads)} file(s), {total_mb:.1f} MB)"
            )

            console.print("[bold cyan]▶ Analyzing video with LLM (detailed)...[/]")
            design = analyze_video_chunked(
                uploads, instruct=instruct, transcript=transcript_lines
            )
    else:
        # ── Fast mode: extract keyframes ──────────────────────────────────
        console.print("[bold cyan]▶ Extracting frames...[/]")
        frames = extract(source_video, work)
        console.print(f"  Extracted {len(frames)} frames")

        console.print("[bold cyan]▶ Analyzing video with LLM...[/]")
        design = analyze(
            frames,
            instruct=instruct,
            transcript=format_transcript(transcript_lines) or None,
        )
    return design


def run(source: str, output_dir: Path | None = None, *, detailed: bool = False, instruct: str | None = None) -> Path:
    """Execute the full pipeline and return the generated project path.

    Args:
        source: Local video file path or URL.
        output_dir: Override run directory (default: projects/<timestamp>_<source>/).
        detailed: If True, send full video to LLM for deep analysis.
        instruct: Optional style instruction (e.g. "medieval theme", "vampire style").

    Returns:
        Path to the generated Godot project directory.
    """
    # Every run gets its own project directory FIRST — logs, work files, raw
    # LLM responses and the generated game all live under it.
    run_dir = runlog.start_run(source, output_dir)
    console.print(f"[bold cyan]▶ Run directory:[/] {run_dir}")
    work = runlog.work_dir()

    # Resolve video source once (URL downloads are cached across runs)
    source_video = resolve_source(source, work)

    # ── Source-language dialogue: subtitles are the only authoritative source ──
    transcript_lines: list[TranscriptLine] = extract_dialogue(source_video)
    if transcript_lines:
        console.print(f"[bold cyan]▶ Transcript:[/] {len(transcript_lines)} subtitle lines from source video")
        log.info("Transcript: %d subtitle lines", len(transcript_lines))
    else:
        console.print("[yellow]▶ No subtitles found — source-language lines will be left empty[/]")
        log.info("Transcript: none found — source-language lines will be left empty")

    # ── Analyze layer: checkpoint first, LLM only on a miss ─────────────────
    key = analysis_key(
        source_video, transcript_lines, detailed=detailed, instruct=instruct
    )
    design = load_checkpoint(run_dir, key)
    if design is not None:
        console.print("[bold cyan]▶ Analysis checkpoint:[/] design.json reused, LLM skipped")
        log.info("Analysis checkpoint hit (key=%.12s) — LLM skipped", key)
    else:
        design = _analyze(
            source_video, work, transcript_lines, detailed=detailed, instruct=instruct
        )
        save_checkpoint(run_dir, design, key)
        log.info("Analysis complete; checkpoint saved (key=%.12s)", key)

    _print_design(design, instruct)
    log.info(
        "Design analyzed: title=%r genre=%r characters=%d objects=%d scenes=%d dialogue=%d",
        design.title, design.genre, len(design.characters), len(design.objects),
        len(design.scenes), len(design.dialogue_samples),
    )

    # ── Generate Godot project (into the run directory) ──────────────────────
    console.print("[bold cyan]▶ Generating Godot project...[/]")
    project_path = generate(design, run_dir, video_path=source_video)
    console.print(f"  [bold green]✓[/] Project created at [link=file://{project_path}]{project_path}[/link]")
    log.log(runlog.NOTICE, "Project created at %s", project_path)

    return project_path
