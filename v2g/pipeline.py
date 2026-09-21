"""End-to-end pipeline: video source → LLM analysis → Godot project."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from v2g.config import settings
from v2g.godot.generator import generate
from v2g.llm.analyzer import GameDesign, analyze, analyze_video, analyze_video_chunked
from v2g.video.extractor import prepare_video_for_upload, resolve_source, split_video

console = Console()


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


def run(source: str, output_dir: Path | None = None, *, detailed: bool = False, instruct: str | None = None) -> Path:
    """Execute the full pipeline and return the generated project path.

    Args:
        source: Local video file path or URL.
        output_dir: Override output directory (default: projects/<title>/).
        detailed: If True, send full video to LLM for deep analysis.
        instruct: Optional style instruction (e.g. "medieval theme", "vampire style").

    Returns:
        Path to the generated Godot project directory.
    """
    # Resolve video source once (avoid double download for URLs)
    source_video, _tmp = resolve_source(source)

    if detailed:
        # ── Detailed mode: send full video to LLM ────────────────────────
        from v2g.video.extractor import _get_duration

        duration = _get_duration(source_video)
        chunk_dur = settings.chunk_duration

        if duration > chunk_dur + 30:
            # Long video: split into chunks and analyze each separately
            n_segments = int(duration / chunk_dur) + 1
            console.print(f"[bold cyan]▶ Long video ({duration:.0f}s) — splitting into ~{n_segments} segments...[/]")
            segments = split_video(source_video, _tmp, segment_duration=chunk_dur)
            console.print(f"  Split into {len(segments)} segments")

            console.print("[bold cyan]▶ Analyzing video segments with LLM (detailed)...[/]")
            design = analyze_video_chunked(segments, instruct=instruct)
        else:
            # Short enough: single upload
            console.print("[bold cyan]▶ Preparing video for detailed analysis...[/]")
            upload_path = prepare_video_for_upload(source_video, _tmp, settings.video_max_mb)
            size_mb = upload_path.stat().st_size / (1024 * 1024)
            console.print(f"  Video ready ({size_mb:.1f} MB)")

            console.print("[bold cyan]▶ Analyzing video with LLM (detailed)...[/]")
            design = analyze_video(upload_path, instruct=instruct)
    else:
        # ── Fast mode: extract keyframes ──────────────────────────────────
        from v2g.video.extractor import extract as smart_extract

        console.print("[bold cyan]▶ Extracting frames...[/]")
        frames = smart_extract(source)
        console.print(f"  Extracted {len(frames)} frames")

        console.print("[bold cyan]▶ Analyzing video with LLM...[/]")
        design = analyze(frames, instruct=instruct)

    _print_design(design, instruct)

    # ── Generate Godot project ────────────────────────────────────────────
    console.print("[bold cyan]▶ Generating Godot project...[/]")
    project_path = generate(design, output_dir, video_path=source_video)
    console.print(f"  [bold green]✓[/] Project created at [link=file://{project_path}]{project_path}[/link]")

    return project_path
