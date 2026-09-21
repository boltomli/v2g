"""End-to-end pipeline: video source → LLM analysis → Godot project."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from v2g.config import settings
from v2g.llm.analyzer import GameDesign, analyze, analyze_video
from v2g.godot.generator import generate
from v2g.video.extractor import extract, prepare_video_for_upload, resolve_source

console = Console()


def _print_design(design: GameDesign) -> None:
    physics = f"\nPhysics: {design.physics}" if design.physics else ""
    progression = f"\nProgression: {design.progression}" if design.progression else ""
    console.print(Panel(
        f"[bold]{design.title}[/]\n"
        f"Genre: {design.genre}\n"
        f"Mechanics: {', '.join(design.mechanics)}\n"
        f"Objects: {len(design.objects)}  |  Levels: {len(design.levels)}"
        f"{physics}{progression}\n\n"
        f"{design.summary}",
        title="🎮 Game Design",
        border_style="green",
    ))


def run(source: str, output_dir: Path | None = None, *, detailed: bool = False) -> Path:
    """Execute the full pipeline and return the generated project path.

    Args:
        source: Local video file path or URL.
        output_dir: Override output directory (default: projects/<title>/).
        detailed: If True, send full video to LLM for deep analysis.

    Returns:
        Path to the generated Godot project directory.
    """
    if detailed:
        # ── Detailed mode: send full video ────────────────────────────────
        console.print("[bold cyan]▶ Preparing video for detailed analysis...[/]")
        video_path, tmp = resolve_source(source)
        upload_path = prepare_video_for_upload(video_path, tmp, settings.video_max_mb)
        size_mb = upload_path.stat().st_size / (1024 * 1024)
        console.print(f"  Video ready ({size_mb:.1f} MB)")

        console.print("[bold cyan]▶ Analyzing video with LLM (detailed)...[/]")
        design = analyze_video(upload_path)
    else:
        # ── Fast mode: extract keyframes ──────────────────────────────────
        console.print("[bold cyan]▶ Extracting frames...[/]")
        frames = extract(source)
        console.print(f"  Extracted {len(frames)} frames")

        console.print("[bold cyan]▶ Analyzing video with LLM...[/]")
        design = analyze(frames)

    _print_design(design)

    # ── Generate Godot project ────────────────────────────────────────────
    console.print("[bold cyan]▶ Generating Godot project...[/]")
    project_path = generate(design, output_dir)
    console.print(f"  [bold green]✓[/] Project created at [link=file://{project_path}]{project_path}[/link]")

    return project_path
