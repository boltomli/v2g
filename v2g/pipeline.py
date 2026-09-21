"""End-to-end pipeline: video source → frames → LLM analysis → Godot project."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from v2g.llm.analyzer import GameDesign, analyze
from v2g.godot.generator import generate
from v2g.video.extractor import extract

console = Console()


def run(source: str, output_dir: Path | None = None) -> Path:
    """Execute the full pipeline and return the generated project path.

    Args:
        source: Local video file path or URL.
        output_dir: Override output directory (default: projects/<title>/).

    Returns:
        Path to the generated Godot project directory.
    """
    # ── Step 1: Extract frames ────────────────────────────────────────────
    console.print("[bold cyan]▶ Extracting frames...[/]")
    frames = extract(source)
    console.print(f"  Extracted {len(frames)} frames")

    # ── Step 2: Analyze with LLM ──────────────────────────────────────────
    console.print("[bold cyan]▶ Analyzing video with LLM...[/]")
    design: GameDesign = analyze(frames)
    console.print(Panel(
        f"[bold]{design.title}[/]\n"
        f"Genre: {design.genre}\n"
        f"Mechanics: {', '.join(design.mechanics)}\n"
        f"Objects: {len(design.objects)}  |  Levels: {len(design.levels)}\n\n"
        f"{design.summary}",
        title="🎮 Game Design",
        border_style="green",
    ))

    # ── Step 3: Generate Godot project ────────────────────────────────────
    console.print("[bold cyan]▶ Generating Godot project...[/]")
    project_path = generate(design, output_dir)
    console.print(f"  [bold green]✓[/] Project created at [link=file://{project_path}]{project_path}[/link]")

    return project_path
