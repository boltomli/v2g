"""CLI entry point for v2g: video → Godot game pipeline."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from v2g import __version__

console = Console()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="v2g",
        description="Transform a video into a playable Godot game using LLM analysis.",
    )
    parser.add_argument("source", help="Video file path or URL (YouTube, direct link, etc.)")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for the Godot project (default: projects/<title>/)",
    )
    parser.add_argument(
        "-d", "--detail",
        action="store_true",
        default=False,
        help="Detailed mode: send full video to LLM for deep analysis (requires video-capable model)",
    )
    parser.add_argument("--version", action="version", version=f"v2g {__version__}")
    args = parser.parse_args(argv)

    console.print(f"[bold]v2g[/] v{__version__} — Video to Godot Game\n")

    try:
        from v2g.pipeline import run
        project_path = run(args.source, args.output, detailed=args.detail)
        console.print(f"\n[bold green]Done![/] Open the project in Godot 4.x:\n  godot --editor {project_path}")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)


if __name__ == "__main__":
    main()
