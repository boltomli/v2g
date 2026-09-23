"""Transcript window slicing in chunked video analysis."""

from v2g.llm import analyzer
from v2g.video.dialogue import TranscriptLine


def _empty_design(title: str) -> analyzer.GameDesign:
    return analyzer.GameDesign(
        title=title, genre="x", summary="s",
        mechanics=[], controls=[], style="", objects=[],
    )


def test_windows_follow_actual_segment_durations_not_chunk_grid(monkeypatch, tmp_path):
    """Windows anchor to cumulative segment durations — size-driven splits and
    the trailing segment never land on the ``chunk_duration`` grid."""
    segs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    monkeypatch.setattr(
        analyzer, "_get_duration", lambda p: 10.0 if p.name == "a.mp4" else 5.0
    )

    seen: list[tuple[str, str | None]] = []

    def fake_analyze(path, *, instruct=None, transcript=None):
        seen.append((path.name, transcript))
        return _empty_design(path.name)

    monkeypatch.setattr(analyzer, "analyze_video", fake_analyze)

    lines = [
        TranscriptLine(3.0, 4.0, "early"),
        TranscriptLine(12.0, 13.0, "late"),
        TranscriptLine(500.0, 501.0, "beyond"),
    ]
    design = analyzer.analyze_video_chunked(segs, transcript=lines)

    assert design is not None
    assert len(seen) == 2
    # windows are [0, 10) and [10, 15) from actual durations
    assert "early" in seen[0][1] and "late" not in seen[0][1]
    assert "late" in seen[1][1] and "early" not in seen[1][1]
    assert "beyond" not in seen[1][1]
