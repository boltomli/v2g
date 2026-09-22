from pathlib import Path

from v2g.video.dialogue import TranscriptLine, extract_dialogue, format_transcript, parse_srt

_SRT = """\
1
00:00:01,000 --> 00:00:02,500
Primera línea

2
00:00:03,000 --> 00:00:04,000
Línea multilínea
segunda parte

garbage index
00:01:05.500 --> 00:01:07.250
Tercera línea
"""


def test_parse_srt_handles_crlf_multiline_and_vtt_timestamps():
    lines = parse_srt(_SRT.replace("\n", "\r\n"))

    assert [ln.text for ln in lines] == [
        "Primera línea",
        "Línea multilínea segunda parte",
        "Tercera línea",
    ]
    assert lines[0].start == 1.0
    assert lines[0].end == 2.5
    assert lines[2].start == 65.5
    assert lines[2].end == 67.25


def test_extract_dialogue_prefers_srt_sidecar_without_touching_ffmpeg(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    (tmp_path / "clip.srt").write_text(_SRT, encoding="utf-8")

    lines = extract_dialogue(video)

    assert len(lines) == 3
    assert lines[0].text == "Primera línea"


def test_extract_dialogue_returns_empty_when_no_subtitles(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")

    assert extract_dialogue(video) == []


def test_format_transcript_windows_by_time():
    lines = [
        TranscriptLine(1.0, 2.0, "a"),
        TranscriptLine(11.0, 12.0, "b"),
        TranscriptLine(21.0, 22.0, "c"),
    ]

    window = format_transcript(lines, start=10.0, end=20.0)

    assert window == "[  11.0s] b"
    assert "a" not in window and "c" not in window


def test_extract_dialogue_finds_language_tagged_sidecar(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    (tmp_path / "clip.en.srt").write_text(_SRT, encoding="utf-8")

    lines = extract_dialogue(video)

    assert len(lines) == 3
    assert lines[0].text == "Primera línea"


def test_extract_dialogue_prefers_exact_sidecar_over_language_variant(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    (tmp_path / "clip.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nExact wins\n", encoding="utf-8"
    )
    (tmp_path / "clip.en.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nTagged loses\n", encoding="utf-8"
    )

    lines = extract_dialogue(video)

    assert [ln.text for ln in lines] == ["Exact wins"]


def test_extract_dialogue_ignores_non_subtitle_sidecars(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    (tmp_path / "clip.danmaku.xml").write_text("<i></i>", encoding="utf-8")

    assert extract_dialogue(video) == []
