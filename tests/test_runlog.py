"""Tests for per-run project directory + logging (runlog)."""

import logging

from v2g import runlog


def setup_function(_):
    runlog.reset()


def teardown_function(_):
    runlog.reset()


def test_start_run_creates_new_directory_each_time(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog.settings, "output_root", tmp_path)

    first = runlog.start_run("clip.mp4")
    second = runlog.start_run("clip.mp4")

    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.parent == tmp_path == second.parent
    assert "clip" in first.name


def test_start_run_honors_explicit_output_dir(tmp_path):
    target = tmp_path / "my_game"
    got = runlog.start_run("clip.mp4", target)
    assert got == target
    assert target.is_dir()


def test_run_log_file_receives_records(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog.settings, "output_root", tmp_path)
    run_dir = runlog.start_run("movie.mp4")

    logging.getLogger("v2g.testprobe").info("probe-line-123")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = (run_dir / "v2g.log").read_text(encoding="utf-8")
    assert "probe-line-123" in content
    assert "Run started" in content
    assert "Run config" in content


def test_work_dir_is_inside_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog.settings, "output_root", tmp_path)
    run_dir = runlog.start_run("clip.mp4")
    work = runlog.work_dir()
    assert work == run_dir / "work"
    assert work.is_dir()


def test_llm_dump_writes_numbered_files(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog.settings, "output_root", tmp_path)
    run_dir = runlog.start_run("clip.mp4")

    p1 = runlog.llm_dump("analysis_raw", "AAA")
    p2 = runlog.llm_dump("analysis_raw", "BBB")

    assert p1 and p2 and p1 != p2
    assert p1.read_text(encoding="utf-8") == "AAA"
    assert p2.read_text(encoding="utf-8") == "BBB"
    assert p1.parent == run_dir / "llm"


def test_llm_dump_outside_run_returns_none():
    assert runlog.llm_dump("x", "y") is None


def test_slug_from_url_and_path():
    assert runlog._slug("https://ex.com/watch?v=abc") == "watch"
    assert runlog._slug("https://ex.com/video.mp4") == "video"
    assert runlog._slug("C:/videos/My Clip.MP4") == "My_Clip"


def test_console_shows_notice_but_not_info(tmp_path, monkeypatch, capsys):
    """Console stream = NOTICE+: recovery/completion lines visible, progress stays in the file."""
    monkeypatch.setattr(runlog.settings, "output_root", tmp_path)
    runlog.start_run("movie.mp4")

    logger = logging.getLogger("v2g.testprobe")
    logger.info("info-line-file-only")
    logger.log(runlog.NOTICE, "notice-line-console")
    for handler in logging.getLogger().handlers:
        handler.flush()

    err = capsys.readouterr().err
    assert "notice-line-console" in err
    assert "info-line-file-only" not in err
