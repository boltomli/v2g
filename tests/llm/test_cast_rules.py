"""Stage 1 casts from the pictures: the transcript is dialogue, never a cast list."""

from v2g.llm import analyzer


def test_both_system_prompts_ground_the_cast_in_the_pictures():
    """Fast and detail mode must share the same rule: a name in the subtitles
    who never shows up on screen is not a character, and one face stays one
    entry however many names the dialogue gives it."""
    for prompt in (analyzer._SYSTEM_FRAMES, analyzer._SYSTEM_VIDEO):
        assert "CHARACTER GROUNDING" in prompt
        assert "APPEAR ON SCREEN" in prompt
        assert "not a cast list" in prompt
        assert "alias" in prompt


def test_the_transcript_header_says_dialogue_not_characters():
    note = analyzer._transcript_note("[  0.0s] MIYAZAKI: she is gone")

    assert "DIALOGUE ONLY" in note
    assert "not a cast list" in note
    assert "MIYAZAKI" in note  # the transcript itself still flows through


def test_the_schema_keeps_unseen_names_out_of_characters():
    assert "ONLY someone actually visible on screen" in analyzer._JSON_SCHEMA
    assert "off-screen voice with NO character entry" in analyzer._JSON_SCHEMA
