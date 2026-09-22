"""Shared error types for LLM output handling."""


class LLMOutputError(ValueError):
    """The LLM returned output that could not be parsed into the expected JSON.

    Subclasses ValueError so existing per-segment recovery in chunked
    analysis (which catches ValueError) keeps working.
    """
