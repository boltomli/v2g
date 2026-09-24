"""Best-effort repair of truncated/malformed LLM JSON — shared by parse layers.

A max_tokens cut (finish=length) leaves an open string and unbalanced
brackets; these helpers close what can be closed, then cut back at
outside-string value boundaries until something parses as a dict.
"""

import json
import logging

log = logging.getLogger(__name__)


def strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return cleaned.strip()


def json_candidates(raw: str) -> list[str]:
    """Ordered candidate bodies: fence-stripped full text, then outermost {…}."""
    cleaned = strip_fences(raw)
    out = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        out.append(cleaned[start : end + 1])
    return list(dict.fromkeys(out))


def close(s: str) -> str | None:
    """Close an unterminated string and unbalanced brackets of *s*.

    Returns None when brackets are mismatched (not worth salvaging).
    """
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and (not stack or stack.pop() != ch):
            return None
    if in_str and esc:
        s = s[:-1]  # dangling escape from truncation — drop it
    return s + ('"' if in_str else "") + "".join(reversed(stack))


def closure_candidates(s: str) -> list[str]:
    """Candidate repairs for truncated JSON, best (fullest) first.

    1. Close the open string/brackets as-is (truncation landed on a complete
       value or inside a string VALUE).
    2. Cut back at each outside-string comma from the end (truncation landed
       mid-key / mid-token) and close from there.
    """
    cands: list[str] = []
    closed = close(s)
    if closed:
        cands.append(closed)

    commas: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == ",":
            commas.append(i)

    for i in reversed(commas[-500:]):
        prefix = s[:i].rstrip()
        if not prefix:
            break
        cand = close(prefix)
        if cand and cand not in cands:
            cands.append(cand)
    return cands


def salvage(raw: str) -> dict | None:
    """Best-effort parse of truncated/malformed LLM JSON into a dict."""
    for cand in json_candidates(raw):
        for attempt in closure_candidates(cand):
            try:
                data = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                log.info("Repaired malformed LLM JSON (%d → %d chars)", len(cand), len(attempt))
                return data
    return None
