"""Best-effort repair of truncated/malformed LLM JSON — shared by parse layers.

Two failure modes, two strategies:

- Stray tokens — the model emits an extra quote/comma mid-document. Bracket
  closing can't fix that, and cut-back would discard everything after the
  error; deleting a few characters at the decode-error position keeps the
  whole document.
- Truncation at max_tokens (finish=length) — an open string and unbalanced
  brackets; close what can be closed, then cut back at outside-string value
  boundaries until something parses as a dict.

Repairs can produce several plausible dicts (a deletion may decode yet mangle
a key), so candidates are yielded best-first and callers with a schema should
validate each instead of trusting the first.
"""

import json
import logging
from collections.abc import Iterator

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


def deletion_repairs(
    s: str, *, max_edits: int = 3, span: int = 4, frontier_cap: int = 48
) -> list[str]:
    """Bodies that decode after deleting a few characters at a decode error.

    Breadth-first, so fewest-edit repairs come first; bounded by *max_edits*
    rounds and *frontier_cap* candidates per round, so garbage input costs a
    handful of decodes instead of a combinatorial search. Truncation is not
    repaired here — unbalanced brackets cannot be closed by deleting — that is
    closure_candidates' job.
    """
    found: list[str] = []
    seen: set[str] = {s}
    frontier = [s]
    for depth in range(max_edits + 1):
        nxt: list[str] = []
        for cur in frontier:
            try:
                json.loads(cur)
            except json.JSONDecodeError as e:
                if depth == max_edits:
                    continue
                for n in range(1, span + 1):
                    lo, hi = max(0, e.pos - n), min(len(cur) - n, e.pos)
                    for start in range(lo, hi + 1):
                        cand = cur[:start] + cur[start + n :]
                        if cand not in seen:
                            seen.add(cand)
                            nxt.append(cand)
            else:
                found.append(cur)  # decodes — never expand further
        if not nxt:
            break
        frontier = nxt[:frontier_cap]
    return found


def repair_candidates(raw: str) -> Iterator[dict]:
    """Dict repairs of *raw*, best first: stray-token deletions (whole document
    kept) before truncation cut-backs (prefix only). Lazy, so a caller that
    validates one candidate successfully never pays for the rest."""
    seen: set[str] = set()
    logged = False

    def decode(body: str, attempt: str) -> dict | None:
        nonlocal logged
        if attempt in seen:
            return None
        seen.add(attempt)
        try:
            data = json.loads(attempt)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        if not logged:
            logged = True
            log.info("Repaired malformed LLM JSON (%d → %d chars)", len(body), len(attempt))
        return data

    for body in json_candidates(raw):
        for attempt in deletion_repairs(body):
            if (data := decode(body, attempt)) is not None:
                yield data
        for attempt in closure_candidates(body):
            if (data := decode(body, attempt)) is not None:
                yield data


def salvage(raw: str) -> dict | None:
    """First dict repair of *raw*, or None — for callers with no schema to validate against."""
    return next(repair_candidates(raw), None)
