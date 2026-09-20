"""Text chunking.

Strategy and rationale
----------------------
Fixed-size character windows with a fixed overlap, snapped to the nearest
sentence or whitespace boundary.

*Why fixed size?* Retrieval quality depends on chunks being large enough to
carry an idea but small enough that a match is precise. ~900 characters is
roughly a paragraph (~200 tokens), so a handful of chunks fits comfortably in a
prompt while each one still stands on its own.

*Why overlap?* A hard cut can split a sentence away from the fact that makes it
meaningful. ~150 characters of overlap means a statement straddling a boundary
still appears whole in one of the two neighbouring chunks.

*Why snap to boundaries?* Cutting mid-word produces fragments that embed poorly
and read badly when shown as a citation. We look back a short distance for a
sentence end, then for any whitespace, and only cut mid-word if neither exists
(e.g. one enormous unbroken token).

*What this deliberately is not:* token-accurate or structure-aware. A
production version would count real tokens and respect document structure
(headings, lists, code blocks). See the README tradeoffs section.
"""

from __future__ import annotations

import re

#: How far back from a hard cut we are willing to look for a nicer break point,
#: as a fraction of the window. Beyond this, honouring structure would shrink
#: chunks too aggressively.
_BOUNDARY_SEARCH_RATIO = 0.25

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")
_WHITESPACE_RUN = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Collapse whitespace runs and trim. Keeps chunk sizes meaningful."""
    return _WHITESPACE_RUN.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split ``text`` into overlapping chunks.

    Returns an empty list for blank input, and a single chunk when the text fits
    in one window.

    Raises:
        ValueError: if ``chunk_size`` is not positive, or ``overlap`` is negative
            or greater than/equal to ``chunk_size`` (which would never advance).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = normalise_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)

        # Only look for a nicer boundary when we are actually cutting the text.
        if end < length:
            end = _find_break_point(cleaned, start, end, chunk_size)

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        # Step back by the overlap, but always make forward progress, and start
        # the next chunk on a word boundary so the overlap is not a fragment.
        start = _snap_to_word_start(cleaned, max(end - overlap, start + 1), limit=end)

    return chunks


def _find_break_point(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    """Move ``hard_end`` back to the closest sentence end, else whitespace."""
    earliest = max(start + 1, hard_end - int(chunk_size * _BOUNDARY_SEARCH_RATIO))
    window = text[earliest:hard_end]

    sentence_breaks = list(_SENTENCE_END.finditer(window))
    if sentence_breaks:
        return earliest + sentence_breaks[-1].end()

    space = text.rfind(" ", earliest, hard_end)
    if space != -1:
        return space + 1

    # No boundary available (e.g. a single very long token): cut hard.
    return hard_end


def _snap_to_word_start(text: str, position: int, limit: int) -> int:
    """Advance ``position`` to the start of a whole word, never past ``limit``.

    Without this, stepping back by the overlap can land mid-word and the next
    chunk would open with a fragment like "percalifragilistic". Moving forward
    only trims text the previous chunk already covered, so nothing is lost.
    """
    if position <= 0 or position >= len(text):
        return position

    if text[position - 1].isspace() or text[position].isspace():
        while position < limit and text[position].isspace():
            position += 1
        return position

    space = text.find(" ", position, limit)
    if space == -1:
        # An unbroken token spans the whole overlap; a mid-word start is the
        # only option left.
        return position
    return space + 1


def estimate_chunk_count(text: str, chunk_size: int = 900, overlap: int = 150) -> int:
    """Cheap size estimate, useful for logging before doing the real work."""
    cleaned = normalise_text(text)
    if not cleaned:
        return 0
    stride = chunk_size - overlap
    return max(1, -(-(len(cleaned) - overlap) // stride))
