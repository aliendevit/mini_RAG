# src/text_utils.py
from __future__ import annotations

import re
from typing import List, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent chunking & embedding:
    - Normalize newlines to '\n'
    - Strip trailing spaces per line
    - Collapse excessive blank lines
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    # collapse 3+ newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _find_best_boundary(text: str, start: int, hard_end: int, lookback: int = 250) -> int:
    """
    Try to move the end backward to a "nice" boundary:
    preference: blank line, newline, sentence boundary.
    """
    if hard_end <= start:
        return hard_end

    window_start = max(start, hard_end - lookback)
    window = text[window_start:hard_end]

    # Priority 1: paragraph boundary "\n\n"
    idx = window.rfind("\n\n")
    if idx != -1 and (window_start + idx) > start + 200:
        return window_start + idx

    # Priority 2: line boundary "\n"
    idx = window.rfind("\n")
    if idx != -1 and (window_start + idx) > start + 200:
        return window_start + idx

    # Priority 3: sentence-ish boundary ". " or "? " or "! "
    for pat in [". ", "? ", "! "]:
        idx = window.rfind(pat)
        if idx != -1 and (window_start + idx) > start + 200:
            return window_start + idx + 1  # keep punctuation

    return hard_end


def chunk_text_smart(
    text: str,
    max_chars: int = 900,
    overlap: int = 150,
) -> List[Tuple[int, int, str]]:
    """
    Chunk text into (start_char, end_char, chunk_text) using:
    - hard max length max_chars
    - overlap between consecutive chunks
    - prefer cutting on paragraph/line/sentence boundaries for better retrieval.
    """
    text = normalize_text(text)
    n = len(text)
    if n == 0:
        return []

    chunks: List[Tuple[int, int, str]] = []
    start = 0

    while start < n:
        hard_end = min(start + max_chars, n)
        end = _find_best_boundary(text, start, hard_end)

        # Ensure forward progress (avoid infinite loops)
        if end <= start:
            end = hard_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))

        if end >= n:
            break

        # overlap: move start forward but keep some context
        start = max(0, end - overlap)

    return chunks
