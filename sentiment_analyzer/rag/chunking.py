"""
Text chunking for RAG ingestion.

Splits on paragraph, then sentence, then word boundaries so chunks don't
cut words in half, and overlaps consecutive chunks so context that spans
a boundary is retrievable from either side.
"""

import re

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_units(text: str, chunk_size: int) -> list[str]:
    """Break text into units no longer than ``chunk_size``."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
                continue
            # Very long sentence: fall back to words, then hard-split.
            current = ""
            for word in sentence.split(" "):
                while len(word) > chunk_size:
                    if current:
                        units.append(current)
                        current = ""
                    units.append(word[:chunk_size])
                    word = word[chunk_size:]
                candidate = f"{current} {word}".strip()
                if len(candidate) > chunk_size:
                    units.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                units.append(current)
    return units


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split ``text`` into chunks of at most ``chunk_size`` characters.

    Consecutive chunks share up to ``chunk_overlap`` trailing characters
    (rounded to whole words) from the previous chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    current = ""
    for unit in _split_units(text, chunk_size):
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        overlap = _tail_words(current, chunk_overlap)
        current = f"{overlap} {unit}".strip() if overlap else unit
        if len(current) > chunk_size:
            current = unit
    if current:
        chunks.append(current)
    return chunks


def _tail_words(text: str, max_chars: int) -> str:
    """Trailing whole words of ``text`` totalling at most ``max_chars``."""
    if max_chars <= 0:
        return ""
    words = text.split(" ")
    tail: list[str] = []
    length = 0
    for word in reversed(words):
        if length + len(word) + (1 if tail else 0) > max_chars:
            break
        tail.insert(0, word)
        length += len(word) + (1 if len(tail) > 1 else 0)
    return " ".join(tail)
