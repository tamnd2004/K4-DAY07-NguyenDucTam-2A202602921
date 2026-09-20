from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    # Split after . ! or ? when followed by whitespace, keeping the terminator
    # on the sentence it belongs to. Covers ". ", "! ", "? " and ".\n".
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        for start in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[start : start + self.max_sentences_per_chunk]
            chunk = " ".join(group).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        parts = self._SENTENCE_BOUNDARY.split(text)
        return [sentence.strip() for sentence in parts if sentence.strip()]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text.strip()
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separator left to try (or the caller asked for a hard cut):
        # fall back to slicing so we never return an oversized chunk.
        if not remaining_separators or remaining_separators[0] == "":
            return self._hard_split(current_text)

        separator, rest = remaining_separators[0], remaining_separators[1:]
        pieces = current_text.split(separator)
        if len(pieces) == 1:
            # This separator does not occur — try the next one.
            return self._split(current_text, rest)

        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            candidate = piece if not buffer else buffer + separator + piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = ""
            if len(piece) <= self.chunk_size:
                buffer = piece
            else:
                # Still too big: go one level deeper in the separator list.
                chunks.extend(self._split(piece, rest))
        if buffer:
            chunks.append(buffer)

        return [stripped for stripped in (chunk.strip() for chunk in chunks) if stripped]

    def _hard_split(self, text: str) -> list[str]:
        return [text[start : start + self.chunk_size] for start in range(0, len(text), self.chunk_size)]


class HeadingChunker:
    """
    Split text at heading boundaries so each policy section stays one chunk.

    A heading is either a Markdown ATX line ("## Điều 4 — ...") or a numbered
    section title ("1.", "2.3.") short enough to be a title rather than a
    numbered clause of body text.

    Rules:
        - Each section becomes one chunk: its heading line plus the body below.
        - Text before the first heading becomes a leading chunk of its own.
        - A section longer than chunk_size is split with RecursiveChunker, and
          the heading is re-attached to every piece so a fragment still says
          which clause it belongs to.
    """

    # Numbered clauses in the Shopee corpus ("2.1. Theo các điều khoản...") run
    # for hundreds of characters, so line length is what separates a section
    # title from a paragraph that merely starts with a number.
    MAX_HEADING_LENGTH = 120

    _ATX_HEADING = re.compile(r"^#{1,6}\s+\S")
    _NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        chunks: list[str] = []
        for heading, body in self._split_sections(text):
            chunks.extend(self._emit_section(heading, body))
        return chunks

    def _is_heading(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if self._ATX_HEADING.match(stripped):
            return True
        return (
            len(stripped) <= self.MAX_HEADING_LENGTH
            and bool(self._NUMBERED_HEADING.match(stripped))
        )

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Group lines into (heading, body) pairs; heading is "" before the first one."""
        sections: list[tuple[str, list[str]]] = []
        heading = ""
        body: list[str] = []
        for line in text.splitlines():
            if self._is_heading(line):
                if heading or any(item.strip() for item in body):
                    sections.append((heading, body))
                heading, body = line.strip(), []
            else:
                body.append(line)
        if heading or any(item.strip() for item in body):
            sections.append((heading, body))
        return [(head, "\n".join(lines).strip()) for head, lines in sections]

    def _emit_section(self, heading: str, body: str) -> list[str]:
        section = f"{heading}\n{body}".strip() if heading else body
        if not section:
            return []
        if len(section) <= self.chunk_size:
            return [section]
        if not heading:
            return RecursiveChunker(chunk_size=self.chunk_size).chunk(section)

        # Reserve room for the heading re-attached below. A very long heading
        # would otherwise shrink the body budget to nothing, so never drop it
        # under half of chunk_size.
        budget = max(self.chunk_size - len(heading) - 1, self.chunk_size // 2)
        pieces = RecursiveChunker(chunk_size=budget).chunk(body)
        if not pieces:
            return [heading]
        return [f"{heading}\n{piece}" for piece in pieces]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = min(50, max(0, chunk_size // 10))
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(chunk) for chunk in chunks]
            comparison[name] = {
                "chunks": chunks,
                "count": len(chunks),
                "avg_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
            }
        return comparison
