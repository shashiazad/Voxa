"""Clause chunker: release TTS text at the first clause boundary or ~10 words, whichever is first."""
from __future__ import annotations

import re

_BOUNDARY = re.compile(r"[,.;:!?](?=\s)|\n")


class ClauseChunker:
    """feed() takes LLM token text, returns finished chunks. A punctuation mark only counts when
    followed by whitespace, so '$1,234.50' and 'e.g.' are never split mid-token."""

    def __init__(self, first_max_words: int = 10, later_max_words: int = 24, min_words: int = 2) -> None:
        self.first_max, self.later_max, self.min_words = first_max_words, later_max_words, min_words
        self._buf = ""
        self._emitted = 0

    def _limit(self) -> int:
        return self.first_max if self._emitted == 0 else self.later_max

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                return out
            out.append(chunk)

    def _take(self) -> str | None:
        m = _BOUNDARY.search(self._buf)
        while m:
            head = self._buf[: m.end()]
            if len(head.split()) >= self.min_words or self._emitted:
                break
            m = _BOUNDARY.search(self._buf, m.end())  # too short ("Sure,"): keep accumulating
        if m:
            return self._cut(m.end())
        words = list(re.finditer(r"\S+\s", self._buf))  # only complete words
        if len(words) >= self._limit():
            return self._cut(words[self._limit() - 1].end())
        return None

    def _cut(self, idx: int) -> str:
        chunk, self._buf = self._buf[:idx].strip(), self._buf[idx:].lstrip()
        self._emitted += 1
        return chunk

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        if rest:
            self._emitted += 1
            return [rest]
        return []
