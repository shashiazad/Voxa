"""Sentence-aware chunking into ~300-500 token windows with overlap.
Tokens are approximated as words / 0.75 (no tokenizer dependency; accurate to ~10% for English)."""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def approx_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75) + 1


def chunk_text(text: str, min_tokens: int = 300, max_tokens: int = 500, overlap_tokens: int = 60) -> list[str]:
    sentences = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_t = 0
    for s in sentences:
        t = approx_tokens(s)
        if t > max_tokens:                      # one huge "sentence": hard-split by words
            words = s.split()
            step = int(max_tokens * 0.75)
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i + step])
                if cur:
                    chunks.append(" ".join(cur)); cur, cur_t = [], 0
                chunks.append(piece)
            continue
        if cur_t + t > max_tokens and cur_t >= min_tokens:
            chunks.append(" ".join(cur))
            # carry trailing sentences as overlap
            tail, tt = [], 0
            for prev in reversed(cur):
                tt += approx_tokens(prev)
                tail.insert(0, prev)
                if tt >= overlap_tokens:
                    break
            cur, cur_t = tail, sum(approx_tokens(x) for x in tail)
        cur.append(s)
        cur_t += t
    if cur:
        if chunks and cur_t < min_tokens // 3 and approx_tokens(chunks[-1]) + cur_t <= max_tokens:
            chunks[-1] += " " + " ".join(cur)    # avoid a tiny trailing chunk
        else:
            chunks.append(" ".join(cur))
    return chunks


def chunk_faq(pairs: list[tuple[str, str]]) -> list[str]:
    """FAQ rows are atomic: one chunk per Q/A pair keeps the answer attached to its question."""
    return [f"Q: {q.strip()}\nA: {a.strip()}" for q, a in pairs if q.strip() and a.strip()]
