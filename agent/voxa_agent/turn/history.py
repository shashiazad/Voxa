"""Cut the assistant message in history down to what the user actually heard."""
from __future__ import annotations

from dataclasses import dataclass

from ..providers.tts import WordTiming

INTERRUPTED = "[interrupted]"


@dataclass
class SpokenChunk:
    original: str                 # LLM text for this clause (what goes in history)
    words: list[WordTiming]       # TTS word timings of the *normalized* text, turn-relative
    start_ms: float
    end_ms: float


def heard_text(chunks: list[SpokenChunk], heard_ms: float) -> tuple[str, bool]:
    """Return (text the user heard, whether anything was cut). Fully heard clauses are kept verbatim;
    the clause being played when audio stopped is cut proportionally by word timing (normalized text
    can have a different word count than the original, e.g. '$5' -> 'five dollars')."""
    out: list[str] = []
    for c in chunks:
        if c.end_ms <= heard_ms:
            out.append(c.original)
            continue
        if c.start_ms >= heard_ms:
            return " ".join(out), True
        n_heard = sum(1 for w in c.words if w.end_ms <= heard_ms)
        frac = n_heard / max(len(c.words), 1)
        orig = c.original.split()
        keep = int(frac * len(orig))
        if keep:
            out.append(" ".join(orig[:keep]))
        return " ".join(out), True
    return " ".join(out), False


def history_message(chunks: list[SpokenChunk], heard_ms: float) -> str | None:
    """Assistant content with the [interrupted] marker, or None if the user heard nothing."""
    text, cut = heard_text(chunks, heard_ms)
    if not text.strip():
        return None
    return f"{text} {INTERRUPTED}" if cut else text
