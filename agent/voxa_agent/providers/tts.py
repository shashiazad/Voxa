"""Streaming TTS interface and Piper (local, free) provider."""
from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np


@dataclass
class WordTiming:
    text: str
    start_ms: float  # relative to the start of the *turn's* audio
    end_ms: float


@dataclass
class TTSChunk:
    pcm: np.ndarray            # int16 mono at `sample_rate`
    sample_rate: int
    words: list[WordTiming]    # words contained in this chunk, turn-relative timing


class TTSContext(ABC):
    """One per agent turn: keeps voice/prosody state and a running audio clock across clauses."""

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[TTSChunk]: ...

    async def aclose(self) -> None: ...


class TTSProvider(ABC):
    name: str
    sample_rate: int

    @abstractmethod
    def context(self) -> TTSContext: ...

    async def warm(self) -> None: ...


def estimate_word_timings(text: str, duration_ms: float, offset_ms: float) -> list[WordTiming]:
    """Piper has no word alignment: distribute duration across words by character weight.
    Good to ~±80 ms per word, which is adequate for history truncation (word granularity)."""
    words = text.split()
    if not words:
        return []
    weights = np.array([len(w) + 1 for w in words], dtype=float)
    ends = np.cumsum(weights) / weights.sum() * duration_ms
    starts = np.concatenate([[0.0], ends[:-1]])
    return [WordTiming(w, offset_ms + s, offset_ms + e) for w, s, e in zip(words, starts, ends)]


class PiperTTS(TTSProvider):
    """Local Piper. `voice` may be a preloaded PiperVoice shared across calls (loading is ~1 s and 60 MB);
    `rate` > 1 speaks faster (Piper's length_scale is the inverse)."""

    name = "piper"

    def __init__(self, model_path: str | None = None, voice=None, rate: float = 1.0) -> None:
        from piper import PiperVoice

        self._voice = voice or PiperVoice.load(model_path)
        self.sample_rate = self._voice.config.sample_rate
        self.rate = rate

    @staticmethod
    def load_voice(model_path: str):
        from piper import PiperVoice

        return PiperVoice.load(model_path)

    def context(self) -> TTSContext:
        return _PiperContext(self)

    async def warm(self) -> None:
        async for _ in self.context().synthesize("Hello."):
            pass


class _PiperContext(TTSContext):
    def __init__(self, tts: PiperTTS) -> None:
        self._tts = tts
        self._clock_ms = 0.0

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        cancelled = False

        from piper.config import SynthesisConfig

        cfg = SynthesisConfig(length_scale=1.0 / max(self._tts.rate, 0.1))

        def run() -> None:
            try:
                for ch in self._tts._voice.synthesize(text, syn_config=cfg):
                    if cancelled:
                        return
                    loop.call_soon_threadsafe(q.put_nowait, ch.audio_int16_array)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, run)
        parts: list[np.ndarray] = []
        try:
            while (a := await q.get()) is not None:
                parts.append(a)
            await fut
        except asyncio.CancelledError:
            cancelled = True  # worker thread stops after its current Piper sentence
            raise
        if not parts:
            return
        pcm = np.concatenate(parts)
        dur = len(pcm) / self._tts.sample_rate * 1000
        words = estimate_word_timings(re.sub(r"\s+", " ", text), dur, self._clock_ms)
        self._clock_ms += dur
        yield TTSChunk(pcm, self._tts.sample_rate, words)
