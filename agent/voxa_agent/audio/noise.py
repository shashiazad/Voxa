"""Neural noise suppression (runs before VAD). RNNoise via pyrnnoise, 48 kHz int16 mono."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class NoiseSuppressor(Protocol):
    def process(self, pcm48: np.ndarray) -> np.ndarray: ...


class PassthroughSuppressor:
    def process(self, pcm48: np.ndarray) -> np.ndarray:
        return pcm48


class RNNoiseSuppressor:
    """RNNoise buffers into 10 ms frames internally; output length can lag input by < 10 ms."""

    def __init__(self) -> None:
        from pyrnnoise import RNNoise

        self._rn = RNNoise(48_000)

    def process(self, pcm48: np.ndarray) -> np.ndarray:
        outs = [np.asarray(f).reshape(-1) for _p, f in self._rn.denoise_chunk(pcm48.astype(np.int16))]
        if not outs:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(outs).astype(np.int16)
