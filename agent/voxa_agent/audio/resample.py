"""Stateful 48 kHz -> 16 kHz mono decimator (windowed-sinc FIR, factor 3)."""
from __future__ import annotations

import numpy as np

_TAPS = 63
_FACTOR = 3


def _design() -> np.ndarray:
    n = np.arange(_TAPS) - (_TAPS - 1) / 2
    cutoff = 0.45 / _FACTOR  # fraction of input rate, a little under 8 kHz
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(_TAPS)
    return (h / h.sum()).astype(np.float32)


class Decimator48to16:
    """Streaming-safe: carries FIR history and decimation phase across calls."""

    def __init__(self) -> None:
        self._h = _design()
        self._hist = np.zeros(_TAPS - 1, dtype=np.float32)
        self._phase = 0

    def process(self, pcm48: np.ndarray) -> np.ndarray:
        """int16 or float32 mono at 48 kHz -> float32 in [-1, 1] at 16 kHz."""
        x = pcm48.astype(np.float32)
        if pcm48.dtype == np.int16:
            x /= 32768.0
        buf = np.concatenate([self._hist, x])
        y = np.convolve(buf, self._h, mode="valid")  # len == len(x)
        self._hist = buf[-(_TAPS - 1):]
        out = y[self._phase::_FACTOR]
        self._phase = (self._phase - len(y)) % _FACTOR
        return out


class Upsampler16to48:
    """Stateful x3 linear-phase interpolator for TTS output (16 kHz -> 48 kHz)."""

    def __init__(self) -> None:
        n = np.arange(_TAPS) - (_TAPS - 1) / 2
        c = 0.45 / _FACTOR
        h = 2 * c * np.sinc(2 * c * n) * np.hamming(_TAPS)
        self._h = (h / h.sum() * _FACTOR).astype(np.float32)
        self._tail = np.zeros(_TAPS - 1, dtype=np.float32)

    def process(self, pcm16k: np.ndarray) -> np.ndarray:
        x = pcm16k.astype(np.float32)
        z = np.zeros(len(x) * _FACTOR, dtype=np.float32)
        z[::_FACTOR] = x
        buf = np.concatenate([self._tail, z])
        y = np.convolve(buf, self._h, mode="valid")
        self._tail = buf[-(_TAPS - 1):]
        return y
