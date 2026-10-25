"""Silero VAD (ONNX, 16 kHz mono, 512-sample / 32 ms windows) + hysteresis segmenter."""
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from enum import Enum

import numpy as np
import onnxruntime as ort

WINDOW = 512
CONTEXT = 64
SR = 16_000
WINDOW_MS = WINDOW * 1000 // SR  # 32


def _model_path() -> str:
    """Locate the model shipped with livekit-plugins-silero *without importing the plugin*: importing it registers a
    plugin, which LiveKit only allows on the main thread (and we construct VADs inside job threads)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("livekit.plugins.silero")
    return str(Path(next(iter(spec.submodule_search_locations))) / "resources" / "silero_vad.onnx")


class SileroVAD:
    """One instance per session (carries recurrent state). CPU, single thread."""

    def __init__(self, model_path: str | None = None) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(model_path or _model_path(), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, window: np.ndarray) -> float:
        assert window.shape == (WINDOW,)
        x = np.concatenate([self._ctx, window[None].astype(np.float32)], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state,
                                                 "sr": np.array(SR, dtype=np.int64)})
        self._ctx = x[:, -CONTEXT:]
        return float(out[0, 0])


class VadEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_ms: int = 96     # speech must persist this long to fire SPEECH_START
    min_silence_ms: int = 160   # silence this long fires SPEECH_END (endpointing adds its own wait)


class VadSegmenter:
    """Feeds arbitrary-size 16 kHz float chunks; emits (event, t_ms) with speech-start backdated."""

    def __init__(self, vad: SileroVAD, cfg: VadConfig | None = None) -> None:
        self.vad, self.cfg = vad, cfg or VadConfig()
        self._buf = np.zeros(0, dtype=np.float32)
        self._t = 0  # ms processed
        self._speaking = False
        self._run_speech = 0
        self._run_silence = 0
        self.last_prob = 0.0

    def set_threshold(self, threshold: float, min_speech_ms: int | None = None) -> None:
        """Used by the barge-in controller to raise the bar while the agent talks."""
        self.cfg.threshold = threshold
        if min_speech_ms is not None:
            self.cfg.min_speech_ms = min_speech_ms

    def push(self, pcm16k: np.ndarray) -> list[tuple[VadEvent, int]]:
        self._buf = np.concatenate([self._buf, pcm16k])
        events: list[tuple[VadEvent, int]] = []
        while len(self._buf) >= WINDOW:
            w, self._buf = self._buf[:WINDOW], self._buf[WINDOW:]
            p = self.last_prob = self.vad.prob(w)
            self._t += WINDOW_MS
            if p >= self.cfg.threshold:
                self._run_speech += WINDOW_MS
                self._run_silence = 0
                if not self._speaking and self._run_speech >= self.cfg.min_speech_ms:
                    self._speaking = True
                    events.append((VadEvent.SPEECH_START, self._t - self._run_speech))
            else:
                self._run_silence += WINDOW_MS
                if not self._speaking:
                    self._run_speech = 0  # isolated short bursts are ignored
                elif self._run_silence >= self.cfg.min_silence_ms:
                    self._speaking = False
                    self._run_speech = 0
                    events.append((VadEvent.SPEECH_END, self._t - self._run_silence))
        return events

    @property
    def speaking(self) -> bool:
        return self._speaking
