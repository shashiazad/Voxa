import wave
from pathlib import Path

import numpy as np
import pytest

from voxa_agent.audio.noise import RNNoiseSuppressor
from voxa_agent.audio.resample import Decimator48to16
from voxa_agent.audio.vad import SileroVAD, VadConfig, VadEvent, VadSegmenter

FIX = Path(__file__).parents[1] / "fixtures" / "order_48k.wav"


def load48() -> np.ndarray:
    with wave.open(str(FIX)) as w:
        assert w.getframerate() == 48000
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def chunks(x, n=480):
    for i in range(0, len(x), n):
        yield x[i:i + n]


def test_decimator_length_and_tone_preserved():
    t = np.arange(48000) / 48000
    x = (np.sin(2 * np.pi * 1000 * t) * 16000).astype(np.int16)
    d = Decimator48to16()
    y = np.concatenate([d.process(c) for c in chunks(x, 480)])
    assert abs(len(y) - 16000) <= 2
    assert 0.4 < np.abs(y[1000:]).max() < 0.6  # amplitude preserved (16000/32768)


def test_decimator_streaming_equals_one_shot():
    x = (np.random.default_rng(0).standard_normal(9600) * 3000).astype(np.int16)
    a = Decimator48to16().process(x)
    d = Decimator48to16()
    b = np.concatenate([d.process(c) for c in chunks(x, 480)])
    assert len(a) == len(b) and np.allclose(a, b, atol=1e-5)


def run_vad(pcm16k, cfg=None):
    seg = VadSegmenter(SileroVAD(), cfg or VadConfig())
    ev = []
    for i in range(0, len(pcm16k), 320):
        ev += seg.push(pcm16k[i:i + 320])
    return ev


def to16(x48):
    return Decimator48to16().process(x48)


def test_vad_detects_speech_then_end():
    speech = to16(load48())
    sil = np.zeros(16000, dtype=np.float32)
    ev = run_vad(np.concatenate([sil, speech, sil]))
    kinds = [e for e, _ in ev]
    assert kinds[0] is VadEvent.SPEECH_START and kinds[-1] is VadEvent.SPEECH_END
    assert 900 <= ev[0][1] <= 1400  # starts after ~1 s of silence


def test_vad_ignores_silence_and_white_noise():
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(32000) * 0.02).astype(np.float32)
    assert run_vad(noise) == []


def test_vad_ignores_short_click():
    x = np.zeros(32000, dtype=np.float32)
    x[8000:8160] = 0.8  # 10 ms click
    assert run_vad(x) == []


def test_rnnoise_reduces_noise_floor_and_keeps_speech():
    speech = load48()
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(len(speech)) * 800
    noisy = np.clip(speech + noise, -32768, 32767).astype(np.int16)
    ns = RNNoiseSuppressor()
    out = np.concatenate([ns.process(c) for c in chunks(noisy)])
    tail = slice(-4800, None)  # trailing region
    assert np.abs(out[tail].astype(float)).mean() < np.abs(noisy[tail].astype(float)).mean() * 0.6
