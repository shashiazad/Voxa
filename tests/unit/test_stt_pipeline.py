import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from voxa_agent.audio.noise import RNNoiseSuppressor
from voxa_agent.audio.vad import SileroVAD, VadSegmenter
from voxa_agent.pipeline import InboundPipeline
from voxa_agent.providers.stt import CircuitBreaker, FallbackSTT, FasterWhisperSTT, STTProvider

FIX = Path(__file__).parents[1] / "fixtures" / "order_48k.wav"


@pytest.fixture(scope="module")
def stt():
    return FasterWhisperSTT("base.en")


def load48():
    with wave.open(str(FIX)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


@pytest.mark.asyncio
async def test_pipeline_end_to_end_transcript_and_latency(stt):
    sig, evs = [], []

    async def on_vad(s): sig.append(s)
    async def on_stt(e): evs.append(e)

    stream = await stt.open("en")
    pipe = InboundPipeline(RNNoiseSuppressor(), VadSegmenter(SileroVAD()), stream, on_vad, on_stt)
    pipe.start()
    audio = np.concatenate([np.zeros(24000, np.int16), load48(), np.zeros(48000, np.int16)])
    for i in range(0, len(audio), 480):  # real-time 10 ms pacing
        pipe.feed(audio[i:i + 480])
        await asyncio.sleep(0.01)
    # VAD may split at the comma pause ("Hi," / "I would like..."); wait for the SPEECH_END count
    for _ in range(150):
        ends = sum(1 for s in sig if s.event.value == "speech_end")
        finals = sum(1 for e in evs if e.is_final)
        starts = sum(1 for s in sig if s.event.value == "speech_start")
        if ends and ends == starts and finals >= ends:
            break
        await asyncio.sleep(0.1)
    await pipe.aclose()

    finals = [e for e in evs if e.is_final]
    final = finals[-1]
    text = " ".join(e.text for e in finals).lower()
    assert "order" in text and "status" in text, text
    assert final.words and final.words[0].end > final.words[0].start
    assert 0 < final.confidence <= 1
    # NS+VAD per 10 ms frame must stay well inside the 30 ms budget
    ms = sorted(pipe.frame_ms)
    print(f"\nNS+VAD per-frame ms p50={ms[len(ms)//2]:.2f} p95={ms[int(len(ms)*.95)]:.2f} "
          f"stt_final_decode_ms={final.latency_ms:.0f} dropped={pipe.dropped_frames}")
    assert ms[int(len(ms) * .95)] < 30
    assert pipe.dropped_frames == 0


class _Bad(STTProvider):
    name = "bad"
    async def open(self, language="en"): raise ConnectionError("down")


class _Good(STTProvider):
    name = "good"
    async def open(self, language="en"): return object()


@pytest.mark.asyncio
async def test_fallback_skips_failed_provider_and_trips_breaker():
    fb = FallbackSTT([_Bad(), _Good()])
    for _ in range(4):
        assert await fb.open() is not None
    assert fb._breakers["bad"].open


def test_breaker_resets_after_window():
    b = CircuitBreaker(failures=1, reset_s=0.0)
    b.fail()
    assert not b.open


async def test_echo_gate_ignores_quiet_signal_next_to_loud_playback_but_not_when_silent():
    from voxa_agent.pipeline import InboundPipeline
    from voxa_agent.audio.noise import PassthroughSuppressor

    stt = await FasterWhisperSTT("base.en").open("en") if False else None

    class NullStream:
        async def push(self, p): pass
        async def end_utterance(self): pass
        async def events(self):
            await asyncio.sleep(3600); yield
        async def aclose(self): pass

    def speechlike(amp):
        w = wave.open(str(FIX)); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        return (x / np.abs(x).max() * amp * 32768).astype(np.int16)

    async def run(ref):
        sig = []
        async def on_vad(s): sig.append(s)
        async def on_stt(e): pass
        pipe = InboundPipeline(PassthroughSuppressor(), VadSegmenter(SileroVAD()), NullStream(), on_vad, on_stt,
                               ref_level=lambda: ref, echo_gate_ratio=0.05)
        pipe.start()
        for i in range(0, 96000, 480):
            pipe.feed(speechlike(0.01)[i:i + 480])      # real speech, but only ~-40 dBFS
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.3)
        await pipe.aclose()
        return len([s for s in sig if s.event.value == "speech_start"]), pipe.gated_frames

    starts_loud, gated = await run(ref=0.2)     # agent playing loudly: 0.01 < 0.05*0.2? no -> 0.01 == 0.01 boundary; use rms below
    assert starts_loud == 0 and gated > 100
    starts_quiet, gated0 = await run(ref=0.0)   # agent silent: the same quiet speech is a real caller
    assert starts_quiet >= 1 and gated0 == 0


async def test_near_field_gate_learns_caller_level_then_ignores_much_quieter_background():
    from voxa_agent.pipeline import InboundPipeline
    from voxa_agent.audio.noise import PassthroughSuppressor

    class NullStream:
        async def push(self, p): pass
        async def end_utterance(self): pass
        async def events(self):
            await asyncio.sleep(3600); yield
        async def aclose(self): pass

    w = wave.open(str(FIX)); base = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    base = base / np.abs(base).max()
    ref = {"v": 0.0}

    async def run_phase(pipe, amp, seconds=3.0):
        n = int(seconds * 48000)
        x = (np.resize(base, n) * amp * 32768).astype(np.int16)
        for i in range(0, n, 480):
            pipe.feed(x[i:i + 480]); await asyncio.sleep(0.0015)
        await asyncio.sleep(0.2)

    sig = []
    async def on_vad(s): sig.append(s.event.value)
    async def on_stt(e): pass
    pipe = InboundPipeline(PassthroughSuppressor(), VadSegmenter(SileroVAD()), NullStream(), on_vad, on_stt,
                           ref_level=lambda: ref["v"], echo_gate_ratio=0.0, near_field_db=9)
    pipe.start()
    await run_phase(pipe, 0.5)                    # caller talks normally, agent silent -> level learned
    assert pipe.user_level > 0.02
    learned = pipe.user_level
    ref["v"] = 0.2                                # agent now speaking
    before = pipe.gated_frames
    await run_phase(pipe, 0.5 * 0.1)              # 20 dB quieter speech (TV in the next room)
    assert pipe.gated_frames - before > 150
    mid = pipe.gated_frames
    await run_phase(pipe, 0.5 * 0.8)              # the caller talking over the agent, slightly softer
    assert pipe.gated_frames - mid < 40
    await pipe.aclose()
    assert pipe.user_level == pytest.approx(learned, rel=0.05)   # level is frozen while the agent speaks
