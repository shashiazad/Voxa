import asyncio
import json

import httpx
import numpy as np
import pytest

from voxa_agent.audio.playout import FRAME, FRAME_MS, Playout
from voxa_agent.metrics import TurnTimings
from voxa_agent.providers.llm import GroqLLM, LLMDelta, LLMProvider
from voxa_agent.providers.tts import TTSChunk, TTSContext, TTSProvider, WordTiming, estimate_word_timings
from voxa_agent.responder import Responder


class FakeSink:
    """Plays in real time: queued_ms drains at wall-clock rate."""
    def __init__(self):
        self.frames = 0
        self._t0 = None
        self.cleared = 0

    async def capture(self, pcm):
        assert len(pcm) == FRAME
        self.frames += 1

    def clear(self):
        self.cleared += 1
        self._drained = self.frames  # everything queued is dropped

    @property
    def queued_ms(self):
        import time
        if self._t0 is None:
            self._t0 = time.monotonic()
        played = (time.monotonic() - self._t0) * 1000
        return max(0.0, self.frames * FRAME_MS - played) if self.frames else 0.0


def pcm_ms(ms):
    return np.ones(48 * ms, dtype=np.int16)


async def test_playout_never_exceeds_200ms_ahead_and_flush_is_instant():
    sink = FakeSink()
    p = Playout(sink)
    p.start()
    p.begin_generation(1)
    await p.enqueue(1, pcm_ms(3000), [])
    peak = 0.0
    for _ in range(40):
        await asyncio.sleep(0.025)
        peak = max(peak, sink.queued_ms)
    assert peak <= 200 + FRAME_MS
    sent_before = sink.frames
    played = p.flush()
    await asyncio.sleep(0.1)
    assert sink.frames == sent_before          # nothing more sent after flush
    assert sink.cleared == 1 and played > 0
    await p.stop()


async def test_stale_generation_dropped():
    sink = FakeSink()
    p = Playout(sink)
    p.start()
    p.begin_generation(2)
    await p.enqueue(1, pcm_ms(100), [])       # stale
    await asyncio.sleep(0.1)
    assert sink.frames == 0
    await p.stop()


async def test_word_events_follow_playout_position():
    sink, seen = FakeSink(), []

    async def on_word(g, w): seen.append((g, w.text))

    p = Playout(sink, on_word=on_word)
    p.start(); p.begin_generation(1)
    await p.enqueue(1, pcm_ms(600), [WordTiming("hello", 0, 200), WordTiming("world", 300, 600)])
    await asyncio.sleep(0.15)
    assert seen == [(1, "hello")]
    await asyncio.sleep(0.6)
    assert [w for _, w in seen] == ["hello", "world"]
    await p.stop()


def test_estimated_word_timings_cover_duration():
    w = estimate_word_timings("a bb ccc", 600, 100)
    assert w[0].start_ms == 100 and abs(w[-1].end_ms - 700) < 1e-6


class FakeTTS(TTSProvider):
    name, sample_rate = "fake", 48_000

    def context(self):
        outer = self

        class C(TTSContext):
            async def synthesize(self, text):
                await asyncio.sleep(0.02)
                yield TTSChunk(np.ones(48 * 100, dtype=np.int16), 48_000,
                               [WordTiming(w, 0, 1) for w in text.split()])
        return C()


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, toks, delay=0.01):
        self.toks, self.delay = toks, delay

    async def stream(self, messages, tools=None, max_tokens=160):
        for t in self.toks:
            await asyncio.sleep(self.delay)
            yield LLMDelta(text=t)


async def test_responder_streams_and_marks_timings():
    sink = FakeSink()
    pl = Playout(sink); pl.start()
    r = Responder(FakeLLM(["Sure", ",", " let", " me", " check", ".", " One", " moment", "."]), FakeTTS(), pl)
    t = TurnTimings(0)
    t.mark("turn_committed")
    await r.start([{"role": "user", "content": "hi"}], t)
    await r.wait()
    await asyncio.sleep(0.4)
    assert "llm_first_token" in t.marks and "tts_first_byte" in t.marks
    assert sink.frames > 0
    assert r.last_text.startswith("Sure, let me check.")
    await pl.stop()


async def test_new_turn_cancels_old_generation_audio():
    sink = FakeSink()
    pl = Playout(sink); pl.start()
    slow = FakeLLM(["word "] * 200, delay=0.02)
    r = Responder(slow, FakeTTS(), pl)
    g1 = await r.start([], TurnTimings(0))
    await asyncio.sleep(0.3)
    played = await r.cancel()
    frames = sink.frames
    await asyncio.sleep(0.2)
    assert sink.frames == frames and played > 0 and r._task.done()
    await pl.stop()


def sse(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


async def test_groq_sse_parsing_text_and_tool_call():
    body = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "order_status", "arguments": "{\"id\":"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"A1\"}"}}]}, "finish_reason": "tool_calls"}]},
    )
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    llm = GroqLLM("k", transport=httpx.MockTransport(handler))
    out = [d async for d in llm.stream([{"role": "user", "content": "x"}], tools=[{"type": "function"}])]
    assert "".join(d.text for d in out) == "Hello"
    tc = next(d.tool_call for d in out if d.tool_call)
    assert tc == {"id": "c1", "name": "order_status", "arguments": "{\"id\":\"A1\"}"}
    assert seen["auth"] == "Bearer k" and seen["body"]["stream"] is True


async def test_groq_error_status_raises():
    llm = GroqLLM("k", transport=httpx.MockTransport(lambda r: httpx.Response(429, text="slow down")))
    with pytest.raises(RuntimeError, match="429"):
        [d async for d in llm.stream([])]


from voxa_agent.providers.llm import FallbackLLM


class Boom(LLMProvider):
    def __init__(self, name, exc, after=0):
        self.name, self.exc, self.after, self.calls = name, exc, after, 0

    async def stream(self, messages, tools=None, max_tokens=160):
        self.calls += 1
        for i in range(self.after):
            yield LLMDelta(text=f"t{i} ")
        raise self.exc


async def test_fallback_llm_skips_failed_provider_and_cools_it_down():
    bad, good = Boom("a", RuntimeError("groq 429: slow down")), FakeLLM(["ok ", "then"], delay=0)
    good.name = "b"
    chain = FallbackLLM([bad, good], cooldown_s=60, cooldowns={})
    async def run():
        return "".join([d.text async for d in chain.stream([])])
    out = [await run() for _ in range(3)]
    assert out == ["ok then"] * 3 and bad.calls == 1        # 429 => skipped for the cool-down window


async def test_fallback_llm_does_not_splice_after_output_started_and_raises_when_all_fail():
    mid = Boom("a", RuntimeError("boom"), after=2)
    chain = FallbackLLM([mid, FakeLLM(["x"], delay=0)], cooldowns={})
    got = []
    with pytest.raises(RuntimeError):
        async for d in chain.stream([]):
            got.append(d.text)
    assert got == ["t0 ", "t1 "]
    with pytest.raises(RuntimeError, match="down"):
        async for _ in FallbackLLM([Boom("a", RuntimeError("down")), Boom("b", RuntimeError("down"))], cooldowns={}).stream([]):
            pass


async def test_llm_failure_is_spoken_not_silent():
    sink = FakeSink(); pl = Playout(sink); pl.start()
    from voxa_agent.responder import ResponderConfig
    r = Responder(Boom("x", RuntimeError("all down")), FakeTTS(), pl, ResponderConfig(error_line="Sorry, trouble right now."))
    await r.start([{"role": "user", "content": "hi"}], TurnTimings(0))
    await r.wait()
    await asyncio.sleep(0.3)
    assert r.last_text == "Sorry, trouble right now." and sink.frames > 0
    await pl.stop()
