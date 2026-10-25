import asyncio

import httpx
import numpy as np

from voxa_agent.providers.stt import GroqWhisperSTT, STTEvent, STTProvider, STTStream, Word


class LocalStream(STTStream):
    def __init__(self):
        self.q = asyncio.Queue(); self.discarded = 0; self.ended = 0
    async def push(self, pcm): 
        if len(pcm) and self.q.empty(): await self.q.put(STTEvent("hel", False, 0.5))
    async def end_utterance(self):
        self.ended += 1; await self.q.put(STTEvent("hello local", True, 0.7, [Word("hello", 0, .3, .7)]))
    def discard_utterance(self): self.discarded += 1
    async def events(self):
        while True:
            yield await self.q.get()
    async def aclose(self): pass


class Local(STTProvider):
    name = "local"
    def __init__(self): self.s = LocalStream()
    async def open(self, language="en"): return self.s


def provider(handler):
    loc = Local()
    return GroqWhisperSTT("k", loc, transport=httpx.MockTransport(handler)), loc


async def utterance(stream, n=16000):
    await stream.push(np.ones(n, dtype=np.float32) * 0.1)
    await stream.end_utterance()
    it = stream.events()
    out = []
    while True:
        ev = await asyncio.wait_for(it.__anext__(), 3)
        out.append(ev)
        if ev.is_final:
            return out


async def test_groq_final_replaces_local_final_and_keeps_interims():
    seen = {}

    def h(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]; seen["body"] = req.content
        return httpx.Response(200, json={"text": " Where is my order?", "words": [{"word": "Where", "start": 0, "end": .2}],
                                         "segments": [{"avg_logprob": -0.1}]})

    p, loc = provider(h)
    evs = await utterance(await p.open())
    assert [e.is_final for e in evs] == [False, True]
    assert evs[0].text == "hel" and evs[1].text == "Where is my order?" and 0.85 < evs[1].confidence < 0.95
    assert seen["auth"] == "Bearer k" and b"whisper-large-v3-turbo" in seen["body"] and b"RIFF" in seen["body"]
    assert loc.s.discarded == 1 and loc.s.ended == 0          # no wasted local decode


async def test_groq_failure_falls_back_to_local_final_and_trips_breaker():
    p, loc = provider(lambda r: httpx.Response(429, json={"error": "rate"}))
    s = await p.open()
    for _ in range(3):
        evs = await utterance(s)
        assert evs[-1].text == "hello local"
    assert p.breaker.open                                      # stop calling a failing API
    assert loc.s.ended == 3


async def test_short_blip_skips_remote_call():
    called = []
    p, loc = provider(lambda r: called.append(1) or httpx.Response(200, json={"text": "x"}))
    s = await p.open()
    await s.push(np.ones(1000, dtype=np.float32)); await s.end_utterance()
    assert not called and loc.s.ended == 1
