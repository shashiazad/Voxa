"""Streaming STT interface, providers, and a circuit-breaking fallback chain."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

log = logging.getLogger("voxa.stt")


@dataclass
class Word:
    text: str
    start: float  # seconds since utterance start
    end: float
    confidence: float


@dataclass
class STTEvent:
    text: str
    is_final: bool
    confidence: float
    words: list[Word] = field(default_factory=list)
    latency_ms: float = 0.0  # audio end -> event, for stt_latency metric


class STTStream(ABC):
    """One per session. Audio in (16 kHz float32), events out. Must stay open for the whole call."""

    @abstractmethod
    async def push(self, pcm16k: np.ndarray) -> None: ...
    @abstractmethod
    async def end_utterance(self) -> None:
        """VAD says speech ended: flush and emit a final."""
    @abstractmethod
    def events(self) -> AsyncIterator[STTEvent]: ...
    async def peek(self) -> None:
        """Optional: emit an accurate interim transcript of the current utterance now (used while deciding
        whether a noise is an interruption). Default: nothing extra."""

    @abstractmethod
    async def aclose(self) -> None: ...


class STTProvider(ABC):
    name: str

    @abstractmethod
    async def open(self, language: str = "en") -> STTStream: ...


class FasterWhisperSTT(STTProvider):
    """Local, free. Whisper is not natively streaming, so we re-decode the growing utterance
    every `partial_interval_s` for interim results and once more on end_utterance for the final.
    Model is loaded once per process (prewarm) and shared; decoding runs in a worker thread."""

    name = "faster-whisper"

    def __init__(self, model_size: str = "base.en", compute_type: str = "int8",
                 partial_interval_s: float = 0.5) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=2)
        self.partial_interval_s = partial_interval_s

    async def open(self, language: str = "en") -> STTStream:
        return _WhisperStream(self._model, language, self.partial_interval_s)


class _WhisperStream(STTStream):
    def __init__(self, model, language: str, interval: float) -> None:
        self._model, self._lang, self._interval = model, language, interval
        self._audio = np.zeros(0, dtype=np.float32)
        self._q: asyncio.Queue[STTEvent | None] = asyncio.Queue(maxsize=64)
        self._last_partial = 0.0
        self._decoding = False
        self._closed = False
        self._epoch = 0  # bumped per utterance; stale interims are discarded

    def _decode(self, audio: np.ndarray) -> STTEvent:
        t0 = time.perf_counter()
        segs, _ = self._model.transcribe(audio, language=self._lang, beam_size=1, word_timestamps=True,
                                         vad_filter=False, condition_on_previous_text=False)
        words: list[Word] = []
        for s in segs:
            for w in s.words or []:
                words.append(Word(w.word.strip(), w.start, w.end, float(w.probability)))
        text = " ".join(w.text for w in words)
        conf = float(np.mean([w.confidence for w in words])) if words else 0.0
        return STTEvent(text, False, conf, words, (time.perf_counter() - t0) * 1000)

    async def _emit(self, audio: np.ndarray, final: bool) -> None:
        epoch = self._epoch
        ev = await asyncio.to_thread(self._decode, audio)
        ev.is_final = final
        if not final and epoch != self._epoch:
            return
        if not ev.text and not final:
            return  # nothing recognised yet; do not emit empty interims
        if final or not self._q.full():  # never block the audio path on a slow consumer
            await self._q.put(ev)

    async def push(self, pcm16k: np.ndarray) -> None:
        self._audio = np.concatenate([self._audio, pcm16k])
        now = time.monotonic()
        if (len(self._audio) > 8000 and not self._decoding
                and now - self._last_partial >= self._interval):
            self._decoding, self._last_partial = True, now
            snapshot = self._audio.copy()

            async def run() -> None:
                try:
                    await self._emit(snapshot, False)
                finally:
                    self._decoding = False

            asyncio.create_task(run())

    def discard_utterance(self) -> None:
        """The caller got its final elsewhere: drop buffered audio and invalidate in-flight interims."""
        self._audio = np.zeros(0, dtype=np.float32)
        self._epoch += 1

    async def end_utterance(self) -> None:
        audio, self._audio = self._audio, np.zeros(0, dtype=np.float32)
        self._epoch += 1
        if len(audio) >= 4800:
            await self._emit(audio, True)

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    async def aclose(self) -> None:
        self._closed = True
        await self._q.put(None)


class DeepgramSTT(STTProvider):
    """Deepgram live streaming (Nova-class) over a persistent WebSocket, linear16 @ 16 kHz.
    NOTE: not exercised in CI (needs DEEPGRAM_API_KEY)."""

    name = "deepgram"

    def __init__(self, api_key: str, model: str = "nova-3") -> None:
        self._key, self._model = api_key, model

    async def open(self, language: str = "en") -> STTStream:
        import websockets

        url = ("wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
               f"&model={self._model}&language={language}&interim_results=true&punctuate=true"
               "&smart_format=true&endpointing=false")
        ws = await websockets.connect(url, additional_headers={"Authorization": f"Token {self._key}"},
                                      ping_interval=5)
        return _DeepgramStream(ws)


class _DeepgramStream(STTStream):
    def __init__(self, ws) -> None:
        self._ws = ws
        self._q: asyncio.Queue[STTEvent | None] = asyncio.Queue(maxsize=64)
        self._reader = asyncio.create_task(self._read())
        self._ka = asyncio.create_task(self._keepalive())

    async def _keepalive(self) -> None:
        while True:
            await asyncio.sleep(4)
            await self._ws.send(json.dumps({"type": "KeepAlive"}))

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                m = json.loads(raw)
                if m.get("type") != "Results":
                    continue
                alt = m["channel"]["alternatives"][0]
                words = [Word(w["word"], w["start"], w["end"], w.get("confidence", 0.0))
                         for w in alt.get("words", [])]
                await self._q.put(STTEvent(alt.get("transcript", ""), bool(m.get("is_final")),
                                           alt.get("confidence", 0.0), words))
        finally:
            await self._q.put(None)

    async def push(self, pcm16k: np.ndarray) -> None:
        await self._ws.send((np.clip(pcm16k, -1, 1) * 32767).astype("<i2").tobytes())

    async def end_utterance(self) -> None:
        await self._ws.send(json.dumps({"type": "Finalize"}))

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    async def aclose(self) -> None:
        self._reader.cancel(); self._ka.cancel()
        await self._ws.close()


class CircuitBreaker:
    def __init__(self, failures: int = 3, reset_s: float = 30.0) -> None:
        self.failures, self.reset_s = failures, reset_s
        self._n, self._opened_at = 0, 0.0

    @property
    def open(self) -> bool:
        if self._n >= self.failures and time.monotonic() - self._opened_at < self.reset_s:
            return True
        return False

    def ok(self) -> None:
        self._n = 0

    def fail(self) -> None:
        self._n += 1
        self._opened_at = time.monotonic()


class FallbackSTT(STTProvider):
    """Tries providers in order, skipping any whose breaker is open. Selection happens at open();
    mid-call switching is done by the session re-calling open() when a stream errors."""

    name = "fallback"

    def __init__(self, providers: list[STTProvider]) -> None:
        self._providers = providers
        self._breakers = {p.name: CircuitBreaker() for p in providers}

    async def open(self, language: str = "en") -> STTStream:
        last: Exception | None = None
        for p in self._providers:
            br = self._breakers[p.name]
            if br.open:
                continue
            try:
                s = await asyncio.wait_for(p.open(language), timeout=2.0)
                br.ok()
                return s
            except Exception as e:  # noqa: BLE001 - any provider failure trips the breaker
                log.warning("stt provider %s failed: %s", p.name, e)
                br.fail()
                last = e
        raise RuntimeError("all STT providers unavailable") from last


class GroqWhisperSTT(STTProvider):
    """Groq-hosted Whisper (large-v3-turbo) for FINAL transcripts: a batch call per VAD-ended utterance over a
    persistent HTTP/2 connection. Interims come from a local streaming-ish provider (`interim`), and if Groq
    errors or rate-limits, the local provider's final is used, so a call never loses a turn."""

    name = "groq-whisper"

    def __init__(self, api_key: str, interim: STTProvider, model: str = "whisper-large-v3-turbo",
                 timeout_s: float = 4.0, transport=None) -> None:
        import httpx

        self._interim, self.model = interim, model
        self._client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1", headers={"Authorization": f"Bearer {api_key}"},
            http2=transport is None, timeout=httpx.Timeout(timeout_s, connect=2.0), transport=transport)
        self.breaker = CircuitBreaker(failures=3, reset_s=30.0)

    async def warm(self) -> None:
        try:
            await self._client.get("/models")
        except Exception:  # noqa: BLE001
            pass

    async def open(self, language: str = "en") -> STTStream:
        return _GroqStream(self, await self._interim.open(language), language)

    async def transcribe(self, pcm16k: np.ndarray, language: str) -> STTEvent:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(pcm16k, -1, 1) * 32767).astype("<i2").tobytes())
        t0 = time.perf_counter()
        form = [("file", ("u.wav", buf.getvalue(), "audio/wav"))] + [
            (k, (None, v)) for k, v in (("model", self.model), ("language", language),
                                        ("response_format", "verbose_json"), ("temperature", "0"),
                                        ("timestamp_granularities[]", "word"),
                                        ("timestamp_granularities[]", "segment"))]
        r = await self._client.post("/audio/transcriptions", files=form)
        r.raise_for_status()
        j = r.json()
        words = [Word(w["word"].strip(), w["start"], w["end"], 0.0) for w in j.get("words", [])]
        segs = j.get("segments") or []
        # Whisper's avg_logprob -> a 0..1 confidence comparable to the local provider's word probabilities
        conf = float(np.exp(np.mean([s["avg_logprob"] for s in segs]))) if segs else 0.0
        for w in words:
            w.confidence = conf
        return STTEvent(j.get("text", "").strip(), True, conf, words, (time.perf_counter() - t0) * 1000)

    async def aclose(self) -> None:
        await self._client.aclose()


class _GroqStream(STTStream):
    def __init__(self, parent: GroqWhisperSTT, local: STTStream, language: str) -> None:
        self._p, self._local, self._lang = parent, local, language
        self._audio = np.zeros(0, dtype=np.float32)
        self._epoch = 0
        self._peeking = False
        self._q: asyncio.Queue[STTEvent | None] = asyncio.Queue(maxsize=64)
        self._pump_task = asyncio.create_task(self._pump())
        self._finals: asyncio.Queue[STTEvent] = asyncio.Queue()   # local finals kept as fallback

    async def _pump(self) -> None:
        """Forward local interims immediately; hold local finals until we know whether Groq answered."""
        async for ev in self._local.events():
            (self._finals if ev.is_final else self._q).put_nowait(ev)

    async def push(self, pcm16k: np.ndarray) -> None:
        self._audio = np.concatenate([self._audio, pcm16k])
        await self._local.push(pcm16k)

    async def peek(self) -> None:
        audio, epoch = self._audio.copy(), self._epoch
        if len(audio) < 8000 or self._peeking or self._p.breaker.open:
            return
        self._peeking = True
        try:
            ev = await self._p.transcribe(audio, self._lang)
            ev.is_final = False
            if epoch == self._epoch and ev.text:     # drop if the utterance ended while we were asking
                await self._q.put(ev)
        except Exception as e:  # noqa: BLE001
            log.warning("groq peek failed (%s)", type(e).__name__)
        finally:
            self._peeking = False

    async def end_utterance(self) -> None:
        audio, self._audio = self._audio, np.zeros(0, dtype=np.float32)
        self._epoch += 1
        if len(audio) < 4800:
            await self._local.end_utterance()
            return
        asyncio.create_task(self._finalize(audio))

    async def _finalize(self, audio: np.ndarray) -> None:
        got: STTEvent | None = None
        if not self._p.breaker.open:
            try:
                got = await self._p.transcribe(audio, self._lang)
                self._p.breaker.ok()
            except Exception as e:  # noqa: BLE001
                self._p.breaker.fail()
                log.warning("groq stt failed (%s); using local final", type(e).__name__)
                from ..metrics import errors
                errors.labels("stt").inc()
        if got is None:
            await self._local.end_utterance()                    # local final = the fallback
            ev = await asyncio.wait_for(self._finals.get(), 5)
            await self._q.put(ev)
            return
        # Drop the local decode for this utterance: reset its buffer without paying for a decode
        reset = getattr(self._local, "discard_utterance", None)
        if reset:
            reset()
        await self._q.put(got)

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    async def aclose(self) -> None:
        self._pump_task.cancel()
        await self._local.aclose()
        await self._q.put(None)



class StubSTT(STTProvider):
    """Capacity-test double (STT_PROVIDER=stub): no recognition, a canned final after each utterance. It exists so
    the load test measures our pipeline (NS, VAD, endpointing, TTS, WebRTC) without 100 CPU-bound Whisper decodes
    or cloud rate limits. Never selected unless explicitly configured."""

    name = "stub"

    def __init__(self, text: str = "What is your return policy for headphones?") -> None:
        self.text = text

    async def open(self, language: str = "en") -> STTStream:
        return _StubStream(self.text)


class _StubStream(STTStream):
    def __init__(self, text: str) -> None:
        self._text, self._n = text, 0
        self._q: asyncio.Queue[STTEvent | None] = asyncio.Queue(maxsize=16)

    async def push(self, pcm16k: np.ndarray) -> None:
        self._n += len(pcm16k)

    async def end_utterance(self) -> None:
        n, self._n = self._n, 0
        if n >= 8000:
            await self._q.put(STTEvent(self._text, True, 0.95, [Word(w, i * 0.25, i * 0.25 + 0.2, 0.95) for i, w in enumerate(self._text.split())]))

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    async def aclose(self) -> None:
        await self._q.put(None)
