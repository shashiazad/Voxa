"""Inbound per-session pipeline: 48k PCM -> NS -> 16k -> VAD -> STT, as independent bounded stages."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from .audio.noise import NoiseSuppressor
from .audio.resample import Decimator48to16
from .audio.vad import VadEvent, VadSegmenter
from .metrics import dropped_frames, ns_vad_frame
from .providers.stt import STTEvent, STTStream

log = logging.getLogger("voxa.pipeline")


@dataclass
class VadSignal:
    event: VadEvent
    t_ms: int
    mono_s: float  # time.monotonic() when detected, for eos_detected metrics


class InboundPipeline:
    """Tasks: ns_vad (CPU-light, per 10 ms frame) -> stt_feed -> stt_read.
    Queues are bounded; on overflow the *oldest* audio is dropped so latency never accumulates."""

    def __init__(self, ns: NoiseSuppressor, vad: VadSegmenter, stt: STTStream,
                 on_vad: Callable[[VadSignal], Awaitable[None]],
                 on_stt: Callable[[STTEvent], Awaitable[None]], queue_frames: int = 50, preroll_ms: int = 320,
                 ref_level: Callable[[], float] | None = None, echo_gate_ratio: float = 0.05,
                 near_field_db: float = 9.0) -> None:
        self._ref_level, self._gate = ref_level, echo_gate_ratio
        self._nf_ratio = 10 ** (-near_field_db / 20) if near_field_db > 0 else 0.0
        self._sm, self._open = 0.0, True
        self._seg_sum, self._seg_n = 0.0, 0     # loudness of the current speech segment while the agent talks
        self.user_level = 0.0          # running RMS of the caller's own voice (learned while the agent is silent)
        self.gated_frames = 0
        self.ns, self.vad, self.stt = ns, vad, stt
        self._on_vad, self._on_stt = on_vad, on_stt
        self._dec = Decimator48to16()
        self._audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=queue_frames)
        self._stt_q: asyncio.Queue[tuple[str, np.ndarray | None]] = asyncio.Queue(maxsize=queue_frames * 10)
        self._preroll: list[np.ndarray] = []
        self._preroll_max = preroll_ms * 16  # samples @16k
        self._tasks: list[asyncio.Task[None]] = []
        self.dropped_frames = 0
        self._speaking = False
        self.frame_ms: list[float] = []  # NS+VAD processing time per frame (metric)

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._ns_vad(), name="ns_vad"),
                       asyncio.create_task(self._stt_feed(), name="stt_feed"),
                       asyncio.create_task(self._stt_read(), name="stt_read")]

    def segment_ratio(self) -> float | None:
        """Loudness of the speech segment now in progress relative to this caller's usual level (None = unknown).
        A TV or a crowd sits far below it; a caller talking over the agent does not."""
        if self.user_level <= 0 or self._seg_n < 15:
            return None
        return (self._seg_sum / self._seg_n) / self.user_level

    def feed(self, pcm48_int16: np.ndarray) -> None:
        """Non-blocking; called from the mic-ingestion loop which must never stall."""
        if self._audio_q.full():
            self._audio_q.get_nowait()
            self.dropped_frames += 1
            dropped_frames.inc()
        self._audio_q.put_nowait(pcm48_int16)

    def _enqueue_stt(self, item: tuple[str, np.ndarray | None]) -> None:
        if self._stt_q.full():
            # Shed the oldest *audio* item; control markers ("end") must never be lost.
            keep = [self._stt_q.get_nowait() for _ in range(self._stt_q.qsize())]
            for i, (kind, _) in enumerate(keep):
                if kind == "audio":
                    del keep[i]
                    self.dropped_frames += 1
                    break
            else:
                keep.pop(0)
            for k in keep:
                self._stt_q.put_nowait(k)
        self._stt_q.put_nowait(item)

    async def _ns_vad(self) -> None:
        n_loops = 0
        while True:
            frame = await self._audio_q.get()
            batch = [frame]
            # Under load frames queue up: take up to 4 (40 ms) at once. One wake-up, one NS/resample/VAD pass for the batch
            # instead of per 10 ms frame: far less Python overhead per call, and it only happens when we are behind.
            while len(batch) < 4 and not self._audio_q.empty():
                batch.append(self._audio_q.get_nowait())
            if len(batch) > 1:
                frame = np.concatenate(batch)
            n_frames = max(1, len(frame) // 480)
            alpha = 1.0 - 0.8 ** n_frames                  # the 50 ms envelope smoothing, independent of batch size
            t0 = time.perf_counter()
            clean = self.ns.process(frame)
            pcm16 = self._dec.process(clean) if len(clean) else np.zeros(0, np.float32)
            vad_in = pcm16
            if self._ref_level is not None and len(pcm16):
                ref = self._ref_level()
                rms = float(np.sqrt(np.mean(np.square(pcm16))))
                if ref <= 0.01:
                    if self.vad.last_prob > 0.7 and self.vad.speaking:     # learn how loud this caller is when talking normally
                        self.user_level = rms if self.user_level == 0 else 0.97 * self.user_level + 0.03 * rms
                elif ref > 0.02:
                    if self.vad.speaking and self.vad.last_prob > 0.5:
                        self._seg_sum += rms; self._seg_n += 1
                    elif not self.vad.speaking:
                        self._seg_sum, self._seg_n = 0.0, 0
                    # Two gates while the agent is speaking. (1) echo: far quieter than our own output. (2) near-field:
                    # far quieter than this caller's own voice, i.e. a TV, a crowd or another room, not the person on the call.
                    # The level is smoothed (~50 ms) and the gate has hysteresis so soft consonants and short dips inside a
                    # real caller's words do not chop them up; the gate follows the *envelope*, not single frames.
                    self._sm = (1 - alpha) * self._sm + alpha * rms
                    floor = max(self._gate * ref, self._nf_ratio * self.user_level)
                    if self._open:
                        self._open = self._sm >= 0.7 * floor
                    else:
                        self._open = self._sm >= floor
                    # gate onsets only: once VAD is already in a speech segment, never chop it (that would end the
                    # caller's segment mid-sentence, starve STT and resolve a real interruption as a false one)
                    if floor > 0 and not self._open and not self.vad.speaking:
                        vad_in = np.zeros_like(pcm16)
                        self.gated_frames += 1
                else:
                    self._sm, self._open = 0.0, True
            events = self.vad.push(vad_in)
            dt = (time.perf_counter() - t0) / n_frames        # per 10 ms of audio
            n_loops += 1
            if n_loops % 25 == 0:                              # sampled: histogram writes are not free in multiprocess mode
                ns_vad_frame.observe(dt)
            self.frame_ms.append(dt * 1000)
            if len(self.frame_ms) > 2000:
                del self.frame_ms[:1000]
            was_speaking = self._speaking
            for ev, t in events:
                await self._on_vad(VadSignal(ev, t, time.monotonic()))
                if ev is VadEvent.SPEECH_START:
                    # Flush pre-roll so the first phonemes (before min_speech_ms confirmed) are kept.
                    for chunk in self._preroll:
                        self._enqueue_stt(("audio", chunk))
                    self._preroll.clear()
                    self._speaking = True
            if len(pcm16):
                if self._speaking:
                    self._enqueue_stt(("audio", pcm16))  # silence never reaches STT (no hallucinations)
                else:
                    self._preroll.append(pcm16)
                    while sum(len(c) for c in self._preroll) > self._preroll_max:
                        self._preroll.pop(0)
            for ev, _t in events:
                if ev is VadEvent.SPEECH_END:
                    self._speaking = False
                    self._enqueue_stt(("end", None))

    async def _stt_feed(self) -> None:
        while True:
            kind, data = await self._stt_q.get()
            if kind == "audio":
                await self.stt.push(data)  # type: ignore[arg-type]
            else:
                await self.stt.end_utterance()

    async def _stt_read(self) -> None:
        async for ev in self.stt.events():
            await self._on_stt(ev)

    async def peek(self) -> None:
        await self.stt.peek()

    async def aclose(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.stt.aclose()
