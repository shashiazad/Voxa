"""Paced outbound audio. Never queues more than `max_ahead_ms` ahead of playout, so a flush is instant,
and a soft pause can rewind the not-yet-heard audio for lossless resume."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

import numpy as np

from ..providers.tts import WordTiming

FRAME_MS = 20
SR = 48_000
FRAME = SR * FRAME_MS // 1000
_FADE_OUT = np.linspace(1.0, 0.0, FRAME, dtype=np.float32)
_FADE_IN = _FADE_OUT[::-1].copy()


class AudioSink(Protocol):
    async def capture(self, pcm48: np.ndarray) -> None: ...
    def clear(self) -> None: ...
    @property
    def queued_ms(self) -> float: ...


@dataclass
class _Item:
    generation_id: int
    pcm: np.ndarray
    words: list[WordTiming]


def _ramp(frame: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    return (frame.astype(np.float32) * ramp).astype(np.int16)


class Playout:
    """One consumer task drains a bounded queue into the sink at real-time pace.
    Stale generation_ids are dropped at dequeue and at flush. Three independent gates:
    `hold` (preemptive generation, released on turn commit), `pause` (soft barge-in), generation match."""

    def __init__(self, sink: AudioSink, on_word: Callable[[int, WordTiming], Awaitable[None]] | None = None,
                 on_first_audio: Callable[[int], Awaitable[None]] | None = None,
                 max_ahead_ms: int = 200, queue_chunks: int = 32,
                 on_frame: Callable[[np.ndarray], None] | None = None) -> None:
        self.sink, self.max_ahead_ms = sink, max_ahead_ms
        self._on_frame = on_frame
        self._on_word, self._on_first = on_word, on_first_audio
        self._q: asyncio.Queue[_Item] = asyncio.Queue(maxsize=queue_chunks)
        self.current_gen = 0
        self._sent_ms: dict[int, float] = {}
        self._first_done: set[int] = set()
        self._pending_words: list[WordTiming] = []
        self._frames: deque[np.ndarray] = deque()      # frames of the item being played
        self._frame_gen = -1
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=20)  # last 400 ms actually sent
        self._stash: list[np.ndarray] = []              # unheard audio rewound by pause()
        self._paused = asyncio.Event(); self._paused.set()
        self._hold = asyncio.Event(); self._hold.set()
        self._fade_in_next = False
        self._task: asyncio.Task[None] | None = None
        self._idle = asyncio.Event(); self._idle.set()
        self._level = 0.0
        self._level_at = 0.0
        self._wake = asyncio.Event()      # set by resume(): frames were put back, wake the idle consumer

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="audio_out")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    # ---- producer side -------------------------------------------------
    async def enqueue(self, generation_id: int, pcm48: np.ndarray, words: list[WordTiming]) -> None:
        if generation_id != self.current_gen:
            return
        pad = (-len(pcm48)) % FRAME
        if pad:
            pcm48 = np.concatenate([pcm48, np.zeros(pad, dtype=np.int16)])
        self._idle.clear()
        await self._q.put(_Item(generation_id, pcm48, words))

    def begin_generation(self, generation_id: int, hold: bool = False) -> None:
        self.current_gen = generation_id
        self._sent_ms.setdefault(generation_id, 0.0)
        self._hold.clear() if hold else self._hold.set()

    async def wait_release(self) -> None:
        """Block until the turn is committed (used before side-effecting tool calls)."""
        await self._hold.wait()

    def release(self) -> None:
        """Turn committed: let held (preemptively generated) audio start playing."""
        self._hold.set()

    # ---- state ---------------------------------------------------------
    def played_ms(self, generation_id: int) -> float:
        """How much of this generation the client has (approximately) heard."""
        if generation_id not in self._sent_ms:
            return 0.0
        return max(0.0, self._sent_ms[generation_id] - self.sink.queued_ms)

    def level(self) -> float:
        """Recent RMS of what we are playing (0..1, ~100 ms smoothing). Zero when silent or paused. The inbound
        pipeline uses it as the reference for the echo gate: residual echo is far quieter than the signal that
        caused it, a caller talking over us is not."""
        if self.is_paused or time.monotonic() - self._level_at > 0.25:
            return 0.0
        return self._level

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    @property
    def has_audio(self) -> bool:
        return not self._idle.is_set() or self.sink.queued_ms > 0

    async def drain(self, generation_id: int, poll_s: float = 0.01) -> bool:
        """Wait until everything for this generation has been played. False if superseded."""
        while True:
            if generation_id != self.current_gen:
                return False
            if self._q.empty() and not self._frames and self._hold.is_set() and self.sink.queued_ms <= 1:
                if self._idle_ready():
                    return True
            await asyncio.sleep(poll_s)

    def _idle_ready(self) -> bool:
        return self._idle.is_set()

    # ---- barge-in controls --------------------------------------------
    async def pause(self) -> None:
        """Soft pause: stop within one frame. Audio already in the sink is clawed back (so nothing the
        user has not heard is lost) and a 20 ms fade-out closes the cut without a click."""
        if self.is_paused:
            return
        self._paused.clear()
        g = self.current_gen
        n = int(round(self.sink.queued_ms / FRAME_MS))
        unheard = [f for _g, f in list(self._history)[-n:]] if n else []
        self.sink.clear()
        if g in self._sent_ms:
            self._sent_ms[g] = max(0.0, self._sent_ms[g] - len(unheard) * FRAME_MS)
        # drop those frames from history so a second pause does not double count
        for _ in range(min(n, len(self._history))):
            self._history.pop()
        if unheard:
            await self.sink.capture(_ramp(unheard[0], _FADE_OUT))
            self._sent_ms[g] = self._sent_ms.get(g, 0.0) + FRAME_MS
            unheard = unheard[1:]
        self._stash = unheard

    def resume(self) -> None:
        """False interruption: replay the unheard remainder, fading in."""
        if not self.is_paused:
            return
        if self._stash:
            self._frames.extendleft(reversed(self._stash))
            self._stash = []
            self._fade_in_next = True
        self._paused.set()
        self._wake.set()

    def flush(self) -> float:
        """Confirmed interruption / cancel: drop everything. Returns heard ms of the cut generation."""
        gen = self.current_gen
        played = self.played_ms(gen)
        while not self._q.empty():
            self._q.get_nowait()
        self._frames.clear()
        self._stash = []
        self.sink.clear()
        self._pending_words.clear()
        self.current_gen = gen + 1_000_000   # nothing in flight can match; next begin_generation sets real id
        self._paused.set()
        self._hold.set()
        self._idle.set()
        return played

    # ---- consumer ------------------------------------------------------
    async def _run(self) -> None:
        while True:
            if not self._frames:
                if self._q.empty() and not self._stash:
                    self._idle.set()
                item = await self._next_item()    # event-driven: an idle call costs no wakeups
                if item is None:
                    continue
                if item.generation_id != self.current_gen:
                    continue
                self._pending_words.extend(item.words)
                self._frame_gen = item.generation_id
                self._frames.extend(item.pcm[i:i + FRAME] for i in range(0, len(item.pcm), FRAME))
            await self._hold.wait()
            await self._paused.wait()
            g = self._frame_gen
            if g != self.current_gen:
                self._frames.clear()
                continue
            while self.sink.queued_ms > self.max_ahead_ms:       # stay <=200 ms ahead of playout
                await asyncio.sleep(0.005)
                if not self._paused.is_set() or g != self.current_gen:
                    break
            if not self._paused.is_set() or g != self.current_gen or not self._frames:
                continue
            frame = self._frames.popleft()
            if self._fade_in_next:
                frame, self._fade_in_next = _ramp(frame, _FADE_IN), False
            await self.sink.capture(frame)
            if self._on_frame:
                self._on_frame(frame)
            rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32))))) / 32768.0
            self._level, self._level_at = 0.8 * self._level + 0.2 * rms, time.monotonic()
            self._history.append((g, frame))
            self._sent_ms[g] = self._sent_ms.get(g, 0.0) + FRAME_MS
            if g not in self._first_done:
                self._first_done.add(g)
                if self._on_first:
                    await self._on_first(g)
            await self._emit_words(g)

    async def _next_item(self) -> "_Item | None":
        """Next queued chunk, or None if resume() refilled `_frames` while we were waiting."""
        if self._frames:
            return None
        try:
            return self._q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self._wake.clear()
        getter = asyncio.ensure_future(self._q.get())
        waker = asyncio.ensure_future(self._wake.wait())
        try:
            done, pending = await asyncio.wait({getter, waker}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            getter.cancel(); waker.cancel()
            raise
        for p in pending:
            p.cancel()
        return getter.result() if getter in done else None

    async def _emit_words(self, g: int) -> None:
        now = self.played_ms(g)
        while self._pending_words and self._pending_words[0].start_ms <= now and self._on_word:
            await self._on_word(g, self._pending_words.pop(0))
        if not self._on_word:
            self._pending_words.clear()


class RtcSink:
    """AudioSink backed by livekit rtc.AudioSource (clear_queue = instant flush)."""

    def __init__(self, source) -> None:  # rtc.AudioSource
        from livekit import rtc

        self._rtc, self._src = rtc, source

    async def capture(self, pcm48: np.ndarray) -> None:
        await self._src.capture_frame(self._rtc.AudioFrame(pcm48.tobytes(), SR, 1, len(pcm48)))

    def clear(self) -> None:
        self._src.clear_queue()

    @property
    def queued_ms(self) -> float:
        return self._src.queued_duration * 1000
