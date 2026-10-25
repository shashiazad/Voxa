"""Headless WebRTC scenario runner: streams prerecorded audio into a real room and checks outcomes."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from livekit import rtc

from loopback import _token

FIX = Path(__file__).parents[1] / "fixtures"
SR = 48000


def load(name: str) -> np.ndarray:
    with wave.open(str(FIX / f"{name}_48k.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


class Client:
    def __init__(self) -> None:
        self.room = rtc.Room()
        self.events: list[tuple[float, str, dict]] = []
        self.agent_active: list[tuple[float, bool]] = []     # (t, nonsilent) per 10 ms frame
        self._mic_q: deque[np.ndarray] = deque()
        self._echo_q: deque[tuple[float, np.ndarray]] = deque()
        self.echo_gain = 0.0
        self.echo_delay = 0.08
        self.echo_rir = False                      # True: room impulse response + mild speaker nonlinearity
        self._echo_hist = np.zeros(0)
        self._tasks: list[asyncio.Task] = []
        self.injections: list[tuple[float, float]] = []   # wall-clock windows where the 'user' really spoke/made noise

    async def connect(self, tenant: str = "acme") -> None:
        t = _token(tenant)

        @self.room.on("data_received")
        def _d(pkt: rtc.DataPacket) -> None:
            if pkt.topic.startswith("voxa."):
                try:
                    d = json.loads(pkt.data)
                except Exception:  # noqa: BLE001
                    d = {}
                self.events.append((time.monotonic(), pkt.topic, d))

        @self.room.on("track_subscribed")
        def _ts(track, *_):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self._tasks.append(asyncio.create_task(self._listen(track)))

        await self.room.connect(t["url"], t["token"])
        self.src = rtc.AudioSource(SR, 1, queue_size_ms=200)
        await self.room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("mic", self.src),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self._tasks.append(asyncio.create_task(self._pump()))

    async def _listen(self, track) -> None:
        async for ev in rtc.AudioStream(track, sample_rate=SR, num_channels=1):
            now = time.monotonic()
            x = np.frombuffer(ev.frame.data, dtype=np.int16)
            self.agent_active.append((now, bool(np.abs(x).max() > 300)))
            if self.echo_gain:
                self._echo_q.append((now + self.echo_delay, self._echo_path(x)))

    _TAPS = [(0, 1.0), (96, 0.55), (310, 0.38), (720, 0.27), (1450, 0.2), (2600, 0.13), (4300, 0.08), (7000, 0.05)]

    def _echo_path(self, x: np.ndarray) -> np.ndarray:
        """Agent audio -> speaker -> room -> mic. With echo_rir: reflections out to ~150 ms and a soft-clipping
        loudspeaker; the gain is the *residual* coupling (what is left after the browser's AEC)."""
        if not self.echo_rir:
            return (x * self.echo_gain).astype(np.int16)
        f = x.astype(np.float64) / 32768
        f = np.tanh(2.0 * f) / 2.0                                    # loudspeaker non-linearity
        hist = np.concatenate([self._echo_hist, f])[-9000:]
        self._echo_hist = hist
        out = np.zeros(len(f))
        norm = float(np.sqrt(sum(g * g for _, g in self._TAPS)))    # unit energy gain: `echo_gain` is the true coupling
        for d, g in self._TAPS:
            seg = hist[len(hist) - len(f) - d: len(hist) - d] if len(hist) - d >= len(f) else np.zeros(len(f))
            out += g / norm * seg
        return np.clip(out * self.echo_gain * 32768, -32768, 32767).astype(np.int16)

    async def _pump(self) -> None:
        """Continuous 10 ms mic frames (silence when idle) paced on a monotonic clock."""
        nxt = time.monotonic()
        while True:
            if self._mic_q:
                f = self._mic_q.popleft().astype(np.int32)
            else:
                f = np.zeros(480, np.int32)
            while self._echo_q and self._echo_q[0][0] <= time.monotonic():
                e = self._echo_q.popleft()[1].astype(np.int32)
                f[: len(e)] += e[:480]
            await self.src.capture_frame(rtc.AudioFrame(np.clip(f, -32768, 32767).astype(np.int16).tobytes(), SR, 1, 480))
            nxt += 0.01
            await asyncio.sleep(max(0, nxt - time.monotonic()))

    def play(self, pcm: np.ndarray) -> float:
        """Queue audio for the mic; returns the (approximate) wall-clock time its first sample leaves."""
        start = time.monotonic() + len(self._mic_q) * 0.01
        self.injections.append((start, start + len(pcm) / SR))
        for i in range(0, len(pcm), 480):
            c = pcm[i:i + 480]
            self._mic_q.append(np.pad(c, (0, 480 - len(c))))
        return start

    def self_caused(self, t: float, slack: float = 1.0) -> bool:
        """True if an event at time t cannot be explained by anything the test injected into the mic."""
        return not any(a - 0.2 <= t <= b + slack for a, b in self.injections)

    def queued_s(self) -> float:
        return len(self._mic_q) * 0.01

    async def wait_mic_idle(self) -> None:
        while self._mic_q:
            await asyncio.sleep(0.02)

    def agent_speaking_since(self, within: float = 0.45) -> float:
        """Seconds the agent has been continuously (gaps <within) producing audio, ending now."""
        now, run_start, last = time.monotonic(), None, None
        for t, a in reversed(self.agent_active):
            if now - t > 8:
                break
            if not a:
                continue
            if last is None:
                if now - t > within:
                    return 0.0
                last = run_start = t
            elif last - t <= within:
                last = t
                run_start = t
            else:
                break
        return (now - run_start) if run_start else 0.0

    def last_agent_audio(self) -> float | None:
        return next((t for t, a in reversed(self.agent_active) if a), None)

    def evs(self, topic: str, since: float = 0.0):
        return [(t, d) for t, tp, d in self.events if tp == topic and t >= since]

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.room.disconnect()


async def wait_for(pred, timeout: float, step: float = 0.02) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(step)
    return False


@dataclass
class Result:
    name: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)


async def start(c: Client) -> None:
    await c.connect()
    await wait_for(lambda: c.evs("voxa.agent_audio_start"), 10)
    await wait_for(lambda: c.agent_speaking_since() == 0.0 and c.last_agent_audio() is not None and
                   time.monotonic() - c.last_agent_audio() > 0.5, 10)   # greeting done


async def get_agent_talking(c: Client) -> float:
    """Ask a question and wait until the (long) answer is audibly under way. Returns time of audio onset."""
    # a question whose grounded answer is long enough to interrupt (real LLMs answer short to "status")
    c.play(np.concatenate([np.zeros(4800, np.int16), load("q_returns"), np.zeros(48000, np.int16)]))
    ok = await wait_for(lambda: c.agent_speaking_since() > 1.2, 14)
    assert ok, "agent never started a long answer"
    return time.monotonic()


async def s_clean() -> Result:
    c = Client(); await start(c)
    t0 = time.monotonic()
    speech = load("status")
    c.play(np.concatenate([np.zeros(4800, np.int16), speech, np.zeros(48000, np.int16)]))
    speech_end = t0 + (4800 + len(speech)) / SR
    ok = await wait_for(lambda: c.last_agent_audio() and c.last_agent_audio() > speech_end, 12)
    first = next((t for t, a in c.agent_active if a and t > speech_end), None)
    finals = [d["text"] for _, d in c.evs("voxa.transcript") if d.get("final")]
    await c.close()
    v2v = (first - speech_end) * 1000 if first else None
    return Result("clean_speech", bool(ok and first), f"v2v_client={v2v and round(v2v)}ms transcripts={finals}",
                  {"v2v_ms": v2v})


async def s_pause() -> Result:
    c = Client(); await start(c)
    speech = load("pause_number")
    pcm = np.concatenate([np.zeros(4800, np.int16), speech, np.zeros(48000, np.int16)])
    t0 = c.play(pcm)
    user_done = t0 + (4800 + len(speech)) / SR
    await wait_for(lambda: bool(c.evs("voxa.agent_audio_start", t0)), 12)
    starts = c.evs("voxa.agent_audio_start", t0)
    vads = c.evs("voxa.vad", t0)
    ends = [t for t, d in vads if d["event"] == "speech_end"]
    premature = [t for t, _ in starts if t < user_done - 0.05]
    finals = [d["text"] for _, d in c.evs("voxa.transcript", t0) if d.get("final")]
    await c.close()
    return Result("mid_utterance_pause", bool(starts) and not premature,
                  f"vad_ends={len(ends)} replies={len(starts)} premature={len(premature)} finals={finals}")


async def _interrupt_with(name: str, expect_interrupt: bool, echo: bool = False) -> Result:
    c = Client(); await start(c)
    if echo:
        c.echo_gain = 0.05   # residual after browser AEC (about -26 dB); raw copy would be unrealistic
    await get_agent_talking(c)
    t_inj = c.play(load(name) if name else np.zeros(4800, np.int16))
    ev0 = time.monotonic()
    await wait_for(lambda: bool(c.evs("voxa.interrupt", ev0)) and any(
        d.get("stage") == 2 or d.get("outcome") for _, d in c.evs("voxa.interrupt", ev0)), 6)
    await asyncio.sleep(1.5)
    ints = c.evs("voxa.interrupt", ev0)
    stage2 = [(t, d) for t, d in ints if d.get("stage") == 2]
    stage1 = [(t, d) for t, d in ints if d.get("stage") == 1]
    still_talking = c.agent_speaking_since() > 0.3
    detail = f"stage1={len(stage1)} stage2={len(stage2)} outcomes={[d.get('outcome') for _, d in ints if d.get('outcome')]}"
    m: dict = {}
    if expect_interrupt:
        window = [t for t, a_ in c.agent_active if a_ and t_inj <= t <= t_inj + 3]
        stop = max([t for t in window if not stage1 or t <= stage1[0][0] + 0.3] or [t_inj])
        m["stop_after_onset_ms"] = (stop - t_inj) * 1000
        if stage1:
            m["stage1_event_after_onset_ms"] = (stage1[0][0] - t_inj) * 1000
        # after stage 1 (soft pause) no agent audio may be heard for the rest of the window
        t_end = stage2[0][0] if stage2 else t_inj + 3
        later = [t for t in window if stage1 and stage1[0][0] + 0.25 < t < t_end]
        detail += f" stop_after_onset={m['stop_after_onset_ms']:.0f}ms stage1_event_at={m.get('stage1_event_after_onset_ms', -1):.0f}ms audio_after_pause={len(later)}"
        await wait_for(lambda: bool(c.evs("voxa.turn_metrics", t_inj)), 12)
        finals = [d["text"] for _, d in c.evs("voxa.transcript", ev0) if d.get("final")]
        detail += f" new_turn_finals={finals}"
        # soft pause fires after stage1_min_speech (200 ms) + VAD/pipeline latency; audio must be gone by then
        ok = bool(stage2) and m["stop_after_onset_ms"] <= 160 + 300 and not later
    else:
        ok = not stage2
        detail += f" still_talking_after={still_talking}"
    await c.close()
    return Result(("echo" if echo else name or "silence"), ok, detail, m)


SCENARIOS = {
    "clean": s_clean,
    "pause": s_pause,
    "interrupt": lambda: _interrupt_with("interrupt", True),
    "cough": lambda: _interrupt_with("cough", False),
    "door_slam": lambda: _interrupt_with("door_slam", False),
    "backchannel": lambda: _interrupt_with("backchannel", False),
    "echo": lambda: _interrupt_with("", False, echo=True),
}


async def main(names: list[str], reps: int) -> None:
    out = []
    for n in names:
        for i in range(reps):
            try:
                r = await SCENARIOS[n]()
            except Exception as e:  # noqa: BLE001
                r = Result(n, False, f"EXC {e!r}")
            out.append(r)
            print(f"{'PASS' if r.passed else 'FAIL'} {n}#{i} {r.detail}", flush=True)
    print(json.dumps([r.__dict__ for r in out]), file=open(FIX.parent / "scenario_results.json", "w"))
    print(f"\n{sum(r.passed for r in out)}/{len(out)} passed")


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(SCENARIOS)
    asyncio.run(main(names, int(sys.argv[2]) if len(sys.argv) > 2 else 1))
