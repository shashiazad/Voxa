"""Headless client: stream a WAV (48 kHz mono int16) over WebRTC; print VAD/transcript events with timing."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import wave

import numpy as np
from livekit import rtc

from loopback import _token


async def main(path: str) -> None:
    t = _token("acme")
    room = rtc.Room()
    t0 = time.perf_counter()
    log: list[tuple[float, str, dict]] = []

    @room.on("data_received")
    def _d(pkt: rtc.DataPacket) -> None:
        if pkt.topic in ("voxa.vad", "voxa.transcript"):
            log.append((time.perf_counter() - t0, pkt.topic, json.loads(pkt.data)))

    await room.connect(t["url"], t["token"])
    src = rtc.AudioSource(48000, 1)
    await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("mic", src),
                                               rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(2)
    with wave.open(path) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    pcm = np.concatenate([np.zeros(24000, np.int16), pcm, np.zeros(72000, np.int16)])
    t_start = time.perf_counter()
    for i in range(0, len(pcm), 480):
        f = rtc.AudioFrame(pcm[i:i + 480].tobytes(), 48000, 1, len(pcm[i:i + 480]))
        await src.capture_frame(f)
    await asyncio.sleep(3)
    await room.disconnect()
    for ts, topic, d in log:
        print(f"{ts - (t_start - t0):6.2f}s {topic:16s} {d}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
