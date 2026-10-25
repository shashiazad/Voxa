"""Headless client: play a WAV, measure client-side voice-to-voice (end of user speech -> first agent audio)."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import wave

import numpy as np
from livekit import rtc

from loopback import _token


async def run_turn(path: str, tenant: str = "acme", settle_s: float = float(__import__("os").environ.get("VOXA_SETTLE", "6"))) -> dict:
    t = _token(tenant)
    room = rtc.Room()
    events: list[tuple[float, str, dict]] = []
    state = {"speech_end": None, "first_audio": None, "gen": None}

    @room.on("data_received")
    def _d(pkt: rtc.DataPacket) -> None:
        if pkt.topic.startswith("voxa."):
            events.append((time.monotonic(), pkt.topic, json.loads(pkt.data) if pkt.data[:1] == b"{" else {}))

    async def listen(track: rtc.Track) -> None:
        async for ev in rtc.AudioStream(track, sample_rate=48000, num_channels=1):
            now = time.monotonic()
            x = np.frombuffer(ev.frame.data, dtype=np.int16)
            if state["speech_end"] and not state["first_audio"] and np.abs(x).max() > 500:
                state["first_audio"] = now
                gen = next((e[2]["gen"] for e in reversed(events) if e[1] == "voxa.agent_audio_start"), None)
                await room.local_participant.publish_data(
                    json.dumps({"gen": gen, "type": "first_audio_played"}).encode(), reliable=True, topic="voxa.ack")

    @room.on("track_subscribed")
    def _ts(track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(listen(track))

    await room.connect(t["url"], t["token"])
    src = rtc.AudioSource(48000, 1)
    await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("mic", src),
                                               rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(4)  # let the greeting finish
    with wave.open(path) as w:
        speech = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    pcm = np.concatenate([np.zeros(9600, np.int16), speech, np.zeros(96000, np.int16)])
    t0 = time.monotonic()
    speech_end_at = t0 + (9600 + len(speech)) / 48000
    for i in range(0, len(pcm), 480):
        await src.capture_frame(rtc.AudioFrame(pcm[i:i + 480].tobytes(), 48000, 1, len(pcm[i:i + 480])))
        if time.monotonic() >= speech_end_at:
            state["speech_end"] = speech_end_at
    await asyncio.sleep(settle_s)
    await room.disconnect()
    metrics = next((e[2] for e in events if e[1] == "voxa.turn_metrics"), {})
    return {"v2v_client_ms": (state["first_audio"] - speech_end_at) * 1000 if state["first_audio"] else None,
            "server": metrics,
            "transcripts": [e[2]["text"] for e in events if e[1] == "voxa.transcript" and e[2].get("final")],
            "agent_words": " ".join(e[2]["word"] for e in events if e[1] == "voxa.agent_word")}


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    res = [asyncio.run(run_turn(sys.argv[1])) for _ in range(n)]
    for r in res:
        print(json.dumps(r, indent=1))
