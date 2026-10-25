"""Headless client: joins via the API token, sends data-channel pings, reports RTT percentiles."""
from __future__ import annotations

import asyncio
import json
import statistics
import struct
import time
import urllib.request

from livekit import rtc

API = "http://localhost:8000"


def _token(tenant: str) -> dict:
    import os
    key = os.environ.get("VOXA_API_KEY", "acme_dev_key_change_me")   # per-tenant API key (sample tenant)
    req = urllib.request.Request(f"{API}/v1/rooms/token", method="POST", data=b"{}",
                                 headers={"content-type": "application/json", "X-API-Key": key})
    return json.load(urllib.request.urlopen(req))


async def main(n: int = 50) -> None:
    t = _token("acme")
    room = rtc.Room()
    rtts: list[float] = []
    pending: dict[int, float] = {}

    @room.on("data_received")
    def _d(pkt: rtc.DataPacket) -> None:
        if pkt.topic == "voxa.pong":
            seq = struct.unpack("!I", pkt.data[:4])[0]
            if seq in pending:
                rtts.append((time.perf_counter() - pending.pop(seq)) * 1000)

    await room.connect(t["url"], t["token"])
    await asyncio.sleep(2)  # let the agent join
    for i in range(n):
        pending[i] = time.perf_counter()
        await room.local_participant.publish_data(struct.pack("!I", i), reliable=False, topic="voxa.ping")
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.5)
    await room.disconnect()
    if not rtts:
        raise SystemExit("no pongs received - is the agent running?")
    rtts.sort()
    print(f"data-channel RTT ms  n={len(rtts)}  p50={statistics.median(rtts):.1f}  "
          f"p95={rtts[int(len(rtts) * .95) - 1]:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
