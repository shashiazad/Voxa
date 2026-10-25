"""End-to-end for recording consent and session resume against the running stack.

  recording : consent 'yes' -> agent asks, records, uploads; the console API serves a signed link and the audio has content.
              consent 'no'  -> nothing is recorded.
  resume    : kill the agent process mid-call; the API re-dispatches a worker into the same room; the new agent restores the
              conversation (skips the greeting, says the reconnect line) and keeps answering.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave

import numpy as np

import scenarios as sc
from scenarios import SR, Client, load, wait_for
from loopback import API, _token

KEY = os.environ.get("VOXA_API_KEY", "acme_dev_key_change_me")
ADMIN = {"Authorization": "Bearer " + os.environ.get("ADMIN_TOKEN", "dev-admin-token"), "Content-Type": "application/json"}


def http(method, path, body=None, headers=None, raw=False):
    req = urllib.request.Request(API + path if path.startswith("/") else path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None, headers=headers or {"X-API-Key": KEY})
    data = urllib.request.urlopen(req).read()
    return data if raw else (json.loads(data) if data else None)


def set_recording(on: bool) -> dict:
    cfg = http("GET", "/v1/config")
    new = json.loads(json.dumps(cfg)); new["guardrails"]["record_calls"] = on
    http("PUT", "/v1/admin/tenants/acme/config", new, ADMIN)
    return cfg


async def say(c: Client, wav: str, settle: float = 4.0):
    n = len(c.evs("voxa.agent_audio_start"))
    c.play(np.concatenate([np.zeros(4800, np.int16), load(wav), np.zeros(SR, np.int16)]))
    await wait_for(lambda: len(c.evs("voxa.agent_audio_start")) > n, 20)
    await asyncio.sleep(settle)


async def wait_idle(c: Client, quiet: float = 1.2, timeout: float = 30):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        last = c.last_agent_audio()
        if last is not None and time.monotonic() - last > quiet:
            return
        await asyncio.sleep(0.2)


async def recording(consent: bool) -> dict:
    c = Client(); await c.connect()
    await wait_for(lambda: c.evs("voxa.agent_audio_start"), 15)
    await wait_idle(c)                       # greeting done -> the agent asks for consent next
    await wait_for(lambda: len(c.evs("voxa.agent_audio_start")) >= 2, 15)
    await wait_idle(c)
    words = " ".join(d["word"] for _, d in c.evs("voxa.agent_word"))
    asked = "record" in words.lower()
    await say(c, "q_yes" if consent else "q_no")
    await wait_idle(c)
    await say(c, "q_returns", 3)             # a real question after consent
    await wait_idle(c, 1.5, 40)
    room = c.room.name
    await c.close()
    # the job shuts down once LiveKit closes the empty room (~20 s), then mixes + uploads: poll for the result
    convs = http("GET", "/v1/conversations?limit=5")
    conv = next(x for x in convs if x["room"] == room)
    detail = http("GET", f"/v1/conversations/{conv['id']}")
    out = {"asked": asked, "consent_flag": detail["consent_recording"]}
    try:
        for _ in range(30 if consent else 12):
            try:
                link = http("GET", f"/v1/conversations/{conv['id']}/recording")
                break
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                await asyncio.sleep(3)
        else:
            raise urllib.error.HTTPError("", 404, "no recording", {}, None)
        audio = urllib.request.urlopen(link["url"]).read()
        w = wave.open(io.BytesIO(audio))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        out.update(recorded=True, seconds=round(w.getnframes() / w.getframerate(), 1), rate=w.getframerate(), peak=int(np.abs(x).max()))
    except urllib.error.HTTPError as e:
        out.update(recorded=False, status=e.code)
    return out


async def resume() -> dict:
    t = _token("acme")
    # widget-style token: resume needs the room-bound token, so go through the public widget endpoint
    wk = http("GET", "/v1/widget")["widget_key"]
    origin = http("GET", "/v1/widget")["allowed_origins"][0]
    req = urllib.request.Request(API + "/v1/widget/token", method="POST", headers={"X-Widget-Key": wk, "Origin": origin})
    tok = json.loads(urllib.request.urlopen(req).read())
    from livekit import rtc
    c = Client(); c.room = rtc.Room()
    # reuse the Client machinery but with the widget token
    import scenarios as s2
    c.events = []
    @c.room.on("data_received")
    def _d(pkt):
        if pkt.topic.startswith("voxa."):
            try: d = json.loads(pkt.data)
            except Exception: d = {}
            c.events.append((time.monotonic(), pkt.topic, d))
    @c.room.on("track_subscribed")
    def _ts(track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO: c._tasks.append(asyncio.create_task(c._listen(track)))
    await c.room.connect(tok["url"], tok["token"])
    c.src = rtc.AudioSource(SR, 1, queue_size_ms=200)
    await c.room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("mic", c.src), rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    c._tasks.append(asyncio.create_task(c._pump()))
    await wait_for(lambda: c.evs("voxa.agent_audio_start"), 20); await wait_idle(c)
    await say(c, "q_returns", 2); await wait_idle(c, 1.5, 40)
    await asyncio.sleep(2)                    # heartbeat has stored the history in Redis by now (saved after each reply)
    # the agent process dies mid-call
    subprocess.run(["docker", "compose", "-f", "../../infra/docker-compose.yml", "kill", "-s", "KILL", "agent"], cwd=os.path.dirname(__file__), check=True, capture_output=True)
    await asyncio.sleep(1)
    subprocess.run(["docker", "compose", "-f", "../../infra/docker-compose.yml", "up", "-d", "agent"], cwd=os.path.dirname(__file__), check=True, capture_output=True)
    await asyncio.sleep(1)
    # wait for a worker to register, then ask the API to resume (the widget does this automatically)
    for _ in range(60):
        out = subprocess.run(["docker", "compose", "-f", "../../infra/docker-compose.yml", "logs", "agent", "--since", "20s"], cwd=os.path.dirname(__file__), capture_output=True, text=True).stdout
        if "registered worker" in out:
            break
        await asyncio.sleep(3)
    await asyncio.sleep(8)
    n_before = len(c.evs("voxa.agent_audio_start"))
    r = urllib.request.Request(API + "/v1/widget/resume", method="POST", data=json.dumps({"token": tok["csat_token"]}).encode(), headers={"content-type": "application/json"})
    res = json.loads(urllib.request.urlopen(r).read())
    ok_dispatch = res["room"] == tok["room"]
    spoke = await wait_for(lambda: len(c.evs("voxa.agent_audio_start")) > n_before, 40)
    await wait_idle(c)
    words_after = " ".join(d["word"] for _, d in c.evs("voxa.agent_word"))[-200:]
    await say(c, "q_returns", 2); await wait_idle(c, 1.5, 40)
    answered = len(c.evs("voxa.agent_audio_start")) > n_before + 1
    await c.close()
    return {"dispatched_same_room": ok_dispatch, "agent_spoke_after_resume": spoke, "tail_of_speech": words_after, "answered_new_question": answered,
            "greeted_again": "thanks for calling" in words_after.lower()}


async def main(which: list[str]):
    if "recording" in which:
        orig = set_recording(True)
        try:
            for consent in (True, False):
                print("recording consent=", consent, json.dumps(await recording(consent)), flush=True)
        finally:
            http("PUT", "/v1/admin/tenants/acme/config", orig, ADMIN)
    if "resume" in which:
        print("resume", json.dumps(await resume()), flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["recording", "resume"]))
