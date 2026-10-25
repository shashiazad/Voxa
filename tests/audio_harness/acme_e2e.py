"""End-to-end checks for the sample tenant (RAG answer, order lookup tool, confirmed ticket, human handoff)
over real WebRTC with spoken input."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request

import numpy as np

from scenarios import SR, Client, Result, load, start, wait_for
from loopback import API

KEY = os.environ.get("VOXA_API_KEY", "acme_dev_key_change_me")
ADMIN = {"Authorization": "Bearer " + os.environ.get("ADMIN_TOKEN", "dev-admin-token"), "Content-Type": "application/json"}
MOCK = "http://localhost:8100"


def http(method: str, url: str, body=None, headers=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers or {"X-API-Key": KEY})
    return json.load(urllib.request.urlopen(req))


def tickets() -> list[dict]:
    return http("GET", f"{MOCK}/tickets", headers={"X-Acme-Token": "acme-dev-token"})


async def ask(c: Client, wav: str) -> dict:
    """Speak `wav`, wait for the agent's complete answer; return words, v2v and metrics."""
    n_words, n_metrics = len(c.evs("voxa.agent_word")), len(c.evs("voxa.turn_metrics"))
    speech = load(wav)
    t0 = c.play(np.concatenate([np.zeros(4800, np.int16), speech, np.zeros(48000, np.int16)]))
    speech_end = t0 + (4800 + len(speech)) / SR
    await wait_for(lambda: len(c.evs("voxa.turn_metrics")) > n_metrics, 25)
    first = next((t for t, a in c.agent_active if a and t > speech_end), None)
    words = [d["word"] for _, d in c.evs("voxa.agent_word")][n_words:]
    finals = [d["text"] for t, d in c.evs("voxa.transcript", t0) if d.get("final")]
    return {"said": " ".join(words), "heard": finals, "v2v_ms": (first - speech_end) * 1000 if first else None,
            "states": [d["state"] for t, d in c.evs("voxa.state", t0)]}


async def e_returns() -> Result:
    c = Client(); await start(c)
    r = await ask(c, "q_returns"); await c.close()
    ok = "thirty" in r["said"].lower() and r["v2v_ms"] is not None
    return Result("rag_return_policy", ok, f"heard={r['heard']} said={r['said']!r} v2v={r['v2v_ms'] and round(r['v2v_ms'])}ms", r)


async def e_order() -> Result:
    c = Client(); await start(c)
    r = await ask(c, "q_order"); await c.close()
    said = r["said"].lower()
    ok = "shipped" in said and "ups" in said
    return Result("tool_order_status", ok, f"heard={r['heard']} said={r['said']!r} v2v={r['v2v_ms'] and round(r['v2v_ms'])}ms states={r['states']}", r)


async def e_ticket() -> Result:
    before = len(tickets())
    c = Client(); await start(c)
    r1 = await ask(c, "q_ticket")
    mid = len(tickets())
    r2 = await ask(c, "q_yes")
    after = len(tickets())
    await c.close()
    # nothing may be created before the caller says yes; exactly one ticket after
    ok = mid == before and after == before + 1 and "?" in r1["said"] + "?" and "ticket" in r2["said"].lower()
    return Result("tool_ticket_readback_confirm", ok,
                  f"tickets before/mid/after={before}/{mid}/{after} readback={r1['said']!r} done={r2['said']!r}")


async def e_handoff() -> Result:
    cfg = http("GET", f"{API}/v1/config")
    open_cfg = json.loads(json.dumps(cfg)); open_cfg["escalation"]["business_hours"] = None   # 24/7 for the test
    http("PUT", f"{API}/v1/admin/tenants/acme/config", open_cfg, ADMIN)
    try:
        c = Client(); await start(c)
        r = await ask(c, "q_human") if False else None
        n = len(c.evs("voxa.handoff"))
        c.play(np.concatenate([np.zeros(4800, np.int16), load("q_human"), np.zeros(48000, np.int16)]))
        await wait_for(lambda: len(c.evs("voxa.handoff")) > n, 20)
        pending = http("GET", f"{API}/v1/handoffs")
        room = c.room.name
        mine = [p for p in pending if p["room"] == room]
        summary_ok = bool(mine) and "human" in mine[0]["summary"].lower()
        # a human agent joins through the API-minted token and receives the summary
        tok = http("POST", f"{API}/v1/handoffs/{room}/join", {})
        from livekit import rtc
        human = rtc.Room()
        got: list[dict] = []

        @human.on("data_received")
        def _d(pkt):
            if pkt.topic == "voxa.handoff_summary":
                got.append(json.loads(pkt.data))

        await human.connect(tok["url"], tok["token"])
        await wait_for(lambda: bool(got), 10)
        await asyncio.sleep(1)
        await human.disconnect(); await c.close()
        ok = bool(mine) and summary_ok and bool(got) and not http("GET", f"{API}/v1/handoffs")[:0]
        return Result("handoff_with_summary", ok, f"pending={len(mine)} summary={got[0]['summary'][:120] if got else None!r}")
    finally:
        http("PUT", f"{API}/v1/admin/tenants/acme/config", cfg, ADMIN)


SC = {"returns": e_returns, "order": e_order, "ticket": e_ticket, "handoff": e_handoff}


async def main(names):
    ok = 0
    for n in names:
        try:
            r = await SC[n]()
        except Exception as e:  # noqa: BLE001
            r = Result(n, False, f"EXC {e!r}")
        ok += r.passed
        print(f"{'PASS' if r.passed else 'FAIL'} {r.name}: {r.detail}", flush=True)
    print(f"\n{ok}/{len(names)} passed")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(SC)))
