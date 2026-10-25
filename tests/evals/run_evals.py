"""LLM-simulated caller evals. A Groq model plays a customer with a goal; its words are synthesized (Piper), sent over
real WebRTC, and the agent's spoken replies are captured from the word-synced captions. A second model judges the
transcript against ground-truth facts for task success and hallucination. Latency comes from the real calls.

  python tests/evals/run_evals.py [persona ...]
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path[:0] = [str(ROOT / "tests" / "audio_harness"), str(ROOT / "agent"), str(ROOT / "shared")]
import scenarios as sc  # noqa: E402
from scenarios import SR, Client  # noqa: E402

OUT = ROOT / "docs" / "reports" / "evals.json"
CALLER_MODEL = os.environ.get("EVAL_CALLER_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")
MOCK = "http://localhost:8100"

PERSONAS = {
    "returns": {
        "goal": "Find out whether you can return headphones you already opened, and how long you have. You are a bit unsure and ask one follow-up about refunds.",
        "facts": "Opened headphones/earbuds can be returned within 30 days of delivery if in working condition and the ear tips are included. "
                 "Refunds go to the original payment method within 5 to 7 business days after the item arrives at the warehouse.",
        "check": None,
    },
    "order_status": {
        "goal": "Ask where your order is. Your order number is A C 4 8 2 1 3 (say it as letters and digits). Ask who the carrier is.",
        "facts": "Order AC48213: status shipped, carrier UPS, estimated delivery 2026-10-28, item AirBeat Pro earbuds, tracking 1Z999AA10123456784.",
        "check": None,
    },
    "ticket": {
        "goal": "Your AirBeat Pro earbuds stopped charging. Ask the agent to open a support ticket for order A C 4 8 2 1 3. When asked to confirm details, say yes.",
        "facts": "A support ticket must be created for the broken earbuds (order AC48213) via the create_ticket tool after the caller confirms.",
        "check": "ticket",
    },
    "out_of_scope": {
        "goal": "Ask whether Acme offers a student discount and whether they do gift wrapping. Accept whatever answer you get politely.",
        "facts": "The knowledge base says NOTHING about student discounts or gift wrapping. The only price policy is a price match with authorized retailers within 14 days. "
                 "A correct agent says it does not know and offers a person; an agent that states a discount or a gift-wrap policy is hallucinating.",
        "check": None,
    },
    "human": {
        "goal": "You are frustrated. Say you want to talk to a real person about a billing dispute.",
        "facts": "The agent should hand the caller to a human (warm transfer) rather than try to resolve a billing dispute itself.",
        "check": "handoff",
    },
}


async def chat(model: str, messages: list[dict], max_tokens: int = 300, tries: int = 6) -> str:
    async with httpx.AsyncClient(timeout=40, headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]}) as c:
        for i in range(tries):
            r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                             json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6,
                                   **({"reasoning_effort": "low"} if "gpt-oss" in model else {})})
            if r.status_code == 429:
                await asyncio.sleep(8 * (i + 1)); continue
            if r.status_code >= 400:
                raise RuntimeError(f"groq {r.status_code}: {r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("rate limited")


def synth(text: str) -> np.ndarray:
    from piper import PiperVoice
    from voxa_agent.audio.resample import Upsampler16to48
    global _V
    try:
        _V
    except NameError:
        _V = PiperVoice.load(str(ROOT / "models" / "en_US-ryan-low.onnx"))
    pcm = np.concatenate([c.audio_int16_array for c in _V.synthesize(text)])
    return np.clip(Upsampler16to48().process(pcm), -32768, 32767).astype(np.int16)


def tickets() -> int:
    req = urllib.request.Request(f"{MOCK}/tickets", headers={"X-Acme-Token": "acme-dev-token"})
    return len(json.load(urllib.request.urlopen(req)))


async def run_persona(name: str, p: dict, max_turns: int = 6) -> dict:
    c = Client(); await sc.start(c)
    t_before = tickets()
    convo: list[tuple[str, str]] = []
    v2v: list[float] = []
    caller_sys = ("You are role-playing a customer on a phone call with Acme Electronics' voice assistant. "
                  f"Your goal: {p['goal']} Speak naturally in one or two SHORT sentences (under 20 words), no stage directions. "
                  "When your goal is met or the agent cannot help, say a short goodbye as your last line.")
    agent_said = ""
    for turn in range(max_turns):
        hist = [{"role": "system", "content": caller_sys}]
        for who, text in convo:
            hist.append({"role": "assistant" if who == "caller" else "user", "content": text})
        if not convo:
            hist.append({"role": "user", "content": "(The agent just greeted you. Say your first line.)"})
        line = await chat(CALLER_MODEL, hist, 80)
        line = line.strip('"')
        pcm = synth(line)
        n_metrics = len(c.evs("voxa.turn_metrics")); n_words = len(c.evs("voxa.agent_word")); n_ho = len(c.evs("voxa.handoff"))
        t0 = c.play(np.concatenate([np.zeros(4800, np.int16), pcm, np.zeros(int(1.2 * SR), np.int16)]))
        end = t0 + (4800 + len(pcm)) / SR
        convo.append(("caller", line))
        await sc.wait_for(lambda: len(c.evs("voxa.turn_metrics")) > n_metrics or len(c.evs("voxa.handoff")) > n_ho, 40)
        await asyncio.sleep(0.8)
        first = next((t for t, a in c.agent_active if a and t > end - 0.05), None)
        if first:
            v2v.append((first - end) * 1000)
        words = [d["word"] for _, d in c.evs("voxa.agent_word")][n_words:]
        agent_said = " ".join(words)
        convo.append(("agent", agent_said or "(silence)"))
        if len(c.evs("voxa.handoff")) > n_ho or any(w in line.lower() for w in ("goodbye", "bye")):
            break
        await asyncio.sleep(2)               # stay under free-tier token limits
    handoff = bool(c.evs("voxa.handoff"))
    created = tickets() - t_before
    await c.close()
    transcript = "\n".join(f"{'CALLER' if w == 'caller' else 'AGENT'}: {t}" for w, t in convo)
    verdict_raw = await chat(JUDGE_MODEL, [{"role": "system", "content":
        "You are a strict QA judge for a customer-support voice agent. Given the caller's goal, GROUND TRUTH facts, and the call transcript, "
        "decide (a) task_success: the caller's goal was met correctly, and (b) hallucination: the agent stated any concrete fact (policy, price, "
        "date, status, discount) that is NOT supported by the ground truth. Spoken numbers appear as words. Reply with ONLY JSON: "
        '{"task_success": true|false, "hallucination": true|false, "reason": "<one sentence>"}'},
        {"role": "user", "content": f"GOAL: {p['goal']}\nGROUND TRUTH: {p['facts']}\nTRANSCRIPT:\n{transcript}"}], 250)
    try:
        v = json.loads(verdict_raw[verdict_raw.index("{"):verdict_raw.rindex("}") + 1])
    except Exception:  # noqa: BLE001
        v = {"task_success": False, "hallucination": None, "reason": f"unparseable judge output: {verdict_raw[:80]}"}
    success = bool(v.get("task_success"))
    if p["check"] == "ticket":
        success = success and created == 1                      # side effect verified against the backend, not the judge
    if p["check"] == "handoff":
        success = success and handoff
    return {"persona": name, "task_success": success, "no_hallucination": v.get("hallucination") is False, "turns": len([1 for w, _ in convo if w == "caller"]),
            "v2v_p50": sorted(v2v)[len(v2v) // 2] if v2v else None, "judge": v.get("reason"), "tickets_created": created, "handoff": handoff, "transcript": transcript}


async def main(names: list[str]) -> None:
    # the handoff persona needs 24/7 availability regardless of the real clock
    base = "http://localhost:8000"
    key = os.environ.get("VOXA_API_KEY", "acme_dev_key_change_me")
    admin = {"Authorization": "Bearer " + os.environ.get("ADMIN_TOKEN", "dev-admin-token"), "Content-Type": "application/json"}
    def call(m, path, body=None, h=None):
        r = urllib.request.Request(base + path, method=m, data=json.dumps(body).encode() if body is not None else None, headers=h or {"X-API-Key": key})
        return json.load(urllib.request.urlopen(r))
    cfg = call("GET", "/v1/config")
    open_cfg = json.loads(json.dumps(cfg)); open_cfg["escalation"]["business_hours"] = None
    call("PUT", "/v1/admin/tenants/acme/config", open_cfg, admin)
    results = []
    try:
        for n in names:
            try:
                r = await run_persona(n, PERSONAS[n])
            except Exception as e:  # noqa: BLE001
                r = {"persona": n, "task_success": False, "no_hallucination": False, "turns": 0, "judge": f"ERROR {e!r}"}
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "transcript"}), flush=True)
            print(r.get("transcript", ""), "\n", flush=True)
            await asyncio.sleep(10)
    finally:
        call("PUT", "/v1/admin/tenants/acme/config", cfg, admin)
    OUT.write_text(json.dumps({"meta": {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.platform(),
                                        "tag": f"caller={CALLER_MODEL} judge={JUDGE_MODEL}",
                                        "note": "Single run per persona: a smoke-level eval, not a statistical benchmark. The LLM judge can be wrong; the ticket and handoff outcomes are verified against the backend/agent events, not the judge."},
                               "results": results}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(PERSONAS)))
