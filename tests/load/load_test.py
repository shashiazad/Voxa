"""Concurrent-session load test over real WebRTC.

Run the agent(s) with  STT_PROVIDER=stub LLM_PROVIDER=mock JOB_EXECUTOR=thread  so the test measures *our* pipeline
(noise suppression, VAD, endpointing, TTS, WebRTC) rather than a cloud API's rate limits. Callers are spread over
several generator processes so the generator is not the bottleneck.

  python tests/load/load_test.py --steps 10,25,50,100 --hold 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[1] / "audio_harness")]
import scenarios as sc  # noqa: E402
from scenarios import SR, load  # noqa: E402

OUT = Path(__file__).parents[2] / "docs" / "reports" / "load.json"
FRAME = 960  # 20 ms


async def caller(idx: int, stop_at: float, q: "mp.Queue", think_s: float) -> None:
    """One simulated caller: connect, then speak a short question every `think_s` seconds, measuring voice-to-voice."""
    c = sc.Client()
    try:
        await c.connect()
    except Exception as e:  # noqa: BLE001
        q.put(("error", idx, f"connect: {e!r}")); return
    q.put(("up", idx, time.time()))
    speech = np.concatenate([np.zeros(2400, np.int16), load("status"), np.zeros(SR, np.int16)])
    end_off = (2400 + len(load("status"))) / SR
    await asyncio.sleep(np.random.uniform(0, think_s))            # de-synchronise callers
    try:
        while time.time() < stop_at:
            n_before = len(c.evs("voxa.agent_audio_start"))
            t0 = c.play(speech)
            end = t0 + end_off
            ok = await sc.wait_for(lambda: len(c.evs("voxa.agent_audio_start")) > n_before, 15)
            await asyncio.sleep(0.4)           # the audio frames trail the data-channel event by a few hundred ms
            first = next((t for t, a in c.agent_active if a and t > end - 0.05), None) if ok else None
            q.put(("v2v", idx, time.time(), (first - end) * 1000 if first else None))
            await asyncio.sleep(think_s)
    finally:
        await c.close()
        q.put(("down", idx, time.time()))


def proc_main(ids: list[int], stop_at: float, q, think_s: float) -> None:
    async def run():
        await asyncio.gather(*(caller(i, stop_at, q, think_s) for i in ids))
    asyncio.run(run())


def docker_stats() -> tuple[float, float]:
    """Sum CPU (cores) and memory (MB) across the agent containers."""
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}} {{.MemUsage}}"],
                         capture_output=True, text=True).stdout.splitlines()
    cpu = mem = 0.0
    for line in out:
        name, c, m, *_ = line.split()
        if "agent" not in name:
            continue
        cpu += float(c.strip("%")) / 100
        v, unit = m[:-3] if m.endswith("iB") else m, None
        num = float("".join(ch for ch in m if ch.isdigit() or ch == "."))
        mem += num * (1024 if "GiB" in m.split("/")[0] else 1 if "MiB" in m.split("/")[0] else 1 / 1024)
    return cpu, mem


def prom(query: str) -> float | None:
    try:
        r = json.load(urllib.request.urlopen("http://localhost:9090/api/v1/query?query=" + urllib.parse.quote(query), timeout=5))
        res = r["data"]["result"]
        return float(res[0]["value"][1]) if res else None
    except Exception:  # noqa: BLE001
        return None


def pct(a, q):
    a = sorted(x for x in a if x is not None)
    return a[min(len(a) - 1, int(q * len(a)))] if a else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="10,25,50,100")
    ap.add_argument("--hold", type=int, default=60)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--think", type=float, default=12.0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(",")]
    base_cpu, base_mem = docker_stats()
    q: mp.Queue = mp.Queue()
    results = []
    for n in steps:
        stop_at = time.time() + 40 + a.hold                       # ramp-up window + measurement window
        ids = list(range(n))
        procs = [mp.Process(target=proc_main, args=(ids[i::a.procs], stop_at, q, a.think)) for i in range(min(a.procs, n))]
        t_start = time.time()
        for p in procs:
            p.start()
        v2v, errors, up = [], [], 0
        cpu_samples, mem_samples = [], []
        hold_from = t_start + 40
        last_sample = 0.0
        while time.time() < stop_at + 20 and (any(p.is_alive() for p in procs)):
            try:
                kind, idx, *rest = q.get(timeout=1)
                if kind == "up":
                    up += 1
                elif kind == "error":
                    errors.append(rest[0])
                elif kind == "v2v" and rest[0] >= hold_from:
                    v2v.append(rest[1])
            except Exception:  # noqa: BLE001
                pass
            if time.time() >= hold_from and time.time() - last_sample > 5 and time.time() < stop_at:
                last_sample = time.time()
                c, m = docker_stats(); cpu_samples.append(c); mem_samples.append(m)
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        ok = [x for x in v2v if x is not None]
        cpu = float(np.mean(cpu_samples)) if cpu_samples else 0.0
        mem = float(np.mean(mem_samples)) if mem_samples else 0.0
        r = {"n": n, "connected": up, "connect_errors": len(errors), "turns": len(v2v), "turns_without_reply": len(v2v) - len(ok),
             "cpu_cores": cpu, "rss_mb": mem, "cpu_per_session_millicores": (cpu - base_cpu) / n * 1000 if n else None,
             "mb_per_session": (mem - base_mem) / n if n else None, "v2v_p50": pct(ok, .5), "v2v_p95": pct(ok, .95),
             "dropped": prom("sum(increase(voxa_audio_frames_dropped_total[2m]))"),
             "nsvad_p95_ms": (lambda x: x * 1000 if x else None)(prom("histogram_quantile(0.95, sum(increase(voxa_ns_vad_frame_seconds_bucket[2m])) by (le))"))}
        results.append(r)
        print(json.dumps(r), flush=True)
        time.sleep(15)                                                  # let sessions drain before the next step
    OUT.write_text(json.dumps({"meta": {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.platform(),
                                        "tag": "stub STT, mock LLM, local Piper TTS, thread executor",
                                        "note": a.note or "Generator, SFU and agents share one 10-core laptop VM, so these numbers are pessimistic. "
                                        "STT/LLM network latency and rate limits are excluded by design (stub/mock)."},
                               "baseline": {"cpu": base_cpu, "mem_mb": base_mem}, "steps": results}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
