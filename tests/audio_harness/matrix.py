"""Scenario matrix for the requirements table: noise robustness (SNR 5-20 dB), false/missed interruptions,
backchannels, mid-utterance pauses, speakerphone echo, and latency. Writes docs/reports/<mode>.json.

  python matrix.py robustness --reps 4      (run the agent with LLM_PROVIDER=mock: answer text is irrelevant here,
                                             and Groq's free-tier token limits would otherwise dominate the results)
  python matrix.py latency --n 20           (run the agent with the real LLM + STT)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import scenarios as sc
from scenarios import SR, Client, load, wait_for

OUT = Path(__file__).parents[2] / "docs" / "reports"
NOISES = ["cafe", "traffic", "tv", "keyboard"]
SNRS = [5, 10, 15, 20]
LEVELS = (20, 15, 10, 5)
STOP_BUDGET_MS = 160 + 300      # soft-pause hold + transport/jitter allowance (see docs/TUNING.md)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-9))


def active_rms(x: np.ndarray, frame: int = 960) -> float:
    """RMS over frames that carry speech (above a floor), so SNR is not diluted by the silence around it."""
    fr = [x[i:i + frame] for i in range(0, len(x) - frame, frame)]
    levels = np.array([rms(f) for f in fr])
    keep = levels > 0.25 * levels.max()
    return float(np.sqrt(np.mean(levels[keep] ** 2)))


def mix(speech: np.ndarray, noise_name: str, snr_db: float, lead_s: float = 1.5, tail_s: float = 3.0, seed: int = 0):
    noise = load(f"noise_{noise_name}").astype(np.float64)
    n = int((lead_s + tail_s) * SR) + len(speech)
    start = np.random.default_rng(seed).integers(0, len(noise) - n)
    nz = noise[start:start + n]
    gain = active_rms(speech) / (10 ** (snr_db / 20)) / rms(nz)
    sp = np.zeros(n); sp[int(lead_s * SR):int(lead_s * SR) + len(speech)] = speech
    return np.clip(sp + nz * gain, -32768, 32767).astype(np.int16), int(lead_s * SR), int(lead_s * SR) + len(speech)


def noise_only(noise_name: str, ref_speech: np.ndarray, snr_db: float, seconds: float, seed: int = 0) -> np.ndarray:
    noise = load(f"noise_{noise_name}").astype(np.float64)
    n = int(seconds * SR)
    start = np.random.default_rng(seed).integers(0, len(noise) - n)
    nz = noise[start:start + n]
    return np.clip(nz * (active_rms(ref_speech) / (10 ** (snr_db / 20)) / rms(nz)), -32768, 32767).astype(np.int16)


async def agent_replies_since(c: Client, t: float) -> list[float]:
    return [tt for tt, _ in c.evs("voxa.agent_audio_start", t)]


# ------------------------------------------------------------------ scenarios (each returns a dict of facts)
async def run_noise(noise: str, snr: int, rep: int) -> dict:
    c = Client(); await sc.start(c)
    pcm, s0, s1 = mix(load("q_returns"), noise, snr, seed=rep)
    t0 = c.play(pcm)
    speech_end = t0 + s1 / SR
    await wait_for(lambda: bool(c.evs("voxa.agent_audio_start", speech_end)), 20)
    await asyncio.sleep(0.5)
    finals = [d["text"].lower() for t, d in c.evs("voxa.transcript", t0) if d.get("final")]
    early = [t for t in await agent_replies_since(c, t0) if t < t0 + s0 / SR + 0.4]
    ok_text = any("return" in f and "polic" in f for f in finals)
    replied = bool(c.evs("voxa.agent_audio_start", speech_end))
    await c.close()
    return {"kind": "noise", "noise": noise, "snr": snr, "transcript_ok": ok_text, "replied": replied,
            "noise_only_replies": len(early), "finals": finals[:3]}


async def run_interrupt(rep: int, wav: str = "interrupt") -> dict:
    c = Client(); await sc.start(c)
    await sc.get_agent_talking(c)
    t_inj = c.play(load(wav)); ev0 = time.monotonic()
    await wait_for(lambda: any(d.get("stage") == 2 for _, d in c.evs("voxa.interrupt", ev0)), 6)
    await asyncio.sleep(1.2)
    ints = c.evs("voxa.interrupt", ev0)
    s1 = [t for t, d in ints if d.get("stage") == 1]; s2 = [t for t, d in ints if d.get("stage") == 2]
    window = [t for t, a in c.agent_active if a and t_inj <= t <= t_inj + 3]
    stop = max([t for t in window if not s1 or t <= s1[0] + 0.3] or [t_inj])
    stop_ms = (stop - t_inj) * 1000
    after = [t for t in window if s1 and s1[0] + 0.25 < t < (s2[0] if s2 else t_inj + 3)]
    await c.close()
    return {"kind": "interrupt", "confirmed": bool(s2), "stop_ms": stop_ms, "audio_after_pause": len(after),
            "missed": not s2, "late": bool(s2) and stop_ms > STOP_BUDGET_MS}


async def run_nonint(wav: str | None, rep: int, noise: str | None = None, snr: float = 0) -> dict:
    """Something that must NOT interrupt: cough, door slam, backchannel, or sustained background noise."""
    c = Client(); await sc.start(c)
    await sc.get_agent_talking(c)
    ev0 = time.monotonic()
    if noise:
        c.play(noise_only(noise, load("q_returns"), snr, 5.0, rep))      # background sound at `snr` dB below speech level
    else:
        c.play(load(wav))
    await asyncio.sleep(5.5 if noise else 3.5)
    ints = c.evs("voxa.interrupt", ev0)
    out = {"kind": "nonint", "what": noise and f"noise:{noise}" or wav, "stage1": sum(d.get("stage") == 1 for _, d in ints),
           "stage2": sum(d.get("stage") == 2 for _, d in ints), "outcomes": [d.get("outcome") for _, d in ints if d.get("outcome")]}
    await c.close()
    return out


async def run_echo(gain: float, rep: int, rir: bool = True) -> dict:
    c = Client(); c.echo_gain, c.echo_rir = gain, rir
    await sc.start(c)
    await sc.get_agent_talking(c)
    await asyncio.sleep(6)                   # let the agent speak with its echo leaking into the mic
    ints = c.evs("voxa.interrupt")
    # a confirmed interruption counts as *self-inflicted* only if nothing was injected into the mic around it
    out = {"kind": "echo", "gain_db": round(float(20 * np.log10(gain)), 1), "rir": rir,
           "stage1_self": sum(d.get("stage") == 1 and c.self_caused(t) for t, d in ints),
           "self_interrupts": sum(d.get("stage") == 2 and c.self_caused(t) for t, d in ints),
           "stage2_total": sum(d.get("stage") == 2 for _, d in ints)}
    await c.close()
    return out


async def run_pause(rep: int) -> dict:
    c = Client(); await sc.start(c)
    speech = load("pause_number")
    t0 = c.play(np.concatenate([np.zeros(4800, np.int16), speech, np.zeros(48000, np.int16)]))
    user_done = t0 + (4800 + len(speech)) / SR
    await wait_for(lambda: bool(c.evs("voxa.agent_audio_start", t0)), 15)
    starts = [t for t, _ in c.evs("voxa.agent_audio_start", t0)]
    await c.close()
    return {"kind": "pause", "premature": sum(t < user_done - 0.05 for t in starts), "replied": bool(starts)}


async def run_latency(rep: int) -> dict:
    c = Client(); await sc.start(c)
    q = ["status", "q_returns", "q_order"][rep % 3]
    speech = load(q)
    t0 = c.play(np.concatenate([np.zeros(4800, np.int16), speech, np.zeros(48000, np.int16)]))
    end = t0 + (4800 + len(speech)) / SR
    await wait_for(lambda: bool(c.evs("voxa.turn_metrics", t0)), 45)        # the reply has been fully spoken
    first = next((t for t, a in c.agent_active if a and t > end - 0.05), None)
    m = c.evs("voxa.turn_metrics", t0)
    srv = m[0][1] if m else None
    await c.close()
    return {"kind": "latency", "q": q, "v2v_ms": (first - end) * 1000 if first else None, "server": srv}


# ------------------------------------------------------------------ driver
def pct(a, q):
    a = sorted(x for x in a if x is not None)
    return a[min(len(a) - 1, int(q * len(a)))] if a else None


async def robustness(reps: int, only: set[str] | None = None, levels: tuple = LEVELS) -> dict:
    res: list[dict] = []
    run = lambda g: only is None or g in only  # noqa: E731

    async def go(coro_fn, label):
        for rep in range(reps if label.get("reps") is None else label["reps"]):
            try:
                r = await coro_fn(rep)
            except Exception as e:  # noqa: BLE001
                r = {"kind": label["kind"], "error": repr(e)}
            r["scenario"] = label["name"]
            res.append(r)
            print(f"{label['name']}#{rep}: {json.dumps({k: v for k, v in r.items() if k not in ('scenario', 'kind', 'finals')})}", flush=True)

    if run('interrupt'):
        await go(lambda i: run_interrupt(i), {"name": "interrupt", "kind": "interrupt"})
    if run("nonint"):
        await go(lambda i: run_nonint("cough", i), {"name": "cough", "kind": "nonint"})
        await go(lambda i: run_nonint("door_slam", i), {"name": "door_slam", "kind": "nonint"})
        await go(lambda i: run_nonint("backchannel", i), {"name": "backchannel", "kind": "nonint"})
    if run("noise_speaking"):
        for nz in NOISES:
            for level in levels:    # dB below the caller's level: 20 = background in a room (typical), the rest = stress
                await go(lambda i, nz=nz, level=level: run_nonint(None, i, nz, level),
                         {"name": f"noise_while_speaking:{nz}@{level}dB", "kind": "nonint", "reps": max(1, reps // 2)})
    if run("pause"):
        await go(lambda i: run_pause(i), {"name": "mid_utterance_pause", "kind": "pause", "reps": max(2, reps // 2)})
    if run("noisy"):
        for nz in NOISES:
            for snr in SNRS:
                await go(lambda i, nz=nz, snr=snr: run_noise(nz, snr, i), {"name": f"noisy_speech:{nz}@{snr}dB", "kind": "noise", "reps": max(1, reps // 3)})
    if run("echo"):
        await go(lambda i: run_echo(10 ** (-30 / 20), i), {"name": "speakerphone_echo_residual_-30dB", "kind": "echo"})
        await go(lambda i: run_echo(10 ** (-12 / 20), i), {"name": "speakerphone_echo_raw_-12dB(no_AEC)", "kind": "echo", "reps": max(2, reps // 2)})
    return {"mode": "robustness", "results": res}


async def latency(n: int, gap: float = 12.0) -> dict:
    res = []
    for i in range(n):
        try:
            r = await run_latency(i)
        except Exception as e:  # noqa: BLE001
            r = {"kind": "latency", "error": repr(e)}
        res.append(r)
        print(f"latency#{i}: {r.get('v2v_ms') and round(r['v2v_ms'])} ms llm={(r.get('server') or {}).get('llm')} server={ {k: round(v) if isinstance(v, float) else v for k, v in (r.get('server') or {}).items()} }", flush=True)
        await asyncio.sleep(gap)    # pace turns so the free-tier token limit does not force a fallback model
    return {"mode": "latency", "results": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["robustness", "latency"])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gap", type=float, default=12.0, help="seconds between latency turns")
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="comma list of groups: interrupt,nonint,noise_speaking,pause,noisy,echo")
    ap.add_argument("--levels", default="", help="noise_speaking levels in dB below the caller, e.g. 20,10")
    ap.add_argument("--name", default="", help="output file stem (default: the mode)")
    ap.add_argument("--merge", action="store_true", help="replace matching scenarios in the existing results file")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    levels = tuple(int(x) for x in a.levels.split(",")) if a.levels else LEVELS
    out = asyncio.run(robustness(a.reps, only, levels) if a.mode == "robustness" else latency(a.n, a.gap))
    if a.merge and (OUT / f"{a.name or a.mode}.json").exists():
        old = json.loads((OUT / f"{a.name or a.mode}.json").read_text())
        names = {r["scenario"] for r in out["results"]}
        out["results"] = [r for r in old["results"] if r.get("scenario") not in names] + out["results"]
    out["meta"] = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.platform(), "tag": a.tag}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name or a.mode}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / (a.mode + '.json')}")


if __name__ == "__main__":
    main()
