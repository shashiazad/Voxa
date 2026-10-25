"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { VoxaCall, type CallSnapshot } from "@voxa/client";
import { api, ms } from "@/lib/api";
import { Err } from "./ui";

const LABEL: Record<string, string> = {
  idle: "Ready", connecting: "Connecting…", listening: "Listening", thinking: "Thinking…", speaking: "Speaking",
  paused: "Paused (possible interruption)", handoff: "Handing off to a person", reconnecting: "Reconnecting…", ended: "Call ended", error: "Error",
};
const pct = (a: number[], q: number) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

/** Test call with a live latency overlay. Reused by the onboarding wizard. */
export default function CallPanel({ compact = false }: { compact?: boolean }) {
  const call = useMemo(() => new VoxaCall(), []);
  const [s, setS] = useState<CallSnapshot>(call.snapshot);
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const [timeline, setTimeline] = useState<{ t: number; state: string }[]>([]);
  const t0 = useRef(0); const capsEnd = useRef<HTMLDivElement>(null);
  useEffect(() => call.subscribe(setS), [call]);
  useEffect(() => () => { void call.end(); }, [call]);
  useEffect(() => { capsEnd.current?.scrollIntoView({ block: "end" }); }, [s.captions]);
  useEffect(() => {
    if (!s.serverState) return;
    setTimeline((t) => (t.at(-1)?.state === s.serverState ? t : [...t, { t: Math.round(performance.now() - t0.current), state: s.serverState }].slice(-12)));
  }, [s.serverState]);

  const active = !["idle", "ended", "error"].includes(s.state);
  async function start() {
    setBusy(true); setErr("");
    try {
      const mic = await VoxaCall.checkMic();
      if (mic === "denied") throw new Error("Microphone access is blocked for this site. Allow it in the browser's site settings and retry.");
      const creds = await api<{ url: string; token: string; room: string }>("/rooms/token", { method: "POST" });
      t0.current = performance.now(); setTimeline([]);
      await call.start(creds);
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }

  const v2v = s.metrics.map((m) => m.voice_to_voice_ms).filter((x): x is number => x != null);
  const heard = s.metrics.map((m) => m.client_heard_ms).filter((x): x is number => x != null);
  const last = s.metrics.at(-1);
  const parts = last ? [
    ["Speech-to-text", last.stt_ms, "#6366f1"], ["LLM first token", last.llm_ttft_ms, "#06b6d4"], ["Text-to-speech", last.tts_ttfb_ms, "#f59e0b"],
  ] as const : [];
  const total = parts.reduce((a, [, v]) => a + (v ?? 0), 0) || 1;

  return (
    <div className={compact ? "" : "grid g2"} style={{ alignItems: "start" }}>
      <section className="card" aria-label="Call">
        <div style={{ textAlign: "center", padding: "10px 0 6px" }}>
          <div className={`orb ${s.state}`} style={{ ["--lvl" as string]: s.state === "speaking" ? s.agentLevel.toFixed(2) : 0 }} />
          <p role="status" style={{ fontWeight: 600, marginBottom: 2 }}>{LABEL[s.state]}{s.muted ? " · mic muted" : ""}</p>
          <p className="muted small" style={{ marginTop: 0 }}>{active ? "Speak naturally. You can interrupt at any time." : "Uses your microphone. Echo cancellation is on; headphones give the best results."}</p>
        </div>
        <Err msg={err || s.error} />
        <div className="row" style={{ justifyContent: "center" }}>
          {!active && <button className="btn primary" disabled={busy} onClick={start}>{s.state === "ended" || s.state === "error" ? "Start a new call" : "Start test call"}</button>}
          {active && <>
            <button className="btn" aria-pressed={s.muted} onClick={() => call.setMuted(!s.muted)}>{s.muted ? "Unmute" : "Mute"}</button>
            <button className="btn danger" onClick={() => call.end()}>End call</button></>}
        </div>
        <h3 style={{ marginTop: 18 }}>Live transcript</h3>
        <div role="log" aria-live="polite" style={{ height: compact ? 160 : 260, overflowY: "auto", paddingRight: 4 }}>
          {s.captions.length === 0 && <p className="muted small">The conversation appears here as it happens.</p>}
          {s.captions.map((c, i) => (
            <div key={i} className={`bubble ${c.role}${c.final ? "" : " partial"}${c.text.endsWith("…") ? " cut" : ""}`}>
              <div className="meta">{c.role === "user" ? "You" : "Agent"}</div>{c.text}
            </div>))}
          <div ref={capsEnd} />
        </div>
      </section>

      <section className="card" aria-label="Latency">
        <div className="row spread"><h2 style={{ margin: 0 }}>Latency</h2><span className="badge">{s.serverState || "—"}</span></div>
        <p className="muted small">Voice-to-voice = from end of your speech to the first agent audio. Target p50 ≤ 800 ms, p95 ≤ 1,200 ms.</p>
        <div className="grid g3" style={{ marginBottom: 14 }}>
          <div><div className="muted small">Last turn (server)</div><div style={{ fontSize: 22, fontWeight: 700 }}>{ms(last?.voice_to_voice_ms)}</div></div>
          <div><div className="muted small">Heard in browser</div><div style={{ fontSize: 22, fontWeight: 700 }}>{ms(last?.client_heard_ms)}</div></div>
          <div><div className="muted small">p50 / p95 ({v2v.length})</div><div style={{ fontSize: 22, fontWeight: 700 }}>{pct(v2v, .5) != null ? `${Math.round(pct(v2v, .5)!)} / ${Math.round(pct(v2v, .95)!)}` : "–"}</div></div>
        </div>
        {last ? <>
          <div className="bar" role="img" aria-label="Latency breakdown">{parts.map(([n, v, c]) => <i key={n} style={{ width: `${((v ?? 0) / total) * 100}%`, background: c }} title={`${n}: ${ms(v)}`} />)}</div>
          <div className="row small" style={{ marginTop: 8 }}>{parts.map(([n, v, c]) => <span key={n}><b style={{ color: c }}>■</b> {n} {ms(v)}</span>)}</div>
        </> : <p className="muted small">Metrics appear after the agent's first full reply.</p>}
        {heard.length > 0 && <p className="muted small">Browser-heard median: {Math.round(pct(heard, .5)!)} ms (includes network + jitter buffer).</p>}
        <h3 style={{ marginTop: 16 }}>State timeline</h3>
        <div className="row small" style={{ gap: 6 }}>
          {timeline.length === 0 ? <span className="muted">–</span> : timeline.map((t, i) => <span key={i} className="badge acc" title={`${t.t} ms`}>{t.state.replace("_", " ").toLowerCase()}</span>)}
        </div>
        {s.metrics.length > 0 && <details style={{ marginTop: 14 }}><summary className="small">Per-turn metrics</summary>
          <table className="small"><thead><tr><th>#</th><th>v2v</th><th>stt</th><th>llm</th><th>tts</th><th>heard</th></tr></thead>
            <tbody>{s.metrics.map((m) => <tr key={m.gen}><td>{m.gen}</td><td>{ms(m.voice_to_voice_ms)}</td><td>{ms(m.stt_ms)}</td><td>{ms(m.llm_ttft_ms)}</td><td>{ms(m.tts_ttfb_ms)}</td><td>{ms(m.client_heard_ms)}</td></tr>)}</tbody></table></details>}
      </section>
    </div>
  );
}
