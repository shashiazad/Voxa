"use client";
import Link from "next/link";
import { use, useEffect, useState } from "react";
import { api, ms, when } from "@/lib/api";
import { useApi, useMe } from "@/lib/hooks";

type Turn = { idx: number; role: string; text: string; interrupted: boolean; created_at: string;
  metrics: { latency?: Record<string, number | null> | null; tool?: string; duration_ms?: number } | null };
type Detail = { id: string; room: string; started_at: string; ended_at: string | null; handoff: boolean; csat: number | null; summary: string | null; consent_recording?: boolean; turns: Turn[] };

export default function CallDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data, error } = useApi<Detail>(`/conversations/${id}`);
  const me = useMe(); const [rec, setRec] = useState<string | null>(null); const [msg, setMsg] = useState("");
  useEffect(() => { if (data?.consent_recording && me && me.role !== "viewer") api<{ url: string }>(`/conversations/${id}/recording`).then((r) => setRec(r.url)).catch(() => {}); }, [data, me, id]);
  async function exportJson() {
    const d = await api<unknown>(`/conversations/${id}/export`);
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(d, null, 2)], { type: "application/json" })); a.download = `conversation-${id}.json`; a.click();
  }
  async function erase() {
    if (!confirm("Permanently delete this conversation, its transcript and any recording? This cannot be undone.")) return;
    try { await api(`/conversations/${id}`, { method: "DELETE" }); location.href = "/calls"; } catch (e) { setMsg((e as Error).message); }
  }
  return (<>
    <p><Link href="/calls">← Conversations</Link></p>
    <div className="page-head"><div><h1>Conversation</h1>
      <p className="muted" style={{ margin: 0 }}>{data ? when(data.started_at) : ""} · <span className="mono">{data?.room}</span></p></div>
      <div className="row">{data?.handoff && <span className="badge warn">handed off</span>}{data?.csat && <span className="badge ok">{"★".repeat(data.csat)}</span>}
        {data?.consent_recording && <span className="badge acc">recorded with consent</span>}
        {me?.role === "admin" && <><button className="btn sm" onClick={exportJson}>Export JSON</button><button className="btn sm danger" onClick={erase}>Delete</button></>}</div></div>
    {msg && <div className="alert bad">{msg}</div>}
    {rec && <div className="card" style={{ marginBottom: 16 }}><h3>Recording</h3><audio controls src={rec} style={{ width: "100%" }} /><p className="muted small" style={{ marginBottom: 0 }}>Link expires in 5 minutes.</p></div>}
    {error && <div className="alert bad">{error}</div>}
    {data?.summary && <div className="card" style={{ marginBottom: 16 }}><h3>Handoff summary</h3><p style={{ margin: 0 }}>{data.summary}</p></div>}
    <div className="card">
      {data?.turns.length === 0 && <p className="muted">No turns were recorded.</p>}
      {data?.turns.map((t) => t.role === "tool" ? (
        <div key={t.idx} className="bubble tool">🔧 {t.text}{t.metrics?.duration_ms != null && <span className="muted"> · {t.metrics.duration_ms} ms</span>}</div>
      ) : (
        <div key={t.idx} className={`bubble ${t.role === "user" ? "user" : "agent"}${t.interrupted ? " cut" : ""}`}>
          <div className="meta">{t.role === "user" ? "Caller" : "Agent"}{t.interrupted && " · interrupted by caller"}</div>
          {t.text}
          {t.role === "assistant" && t.metrics?.latency && (
            <div className="row small" style={{ gap: 6, marginTop: 6 }}>
              <span className="badge acc" title="end of speech → first audio">v2v {ms(t.metrics.latency.voice_to_voice_ms)}</span>
              <span className="badge">stt {ms(t.metrics.latency.stt_ms)}</span><span className="badge">llm {ms(t.metrics.latency.llm_ttft_ms)}</span>
              <span className="badge">tts {ms(t.metrics.latency.tts_ttfb_ms)}</span></div>)}
        </div>))}
    </div>
  </>);
}
