"use client";
import Link from "next/link";
import { useState } from "react";
import { Stat } from "@/components/ui";
import { ms, when } from "@/lib/api";
import { useApi } from "@/lib/hooks";

type Summary = { calls: number; handoffs: number; csat_avg: number | null; v2v_p50_ms: number | null; v2v_p95_ms: number | null;
  v2v_samples: number; interrupted_replies: number; replies: number; tool_calls: number; series: { day: string; calls: number }[] };
type Conv = { id: string; started_at: string; handoff: boolean; csat: number | null; user_turns: number; p50_v2v_ms: number | null };

export default function Overview() {
  const [days, setDays] = useState(7);
  const s = useApi<Summary>(`/analytics/summary?days=${days}`, [days]);
  const c = useApi<Conv[]>("/conversations?limit=6");
  const d = s.data; const max = Math.max(1, ...(d?.series.map((x) => x.calls) ?? [1]));
  const slo = d?.v2v_p95_ms != null ? (d.v2v_p95_ms <= 1200 ? "ok" : "bad") : "";
  return (<>
    <div className="page-head"><div><h1>Overview</h1><p className="muted" style={{ margin: 0 }}>How your voice agent is doing.</p></div>
      <select aria-label="Time range" value={days} onChange={(e) => setDays(Number(e.target.value))} style={{ width: "auto" }}>
        <option value={1}>Last 24 hours</option><option value={7}>Last 7 days</option><option value={30}>Last 30 days</option></select></div>
    {s.error && <div className="alert bad">{s.error}</div>}
    <div className="grid g4">
      <Stat label="Calls" value={d?.calls ?? "–"} sub={d ? `${d.handoffs} handed off` : ""} />
      <Stat label="Voice-to-voice p50" value={ms(d?.v2v_p50_ms)} sub={d ? `${d.v2v_samples} replies measured` : ""} />
      <Stat label="Voice-to-voice p95" value={<span className={`badge ${slo}`} style={{ fontSize: 20 }}>{ms(d?.v2v_p95_ms)}</span>} sub="target ≤ 1,200 ms" />
      <Stat label="CSAT" value={d?.csat_avg != null ? `${d.csat_avg.toFixed(1)} / 5` : "–"} sub={d ? `${d.tool_calls} tool calls` : ""} />
    </div>
    <div className="grid g2" style={{ marginTop: 16 }}>
      <section className="card"><h2>Calls per day</h2>
        {d && d.series.length === 0 ? <p className="muted">No calls yet. <Link href="/test">Make a test call</Link>.</p> :
          <div role="img" aria-label="Calls per day" style={{ display: "flex", alignItems: "flex-end", gap: 6, height: 120 }}>
            {d?.series.map((x) => <div key={x.day} title={`${x.day}: ${x.calls}`} style={{ flex: 1, background: "var(--accent)", height: `${(x.calls / max) * 100}%`, minHeight: 4, borderRadius: 4 }} />)}</div>}
        <p className="muted small" style={{ marginBottom: 0 }}>Interrupted replies: {d ? `${d.interrupted_replies} of ${d.replies}` : "–"}</p></section>
      <section className="card"><div className="row spread"><h2>Recent conversations</h2><Link href="/calls">View all</Link></div>
        <table><tbody>{c.data?.map((x) => <tr key={x.id} className="click" onClick={() => (location.href = `/calls/${x.id}`)}>
          <td>{when(x.started_at)}</td><td>{x.user_turns} turns</td><td>{x.handoff && <span className="badge warn">handoff</span>}</td><td>{ms(x.p50_v2v_ms)}</td></tr>)}
          {c.data?.length === 0 && <tr><td className="muted">Nothing yet.</td></tr>}</tbody></table></section>
    </div>
  </>);
}
