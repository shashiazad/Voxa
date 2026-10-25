"use client";
import { useRouter } from "next/navigation";
import { dur, ms, when } from "@/lib/api";
import { useApi } from "@/lib/hooks";

type Conv = { id: string; room: string; started_at: string; ended_at: string | null; handoff: boolean; csat: number | null;
  user_turns: number; interruptions: number; tool_calls: number; p50_v2v_ms: number | null; summary: string | null };

export default function Calls() {
  const { data, error, loading } = useApi<Conv[]>("/conversations?limit=100"); const r = useRouter();
  return (<>
    <div className="page-head"><div><h1>Conversations</h1><p className="muted" style={{ margin: 0 }}>Transcripts, latency, interruptions, tool calls and handoffs.</p></div></div>
    {error && <div className="alert bad">{error}</div>}
    <div className="card" style={{ padding: 0, overflowX: "auto" }}>
      <table><thead><tr><th>Started</th><th>Duration</th><th>Turns</th><th>Interrupted</th><th>Tools</th><th>v2v p50</th><th>CSAT</th><th></th></tr></thead>
        <tbody>{data?.map((c) => (
          <tr key={c.id} className="click" tabIndex={0} onClick={() => r.push(`/calls/${c.id}`)} onKeyDown={(e) => e.key === "Enter" && r.push(`/calls/${c.id}`)}>
            <td>{when(c.started_at)}</td><td>{dur(c.started_at, c.ended_at)}</td><td>{c.user_turns}</td><td>{c.interruptions}</td><td>{c.tool_calls}</td>
            <td>{ms(c.p50_v2v_ms)}</td><td>{c.csat ? "★".repeat(c.csat) : "–"}</td><td>{c.handoff && <span className="badge warn">handoff</span>}</td></tr>))}
          {!loading && data?.length === 0 && <tr><td colSpan={8} className="muted" style={{ padding: 24 }}>No conversations yet.</td></tr>}</tbody></table></div>
  </>);
}
