"use client";
import { useEffect, useRef, useState } from "react";
import { HumanCall } from "@voxa/client";
import { Err } from "@/components/ui";
import { api, when } from "@/lib/api";
import { canEdit, useApi, useMe } from "@/lib/hooks";

type H = { room: string; reason: string; summary: string; ts: number };

export default function Handoffs() {
  const me = useMe(); const list = useApi<H[]>("/handoffs");
  const [live, setLive] = useState<{ room: string; summary: string; muted: boolean } | null>(null); const [err, setErr] = useState("");
  const call = useRef<HumanCall | null>(null);
  useEffect(() => { const t = setInterval(() => { if (!live) void list.reload(); }, 3000); return () => clearInterval(t); }, [live, list]);
  useEffect(() => () => { void call.current?.leave(); }, []);

  async function take(h: H) {
    setErr("");
    try {
      const creds = await api<{ url: string; token: string }>(`/handoffs/${h.room}/join`, { method: "POST" });
      call.current = new HumanCall();
      setLive({ room: h.room, summary: h.summary, muted: false });
      await call.current.join(creds, (s) => setLive((l) => (l ? { ...l, summary: s } : l)), () => { setLive(null); void list.reload(); });
    } catch (e) { setErr((e as Error).message); setLive(null); }
  }
  return (<>
    <div className="page-head"><div><h1>Handoffs</h1><p className="muted" style={{ margin: 0 }}>Callers waiting for a person. Taking a call joins their room with an auto-generated summary.</p></div></div>
    <Err msg={err} />
    {live && <section className="card" style={{ marginBottom: 16, borderColor: "var(--ok)" }}>
      <div className="row spread"><h2 style={{ margin: 0 }}>On a call <span className="badge ok">live</span></h2>
        <div className="row"><button className="btn" onClick={async () => { await call.current?.setMuted(!live.muted); setLive({ ...live, muted: !live.muted }); }}>{live.muted ? "Unmute" : "Mute"}</button>
          <button className="btn danger" onClick={() => call.current?.leave()}>Hang up</button></div></div>
      <h3 style={{ marginTop: 12 }}>Summary</h3><p style={{ marginTop: 0 }}>{live.summary}</p></section>}
    <div className="grid">
      {list.data?.length === 0 && <div className="card muted">No callers are waiting.</div>}
      {list.data?.map((h) => (
        <section key={h.room} className="card"><div className="row spread"><div className="row"><span className="badge warn">{h.reason.replace("_", " ")}</span>
          <span className="muted small">{when(new Date(h.ts * 1000).toISOString())}</span></div>
          <button className="btn primary" disabled={!canEdit(me?.role) || !!live} onClick={() => take(h)}>Take call</button></div>
          <p style={{ marginBottom: 0 }}>{h.summary}</p></section>))}
    </div>
  </>);
}
