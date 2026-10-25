"use client";
import { useEffect, useState } from "react";
import { CopyButton, Err, Ok } from "@/components/ui";
import { api } from "@/lib/api";
import { useApi, useMe } from "@/lib/hooks";

export default function WidgetPage() {
  const me = useMe(); const w = useApi<{ widget_key: string; allowed_origins: string[] }>("/widget");
  const [origins, setOrigins] = useState(""); const [err, setErr] = useState(""); const [ok, setOk] = useState("");
  const [color, setColor] = useState("#4f46e5"); const [title, setTitle] = useState("Talk to us"); const [pos, setPos] = useState("right");
  useEffect(() => { if (w.data) setOrigins(w.data.allowed_origins.join("\n")); }, [w.data]);
  const admin = me?.role === "admin";
  const host = typeof location !== "undefined" ? location.origin : "https://your-console.example.com";
  const snippet = `<script src="${host}/widget.js"\n  data-key="${w.data?.widget_key ?? "wk_…"}"\n  data-title="${title}" data-color="${color}" data-position="${pos}" async></script>`;

  async function save() {
    setErr(""); setOk("");
    try { await api("/widget", { method: "PUT", json: { allowed_origins: origins.split("\n").map((s) => s.trim()).filter(Boolean) } }); setOk("Allowed sites saved."); void w.reload(); }
    catch (e) { setErr((e as Error).message); }
  }
  async function rotate() {
    if (!confirm("Rotate the key? Pages using the old snippet will stop working until updated.")) return;
    await api("/widget/rotate", { method: "POST" }); void w.reload();
  }
  function preview() {
    document.getElementById("voxa-preview")?.remove();
    const s = document.createElement("script");
    s.id = "voxa-preview"; s.src = "/widget.js"; s.dataset.key = w.data!.widget_key; s.dataset.api = "/api";
    s.dataset.title = title; s.dataset.color = color; s.dataset.position = pos; document.body.appendChild(s);
  }
  return (<>
    <div className="page-head"><div><h1>Website widget</h1><p className="muted" style={{ margin: 0 }}>Add voice support to any site with one script tag.</p></div></div>
    <Err msg={err || w.error} /><Ok msg={ok} />
    <div className="grid g2">
      <section className="card"><h2>1. Customize</h2>
        <label className="f"><span>Title</span><input type="text" value={title} maxLength={40} onChange={(e) => setTitle(e.target.value)} /></label>
        <div className="row"><label className="f grow"><span>Accent color</span><input type="color" value={color} onChange={(e) => setColor(e.target.value)} style={{ width: "100%", height: 38 }} /></label>
          <label className="f grow"><span>Position</span><select value={pos} onChange={(e) => setPos(e.target.value)}><option value="right">Bottom right</option><option value="left">Bottom left</option></select></label></div>
        <button className="btn" onClick={preview} disabled={!w.data || !w.data.allowed_origins.length}>Preview on this page</button>
        {w.data && !w.data.allowed_origins.length && <p className="small muted">Add this console's address ({host}) to the allowed sites to preview.</p>}</section>
      <section className="card"><h2>2. Allowed sites</h2>
        <p className="muted small">Only pages served from these origins can start calls. Enter one per line, e.g. <span className="mono">https://www.example.com</span>.</p>
        <textarea rows={4} value={origins} onChange={(e) => setOrigins(e.target.value)} disabled={!admin} aria-label="Allowed sites" />
        <div className="row" style={{ marginTop: 10 }}><button className="btn primary" disabled={!admin} onClick={save}>Save</button>
          <button className="btn danger" disabled={!admin} onClick={rotate}>Rotate key</button></div></section>
    </div>
    <section className="card" style={{ marginTop: 16 }}><div className="row spread"><h2 style={{ margin: 0 }}>3. Paste before &lt;/body&gt;</h2><CopyButton text={snippet} /></div>
      <pre className="snip">{snippet}</pre>
      <p className="muted small" style={{ marginBottom: 0 }}>The key is publishable. Abuse is limited by the allowed-sites list and server-side rate limits.</p></section>
  </>);
}
