"use client";
import { useState } from "react";
import { api, when } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { Err, Ok } from "./ui";

type Doc = { id: string; title: string; source: string; created_at: string; chunks: number };
type Hits = { latency_ms: number; hits: { content: string; score: number }[] };

export default function KnowledgeManager({ readOnly }: { readOnly?: boolean }) {
  const docs = useApi<Doc[]>("/kb/documents");
  const [busy, setBusy] = useState(""); const [err, setErr] = useState(""); const [ok, setOk] = useState("");
  const [url, setUrl] = useState(""); const [q, setQ] = useState(""); const [hits, setHits] = useState<Hits | null>(null);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setErr(""); setOk("");
    for (const f of Array.from(files)) {
      setBusy(`Indexing ${f.name}…`);
      try { const fd = new FormData(); fd.append("file", f); const r = await api<{ chunks: number }>("/kb/documents", { method: "POST", body: fd }); setOk(`${f.name}: ${r.chunks} passages indexed`); }
      catch (e) { setErr(`${f.name}: ${(e as Error).message}`); }
    }
    setBusy(""); void docs.reload();
  }
  async function crawl() {
    setErr(""); setOk(""); setBusy("Crawling…");
    try { const r = await api<{ documents: number; chunks: number }>("/kb/crawl", { method: "POST", json: { url, max_pages: 10 } }); setOk(`${r.documents} pages, ${r.chunks} passages indexed`); setUrl(""); }
    catch (e) { setErr((e as Error).message); }
    setBusy(""); void docs.reload();
  }
  async function del(id: string) { await api(`/kb/documents/${id}`, { method: "DELETE" }); void docs.reload(); }
  async function search() { setErr(""); try { setHits(await api<Hits>("/kb/search", { method: "POST", json: { query: q } })); } catch (e) { setErr((e as Error).message); } }

  return (<div className="grid" style={{ gap: 16 }}>
    <Err msg={err} /><Ok msg={ok} />
    {!readOnly && <div className="grid g2">
      <section className="card"><h2>Upload documents</h2>
        <p className="muted small">PDF, Word, text/Markdown, or a CSV with <span className="mono">question,answer</span> columns. Up to 20 MB each.</p>
        <input type="file" multiple accept=".pdf,.docx,.txt,.md,.csv" aria-label="Upload documents" onChange={(e) => { void upload(e.target.files); e.target.value = ""; }} disabled={!!busy} />
      </section>
      <section className="card"><h2>Crawl a website</h2>
        <div className="row"><input className="grow" type="url" placeholder="https://help.example.com" value={url} onChange={(e) => setUrl(e.target.value)} aria-label="Website URL" />
          <button className="btn" onClick={crawl} disabled={!!busy || !/^https?:\/\//.test(url)}>Crawl</button></div>
        <p className="muted small" style={{ marginBottom: 0 }}>Same-site pages only (up to 10). Private networks are blocked.</p></section>
    </div>}
    {busy && <div className="alert warn" role="status">{busy}</div>}
    <section className="card" style={{ padding: 0, overflowX: "auto" }}>
      <table><thead><tr><th>Document</th><th>Source</th><th>Passages</th><th>Added</th><th></th></tr></thead>
        <tbody>{docs.data?.map((d) => <tr key={d.id}><td>{d.title}</td><td className="mono">{d.source}</td><td>{d.chunks}</td><td>{when(d.created_at)}</td>
          <td>{!readOnly && <button className="btn sm danger" onClick={() => del(d.id)} aria-label={`Delete ${d.title}`}>Delete</button>}</td></tr>)}
          {docs.data?.length === 0 && <tr><td colSpan={5} className="muted" style={{ padding: 24 }}>No knowledge yet. The agent will say it doesn't know and offer a person.</td></tr>}</tbody></table>
    </section>
    {!readOnly && <section className="card"><h2>Test retrieval</h2>
      <p className="muted small">See what the agent would look up for a caller's question.</p>
      <div className="row"><input className="grow" type="text" placeholder="What is your return policy?" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && q.length > 1 && search()} aria-label="Test question" />
        <button className="btn" onClick={search} disabled={q.length < 2}>Search</button></div>
      {hits && <div style={{ marginTop: 12 }}><p className="small muted">{hits.hits.length} passages in {hits.latency_ms} ms</p>
        {hits.hits.map((h, i) => <div key={i} className="bubble" style={{ maxWidth: "100%" }}><div className="meta">relevance {h.score}</div>{h.content.slice(0, 420)}{h.content.length > 420 ? "…" : ""}</div>)}</div>}
    </section>}
  </div>);
}
