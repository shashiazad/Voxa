"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import CallPanel from "@/components/CallPanel";
import KnowledgeManager from "@/components/KnowledgeManager";
import { Err, Field, Slider } from "@/components/ui";
import VoicePicker from "@/components/VoicePicker";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { PRESETS } from "@/lib/presets";

/* eslint-disable @typescript-eslint/no-explicit-any */
const STEPS = ["Your agent", "Knowledge", "Voice", "Turn-taking", "Try it"];

export default function Onboarding() {
  const { data } = useApi<any>("/config"); const [cfg, setCfg] = useState<any>(null);
  const [step, setStep] = useState(0); const [err, setErr] = useState(""); const [saving, setSaving] = useState(false); const [preset, setPreset] = useState("Balanced");
  useEffect(() => { if (data) setCfg(structuredClone(data)); }, [data]);
  if (!cfg) return <p className="muted">Loading…</p>;
  const set = (p: string, v: unknown) => setCfg((c: any) => { const n = structuredClone(c); const k = p.split("."); let o = n; k.slice(0, -1).forEach((x) => (o = o[x])); o[k.at(-1)!] = v; return n; });

  async function next() {
    setSaving(true); setErr("");
    try {
      if (step === 3) Object.entries(PRESETS[preset]).forEach(([k, v]) => (cfg.turn[k] = v));
      if (step !== 1) { const r = await api<{ version: number }>("/config", { method: "PUT", json: cfg }); cfg.version = r.version; }
      setStep(step + 1);
    } catch (e) { setErr((e as Error).message); } finally { setSaving(false); }
  }
  return (<>
    <div className="page-head"><div><h1>Set up your voice agent</h1><p className="muted" style={{ margin: 0 }}>Five quick steps. You can change everything later.</p></div></div>
    <ol className="steps" style={{ listStyle: "none", padding: 0 }} aria-label="Progress">{STEPS.map((s, i) => <li key={s} className={`step ${i === step ? "on" : i < step ? "done" : ""}`} aria-current={i === step ? "step" : undefined}>{i < step ? "✓ " : `${i + 1}. `}{s}</li>)}</ol>
    <Err msg={err} />
    {step === 0 && <div className="card" style={{ maxWidth: 680 }}>
      <Field label="Agent name" hint="What your agent calls itself."><input type="text" value={cfg.identity.agent_name} onChange={(e) => set("identity.agent_name", e.target.value)} /></Field>
      <Field label="What does your agent do?" hint="Describe the role in a sentence or two."><textarea value={cfg.identity.persona} onChange={(e) => set("identity.persona", e.target.value)} /></Field>
      <Field label="Greeting" hint="The first thing callers hear."><textarea value={cfg.scripts.greeting} onChange={(e) => set("scripts.greeting", e.target.value)} /></Field></div>}
    {step === 1 && <><p className="muted">Upload your FAQs, policies or product docs. The agent answers only from this content.</p><KnowledgeManager /></>}
    {step === 2 && <div className="card"><VoicePicker value={cfg.identity.voice_id} rate={cfg.identity.speaking_rate} onChange={(id) => set("identity.voice_id", id)} />
      <Slider label="Speaking rate" value={cfg.identity.speaking_rate} min={0.6} max={1.6} step={0.05} unit="×" onChange={(v) => set("identity.speaking_rate", v)} /></div>}
    {step === 3 && <div className="card" style={{ maxWidth: 680 }}><h2>How quickly should it reply?</h2>
      <div className="grid g3" role="radiogroup" aria-label="Turn-taking style">{Object.keys(PRESETS).map((p) => (
        <div key={p} className="voice" role="radio" aria-checked={preset === p} tabIndex={0} onClick={() => setPreset(p)} onKeyDown={(e) => (e.key === " " || e.key === "Enter") && setPreset(p)}>
          <b>{p}</b><p className="muted small" style={{ marginBottom: 0 }}>{p === "Relaxed" ? "Waits longer. Best for people who pause a lot." : p === "Snappy" ? "Replies fast. Best for quick, simple requests." : "A natural middle ground."}</p></div>))}</div>
      <p className="muted small">Connect order lookups, ticket creation and other actions later under <Link href="/agent">Agent → Tools</Link>.</p></div>}
    {step === 4 && <><p className="muted">Say hello. Try interrupting the agent mid-sentence.</p><CallPanel compact /></>}
    <div className="row" style={{ marginTop: 20 }}>
      {step > 0 && <button className="btn" onClick={() => setStep(step - 1)}>Back</button>}
      {step < 4 && <button className="btn primary" onClick={next} disabled={saving}>{saving ? "Saving…" : step === 3 ? "Save & try it" : "Continue"}</button>}
      {step === 4 && <Link className="btn primary" href="/">Finish</Link>}
    </div>
  </>);
}
