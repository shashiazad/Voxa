"use client";
import { useEffect, useState } from "react";
import { Err, Field, Lines, Ok, Slider } from "@/components/ui";
import VoicePicker from "@/components/VoicePicker";
import { api } from "@/lib/api";
import { canEdit, useApi, useMe } from "@/lib/hooks";
import { PRESETS } from "@/lib/presets";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Cfg = any;
const TABS = ["Identity", "Voice", "Scripts", "Turn-taking", "Escalation", "Tools", "Guardrails", "Prompt"] as const;
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const TOOL_TEMPLATE = { name: "my_tool", description: "What this tool does and when to use it.", kind: "webhook", method: "GET",
  url: "https://api.example.com/items/{item_id}", headers: { Authorization: "Bearer {{secret:my_api_token}}" },
  parameters: { type: "object", properties: { item_id: { type: "string", description: "The item id" } }, required: ["item_id"] },
  side_effect: false, timeout_s: 3, expected_latency_ms: 300 };

export default function AgentPage() {
  const me = useMe(); const ro = !!me && !canEdit(me.role);
  const { data, error, reload } = useApi<Cfg>("/config");
  const [cfg, setCfg] = useState<Cfg | null>(null); const [tab, setTab] = useState<(typeof TABS)[number]>("Identity");
  const [toolsText, setToolsText] = useState<string[]>([]); const [err, setErr] = useState(""); const [ok, setOk] = useState(""); const [saving, setSaving] = useState(false);
  const secrets = useApi<string[]>("/secrets"); const [sec, setSec] = useState({ name: "", value: "" });
  useEffect(() => { if (data) { setCfg(structuredClone(data)); setToolsText(data.tools.map((t: unknown) => JSON.stringify(t, null, 2))); } }, [data]);
  if (!cfg) return <>{error ? <div className="alert bad">{error}</div> : <p className="muted">Loading…</p>}</>;
  const set = (path: string, v: unknown) => setCfg((c: Cfg) => {
    const n = structuredClone(c); const ks = path.split("."); let o = n; ks.slice(0, -1).forEach((k) => (o = o[k])); o[ks.at(-1)!] = v; return n;
  });
  const turn = cfg.turn; const esc = cfg.escalation; const bh = esc.business_hours;

  async function save() {
    setErr(""); setOk(""); setSaving(true);
    try {
      const tools = toolsText.map((t, i) => { try { return JSON.parse(t); } catch (e) { throw new Error(`Tool ${i + 1} is not valid JSON: ${(e as Error).message}`); } });
      await api("/config", { method: "PUT", json: { ...cfg, tools } });
      setOk("Saved. New calls use this configuration."); void reload();
    } catch (e) { setErr((e as Error).message); } finally { setSaving(false); }
  }
  async function addSecret() {
    setErr("");
    try { await api(`/secrets/${sec.name}`, { method: "PUT", json: { value: sec.value } }); setSec({ name: "", value: "" }); void secrets.reload(); setOk("Secret stored (encrypted)."); }
    catch (e) { setErr((e as Error).message); }
  }
  const T = (label: string, path: string, hint?: string, area = false) => (
    <Field label={label} hint={hint}>{area
      ? <textarea disabled={ro} value={path.split(".").reduce((o: any, k) => o[k], cfg) ?? ""} onChange={(e) => set(path, e.target.value)} />
      : <input type="text" disabled={ro} value={path.split(".").reduce((o: any, k) => o[k], cfg) ?? ""} onChange={(e) => set(path, e.target.value)} />}</Field>);

  return (<>
    <div className="page-head"><div><h1>Agent</h1><p className="muted" style={{ margin: 0 }}>Version {cfg.version} · changes apply to the next call.</p></div>
      <button className="btn primary" onClick={save} disabled={ro || saving}>{saving ? "Saving…" : "Save changes"}</button></div>
    {ro && <div className="alert warn">You have read-only access.</div>}
    <Err msg={err} /><Ok msg={ok} />
    <div className="tabs" role="tablist">{TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>)}</div>

    {tab === "Identity" && <div className="card" style={{ maxWidth: 720 }}>
      {T("Organization name", "identity.org_name")}{T("Agent name", "identity.agent_name", "What the agent calls itself.")}
      {T("Persona", "identity.persona", "Who the agent is and how it should behave.", true)}{T("Tone", "identity.tone", "e.g. warm, concise, professional")}
      <Field label="Languages" hint="Comma-separated language codes, e.g. en, es."><input type="text" disabled={ro} value={cfg.identity.languages.join(", ")}
        onChange={(e) => set("identity.languages", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))} /></Field></div>}

    {tab === "Voice" && <div className="card"><VoicePicker value={cfg.identity.voice_id} rate={cfg.identity.speaking_rate} disabled={ro} onChange={(id) => set("identity.voice_id", id)} />
      <Slider label="Speaking rate" value={cfg.identity.speaking_rate} min={0.6} max={1.6} step={0.05} unit="×" disabled={ro} onChange={(v) => set("identity.speaking_rate", v)} hint="1.0 is natural. Preview uses this rate." /></div>}

    {tab === "Scripts" && <div className="card" style={{ maxWidth: 760 }}>
      {T("Greeting", "scripts.greeting", "Pre-synthesized and played within a fraction of a second of connecting.", true)}{T("Closing", "scripts.closing")}
      {T("Clarifying question", "scripts.clarify", "Used when the agent could not understand the caller.")}{T("Reprompt after silence", "scripts.reprompt")}
      {T("Closing after long silence", "scripts.silence_closing")}
      {T("Recording consent question", "scripts.recording_prompt", "Asked only if “Record calls” is on (Guardrails). Recording starts only after an explicit yes.")}
      {T("Reconnect line", "scripts.resume_line", "Spoken when a new agent takes over a call after a failure.")}{T("Tool failure line", "scripts.tool_failure", "Spoken when a tool times out or errors.")}
      <Lines label="Acknowledgment phrases" hint="Spoken while a slow tool runs; rotated so they don't repeat." value={cfg.scripts.fillers} disabled={ro} onChange={(v) => set("scripts.fillers", v)} />
      <Lines label="Fallback lines" value={cfg.scripts.fallback} disabled={ro} onChange={(v) => set("scripts.fallback", v)} rows={2} /></div>}

    {tab === "Turn-taking" && <div className="grid g2">
      <div className="card"><h2>Presets</h2><div className="row">{Object.keys(PRESETS).map((p) => <button key={p} className="btn" disabled={ro}
        onClick={() => Object.entries(PRESETS[p]).forEach(([k, v]) => set(`turn.${k}`, v))}>{p}</button>)}</div>
        <p className="muted small">Relaxed waits longer before replying; Snappy replies sooner but may cut in on slow talkers.</p>
        <h2 style={{ marginTop: 18 }}>Endpointing</h2>
        <Slider label="Wait after a complete sentence" value={turn.endpoint_complete_ms} min={100} max={1500} step={10} unit=" ms" disabled={ro} onChange={(v) => set("turn.endpoint_complete_ms", v)} hint="Lower = faster replies." />
        <Slider label="Wait after an unfinished sentence" value={turn.endpoint_incomplete_ms} min={300} max={4000} step={50} unit=" ms" disabled={ro} onChange={(v) => set("turn.endpoint_incomplete_ms", v)} hint="“My order number is…” — time to find the digits." /></div>
      <div className="card"><h2>Interruptions &amp; noise</h2>
        <Slider label="Voice threshold (agent silent)" value={turn.vad_threshold} min={0.2} max={0.9} step={0.01} disabled={ro} onChange={(v) => set("turn.vad_threshold", v)} hint="Higher ignores quieter speech." />
        <Slider label="Voice threshold (agent speaking)" value={turn.vad_threshold_speaking} min={0.3} max={0.95} step={0.01} disabled={ro} onChange={(v) => set("turn.vad_threshold_speaking", v)} hint="Raised while the agent talks so echo never interrupts it." />
        <Slider label="Speech needed to pause the agent" value={turn.stage1_min_speech_ms} min={100} max={400} step={10} unit=" ms" disabled={ro} onChange={(v) => set("turn.stage1_min_speech_ms", v)} hint="Short noises under this are ignored." />
        <Slider label="Words needed to confirm an interruption" value={turn.interruption_min_words} min={1} max={5} disabled={ro} onChange={(v) => set("turn.interruption_min_words", v)} hint="“mm-hmm” and “okay” never count." />
        <Slider label="Reprompt after silence" value={turn.reprompt_after_s} min={3} max={30} unit=" s" disabled={ro} onChange={(v) => set("turn.reprompt_after_s", v)} />
        <Slider label="End the call after silence" value={turn.close_after_s} min={8} max={90} unit=" s" disabled={ro} onChange={(v) => set("turn.close_after_s", v)} />
        <Lines label="Non-interruptible phrases" hint="If the agent says one of these, the caller cannot barge in (e.g. legal disclosures)." value={turn.non_interruptible_phrases} disabled={ro} onChange={(v) => set("turn.non_interruptible_phrases", v)} rows={3} />
        <Lines label="Extra acknowledgment words" hint="Treated like “mm-hmm”: they do not interrupt." value={turn.extra_backchannels} disabled={ro} onChange={(v) => set("turn.extra_backchannels", v)} rows={2} /></div></div>}

    {tab === "Escalation" && <div className="grid g2">
      <div className="card"><Lines label="Phrases that mean “I want a person”" hint="Regular expressions, one per line." value={esc.human_request_patterns} disabled={ro} onChange={(v) => set("escalation.human_request_patterns", v)} rows={6} />
        <Lines label="Restricted topics" hint="Hand off immediately when mentioned." value={esc.restricted_topics} disabled={ro} onChange={(v) => set("escalation.restricted_topics", v)} rows={3} />
        <Slider label="Frustration sensitivity" value={esc.negative_sentiment_threshold} min={1} max={8} disabled={ro} onChange={(v) => set("escalation.negative_sentiment_threshold", v)} hint="Lower hands off sooner." />
        <Slider label="Hand off after N failed answers in a row" value={esc.max_consecutive_failures} min={1} max={8} disabled={ro} onChange={(v) => set("escalation.max_consecutive_failures", v)} /></div>
      <div className="card"><h2>Business hours</h2>
        <label className="row" style={{ marginBottom: 12 }}><input type="checkbox" disabled={ro} checked={!!bh} onChange={(e) => set("escalation.business_hours", e.target.checked ? { tz: "America/New_York", days: [0, 1, 2, 3, 4], start: "09:00", end: "17:00" } : null)} /> Only transfer to a person during these hours</label>
        {bh && <>
          <Field label="Time zone" hint="IANA name, e.g. America/New_York"><input type="text" disabled={ro} value={bh.tz} onChange={(e) => set("escalation.business_hours.tz", e.target.value)} /></Field>
          <div className="row" style={{ marginBottom: 12 }}>{DAYS.map((d, i) => <label key={d} className="row small"><input type="checkbox" disabled={ro} checked={bh.days.includes(i)}
            onChange={(e) => set("escalation.business_hours.days", e.target.checked ? [...bh.days, i].sort() : bh.days.filter((x: number) => x !== i))} />{d}</label>)}</div>
          <div className="row"><Field label="From"><input type="text" disabled={ro} value={bh.start} onChange={(e) => set("escalation.business_hours.start", e.target.value)} /></Field>
            <Field label="Until"><input type="text" disabled={ro} value={bh.end} onChange={(e) => set("escalation.business_hours.end", e.target.value)} /></Field></div></>}
        {T("Handoff line", "escalation.handoff_line", undefined, true)}{T("After-hours line", "escalation.after_hours_line", "Offered instead of a transfer when closed.", true)}{T("Restricted-topic line", "escalation.restricted_line")}</div></div>}

    {tab === "Tools" && <div className="grid">
      <div className="alert warn small">Tools call your systems with the caller's request. Actions that change data should be marked <span className="mono">side_effect</span>: the agent then reads details back and waits for a yes.</div>
      {toolsText.map((t, i) => <section key={i} className="card"><div className="row spread"><h3>{(() => { try { return JSON.parse(t).name; } catch { return `Tool ${i + 1}`; } })()}</h3>
        <button className="btn sm danger" disabled={ro} onClick={() => setToolsText(toolsText.filter((_, j) => j !== i))}>Remove</button></div>
        <textarea className="code" spellCheck={false} disabled={ro} value={t} onChange={(e) => setToolsText(toolsText.map((x, j) => (j === i ? e.target.value : x)))} aria-label={`Tool ${i + 1} definition`} /></section>)}
      <div><button className="btn" disabled={ro} onClick={() => setToolsText([...toolsText, JSON.stringify(TOOL_TEMPLATE, null, 2)])}>+ Add webhook tool</button></div>
      <section className="card"><h2>Secrets</h2><p className="muted small">Reference as <span className="mono">{"{{secret:NAME}}"}</span> in tool headers. Values are encrypted and never shown again.</p>
        <div className="row small" style={{ marginBottom: 10 }}>{secrets.data?.map((s) => <span key={s} className="badge">{s}</span>)}{secrets.data?.length === 0 && <span className="muted">None yet.</span>}</div>
        {me?.role === "admin" && <div className="row"><input type="text" placeholder="NAME" value={sec.name} onChange={(e) => setSec({ ...sec, name: e.target.value })} aria-label="Secret name" />
          <input type="password" placeholder="value" value={sec.value} onChange={(e) => setSec({ ...sec, value: e.target.value })} autoComplete="off" aria-label="Secret value" />
          <button className="btn" disabled={!sec.name || !sec.value} onClick={addSecret}>Store</button></div>}</section></div>}

    {tab === "Guardrails" && <div className="card" style={{ maxWidth: 720 }}>
      <Lines label="Forbidden topics" hint="The agent politely declines these." value={cfg.guardrails.forbidden_topics} disabled={ro} onChange={(v) => set("guardrails.forbidden_topics", v)} rows={3} />
      <Lines label="Compliance disclaimers" hint="Spoken after the greeting; callers cannot interrupt them." value={cfg.guardrails.compliance_disclaimers} disabled={ro} onChange={(v) => set("guardrails.compliance_disclaimers", v)} rows={3} />
      <Field label="Personal data in transcripts"><select disabled={ro} value={cfg.guardrails.pii_policy} onChange={(e) => set("guardrails.pii_policy", e.target.value)}><option value="redact">Redact emails, phone and card numbers</option><option value="none">Store as spoken</option></select></Field>
      <Slider label="Maximum call length" value={cfg.guardrails.max_session_minutes} min={1} max={120} unit=" min" disabled={ro} onChange={(v) => set("guardrails.max_session_minutes", v)} />
      <Slider label="Keep conversation logs for" value={cfg.guardrails.retention_days} min={0} max={365} unit=" days" disabled={ro} onChange={(v) => set("guardrails.retention_days", v)} hint="0 keeps nothing beyond the call." />
      <label className="row"><input type="checkbox" disabled={ro} checked={cfg.guardrails.record_calls} onChange={(e) => set("guardrails.record_calls", e.target.checked)} /> Record calls (callers are asked for consent)</label></div>}

    {tab === "Prompt" && <div className="card"><p className="muted">Leave empty to use the platform prompt. Voice, security and grounding rules are always appended and cannot be removed. Variables: <span className="mono">{"{{ agent_name }} {{ org_name }} {{ persona }} {{ tone }} {{ languages }} {{ forbidden_topics }}"}</span></p>
      <textarea className="code" style={{ minHeight: 260 }} disabled={ro} placeholder="You are {{ agent_name }}, the support agent for {{ org_name }}…" value={cfg.system_prompt_template ?? ""} onChange={(e) => set("system_prompt_template", e.target.value || null)} aria-label="System prompt template" /></div>}
  </>);
}
