"use client";
import { useState } from "react";

export function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return <label className="f"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}
export const Err = ({ msg }: { msg?: string | null }) => (msg ? <div className="alert bad" role="alert">{msg}</div> : null);
export const Ok = ({ msg }: { msg?: string | null }) => (msg ? <div className="alert ok" role="status">{msg}</div> : null);

export function Slider({ label, hint, value, min, max, step = 1, unit = "", disabled, onChange }: {
  label: string; hint?: string; value: number; min: number; max: number; step?: number; unit?: string; disabled?: boolean; onChange: (v: number) => void;
}) {
  return (
    <label className="f">
      <div className="row spread" style={{ fontWeight: 600, marginBottom: 4 }}><span>{label}</span><span className="mono">{value}{unit}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <small>{hint}</small>}
    </label>
  );
}

export function Lines({ label, hint, value, onChange, disabled, rows = 4 }: {
  label: string; hint?: string; value: string[]; onChange: (v: string[]) => void; disabled?: boolean; rows?: number;
}) {
  return (
    <Field label={label} hint={hint ?? "One per line"}>
      <textarea rows={rows} disabled={disabled} value={value.join("\n")}
        onChange={(e) => onChange(e.target.value.split("\n"))} onBlur={(e) => onChange(e.target.value.split("\n").map((s) => s.trim()).filter(Boolean))} />
    </Field>
  );
}

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return <button type="button" className="btn sm" onClick={async () => { await navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>{done ? "Copied" : label}</button>;
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return <div className="card stat"><div className="l">{label}</div><div className="v">{value}</div>{sub && <div className="muted small">{sub}</div>}</div>;
}
