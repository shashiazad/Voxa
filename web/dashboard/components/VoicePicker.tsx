"use client";
import { useRef, useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";

type Voice = { id: string; name: string; description: string };

export default function VoicePicker({ value, rate, onChange, disabled }: {
  value: string; rate: number; onChange: (id: string) => void; disabled?: boolean;
}) {
  const { data } = useApi<Voice[]>("/voices");
  const [playing, setPlaying] = useState<string | null>(null); const [err, setErr] = useState("");
  const [text, setText] = useState("Hi, thanks for calling. I can check an order or answer questions about our policies.");
  const audio = useRef<HTMLAudioElement | null>(null);
  const current = value === "default" ? "amy" : value;

  async function preview(id: string) {
    setErr(""); setPlaying(id);
    try {
      const blob = await api<Blob>("/voices/preview", { method: "POST", json: { voice_id: id, text, rate } });
      audio.current?.pause();
      audio.current = new Audio(URL.createObjectURL(blob));
      audio.current.onended = () => setPlaying(null);
      await audio.current.play();
    } catch (e) { setErr((e as Error).message); setPlaying(null); }
  }
  return (<div>
    <div className="grid g3" role="radiogroup" aria-label="Voice">
      {data?.map((v) => (
        <div key={v.id} className="voice" role="radio" aria-checked={current === v.id} tabIndex={disabled ? -1 : 0}
          onClick={() => !disabled && onChange(v.id)} onKeyDown={(e) => (e.key === " " || e.key === "Enter") && !disabled && onChange(v.id)}>
          <div className="row spread"><b>{v.name}</b>{current === v.id && <span className="badge acc">selected</span>}</div>
          <p className="muted small" style={{ margin: "4px 0 8px" }}>{v.description}</p>
          <button type="button" className="btn sm" onClick={(e) => { e.stopPropagation(); void preview(v.id); }} disabled={playing === v.id}>
            {playing === v.id ? "Playing…" : "▶ Preview"}</button>
        </div>))}
    </div>
    <label className="f" style={{ marginTop: 14 }}><span>Preview text</span><input type="text" value={text} maxLength={300} onChange={(e) => setText(e.target.value)} /></label>
    {err && <div className="alert bad" role="alert">{err}</div>}
  </div>);
}
