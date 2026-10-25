"use client";
import Link from "next/link";
import { useState } from "react";

export default function Login() {
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault(); setBusy(true); setErr("");
    const f = new FormData(e.currentTarget);
    const r = await fetch("/api/session", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: f.get("email"), password: f.get("password") }) });
    if (r.ok) { location.href = "/"; return; }
    setErr(r.status === 429 ? "Too many attempts. Wait a minute and try again." : "Invalid email or password.");
    setBusy(false);
  }
  return (
    <main className="auth"><form className="card" onSubmit={submit}>
      <div className="brand" style={{ padding: 0, marginBottom: 18 }}><i /> Voxa Console</div>
      <h1>Sign in</h1><p className="muted" style={{ marginTop: 0 }}>Manage your voice support agent.</p>
      {err && <div className="alert bad" role="alert">{err}</div>}
      <label className="f"><span>Email</span><input name="email" type="email" required autoComplete="username" autoFocus /></label>
      <label className="f"><span>Password</span><input name="password" type="password" required autoComplete="current-password" /></label>
      <button className="btn primary" style={{ width: "100%", justifyContent: "center" }} disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
      <p className="small muted" style={{ textAlign: "center", marginBottom: 0 }}>New here? <Link href="/signup">Create an agent</Link></p>
    </form></main>
  );
}
