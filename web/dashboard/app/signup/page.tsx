"use client";
import Link from "next/link";
import { useState } from "react";

export default function Signup() {
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault(); setBusy(true); setErr("");
    const f = new FormData(e.currentTarget);
    const r = await fetch("/api/session?mode=signup", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ org_name: f.get("org"), email: f.get("email"), password: f.get("password") }) });
    if (r.ok) { location.href = "/onboarding"; return; }
    const j = await r.json().catch(() => ({}));
    setErr(r.status === 409 ? "That email is already registered." : r.status === 403 ? "Self-serve signup is disabled on this deployment."
      : r.status === 422 ? "Use a valid email and a password of at least 10 characters." : (typeof j.detail === "string" ? j.detail : "Could not create the account."));
    setBusy(false);
  }
  return (
    <main className="auth"><form className="card" onSubmit={submit}>
      <div className="brand" style={{ padding: 0, marginBottom: 18 }}><i /> Voxa Console</div>
      <h1>Create your voice agent</h1><p className="muted" style={{ marginTop: 0 }}>Takes about five minutes.</p>
      {err && <div className="alert bad" role="alert">{err}</div>}
      <label className="f"><span>Organization</span><input name="org" type="text" required minLength={2} maxLength={120} autoFocus /></label>
      <label className="f"><span>Work email</span><input name="email" type="email" required autoComplete="username" /></label>
      <label className="f"><span>Password</span><input name="password" type="password" required minLength={10} autoComplete="new-password" /><small>At least 10 characters.</small></label>
      <button className="btn primary" style={{ width: "100%", justifyContent: "center" }} disabled={busy}>{busy ? "Creating…" : "Create agent"}</button>
      <p className="small muted" style={{ textAlign: "center", marginBottom: 0 }}>Already have an account? <Link href="/login">Sign in</Link></p>
    </form></main>
  );
}
