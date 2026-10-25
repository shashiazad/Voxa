"use client";
import { useState } from "react";
import { Err, Field, Ok } from "@/components/ui";
import { api, when } from "@/lib/api";
import { useApi, useMe } from "@/lib/hooks";

type User = { id: string; email: string; role: string; created_at: string };

export default function Team() {
  const me = useMe(); const admin = me?.role === "admin";
  const users = useApi<User[]>(admin ? "/users" : null);
  const [f, setF] = useState({ email: "", password: "", role: "viewer" }); const [err, setErr] = useState(""); const [ok, setOk] = useState("");
  async function add(e: React.FormEvent) {
    e.preventDefault(); setErr(""); setOk("");
    try { await api("/users", { method: "POST", json: f }); setOk(`Added ${f.email}`); setF({ email: "", password: "", role: "viewer" }); void users.reload(); }
    catch (er) { setErr((er as Error).message); }
  }
  return (<>
    <div className="page-head"><div><h1>Team</h1><p className="muted" style={{ margin: 0 }}>Roles: <b>admin</b> manages everything, <b>editor</b> changes the agent and runs test calls, <b>viewer</b> is read-only.</p></div></div>
    {!admin && me && <div className="alert warn">Only admins can manage the team.</div>}
    <Err msg={err} /><Ok msg={ok} />
    {admin && <div className="grid g2">
      <section className="card" style={{ padding: 0 }}><table><thead><tr><th>Email</th><th>Role</th><th>Added</th><th></th></tr></thead><tbody>
        {users.data?.map((u) => <tr key={u.id}><td>{u.email}</td><td><span className="badge acc">{u.role}</span></td><td>{when(u.created_at)}</td>
          <td>{u.id !== me?.user_id && <button className="btn sm danger" onClick={async () => { await api(`/users/${u.id}`, { method: "DELETE" }); void users.reload(); }}>Remove</button>}</td></tr>)}</tbody></table></section>
      <form className="card" onSubmit={add}><h2>Add a teammate</h2>
        <Field label="Email"><input type="email" required value={f.email} onChange={(e) => setF({ ...f, email: e.target.value })} /></Field>
        <Field label="Temporary password" hint="At least 10 characters."><input type="password" required minLength={10} value={f.password} onChange={(e) => setF({ ...f, password: e.target.value })} autoComplete="new-password" /></Field>
        <Field label="Role"><select value={f.role} onChange={(e) => setF({ ...f, role: e.target.value })}><option>viewer</option><option>editor</option><option>admin</option></select></Field>
        <button className="btn primary">Add user</button></form></div>}
  </>);
}
