"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMe } from "@/lib/hooks";

const NAV = [
  ["/", "Overview"], ["/agent", "Agent"], ["/knowledge", "Knowledge"], ["/test", "Test call"],
  ["/calls", "Conversations"], ["/handoffs", "Handoffs"], ["/widget", "Website widget"], ["/team", "Team"],
] as const;

export default function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname(); const me = useMe();
  async function out() { await fetch("/api/session", { method: "DELETE" }); location.href = "/login"; }
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand"><i /> Voxa Console</div>
        <nav className="nav" aria-label="Main">
          {NAV.map(([href, label]) => (
            <Link key={href} href={href} aria-current={(href === "/" ? path === "/" : path.startsWith(href)) ? "page" : undefined}>{label}</Link>
          ))}
        </nav>
        <div className="foot small">
          <div><b>{me?.tenant_id ?? "…"}</b></div>
          <div className="muted">{me ? `${me.role} access` : ""}</div>
          <button className="btn sm" style={{ marginTop: 8 }} onClick={out}>Sign out</button>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
