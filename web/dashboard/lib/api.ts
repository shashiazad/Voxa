export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : Array.isArray(detail)
      ? (detail as { loc?: unknown[]; msg: string }[]).map((d) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`).join("; ")
      : `HTTP ${status}`);
  }
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers = new Headers(init.headers);
  let body = init.body;
  if (init.json !== undefined) { headers.set("content-type", "application/json"); body = JSON.stringify(init.json); }
  const r = await fetch(`/api/v1${path}`, { ...init, headers, body, cache: "no-store" });
  if (r.status === 401 && typeof window !== "undefined" && !location.pathname.startsWith("/login")) {
    location.href = "/login"; throw new ApiError(401, "Session expired");
  }
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new ApiError(r.status, (j as { detail?: unknown }).detail ?? r.statusText);
  }
  if (r.status === 204) return undefined as T;
  const ct = r.headers.get("content-type") ?? "";
  return (ct.includes("json") ? r.json() : r.blob()) as Promise<T>;
}

export const ms = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v)} ms`);
export const when = (s: string | null) => (s ? new Date(s).toLocaleString() : "–");
export function dur(a: string, b: string | null) {
  if (!b) return "–";
  const s = Math.max(0, (new Date(b).getTime() - new Date(a).getTime()) / 1000);
  return s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}
