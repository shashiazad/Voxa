import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_URL ?? "http://localhost:8000";
const PUBLIC = new Set(["widget/token", "widget/csat", "widget/resume"]);      // called cross-origin by the embeddable widget: no session cookie

const cors = (origin: string | null): Record<string, string> => ({
  // Access control is enforced by the API (widget key + origin allow-list); the response is only useful if it succeeds.
  "Access-Control-Allow-Origin": origin ?? "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "X-Widget-Key, Content-Type",
  "Access-Control-Max-Age": "600",
  Vary: "Origin",
});

async function handle(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const sub = path.join("/");
  const isPublic = PUBLIC.has(sub);
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new NextResponse(null, { status: 204, headers: isPublic ? cors(origin) : {} });

  const headers = new Headers();
  const ct = req.headers.get("content-type");
  if (ct) headers.set("content-type", ct);
  if (isPublic) {
    const k = req.headers.get("x-widget-key");    // token endpoint only; csat carries its own signed token
    if (k) headers.set("x-widget-key", k);
    if (origin) headers.set("origin", origin);
    headers.set("x-forwarded-for", req.headers.get("x-forwarded-for") ?? "");
  } else {
    const token = (await cookies()).get("vx_session")?.value;
    if (!token) return NextResponse.json({ detail: "not signed in" }, { status: 401 });
    headers.set("authorization", `Bearer ${token}`);
  }
  const hasBody = !["GET", "HEAD"].includes(req.method);
  const upstream = await fetch(`${API}/v1/${sub}${req.nextUrl.search}`, {
    method: req.method, headers, body: hasBody ? await req.arrayBuffer() : undefined, cache: "no-store",
  });
  const out = new Headers();
  const uct = upstream.headers.get("content-type");
  if (uct) out.set("content-type", uct);
  if (isPublic) Object.entries(cors(origin)).forEach(([k, v]) => out.set(k, v));
  return new NextResponse(upstream.status === 204 ? null : await upstream.arrayBuffer(), { status: upstream.status, headers: out });
}
export { handle as GET, handle as POST, handle as PUT, handle as DELETE, handle as OPTIONS };
