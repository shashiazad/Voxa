import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_URL ?? "http://localhost:8000";
const secure = process.env.COOKIE_SECURE === "1";

/** POST {email,password[,org_name]} -> sets the httpOnly session cookie. DELETE -> sign out. GET -> who am I. */
export async function POST(req: NextRequest) {
  const body = await req.json();
  const signup = req.nextUrl.searchParams.get("mode") === "signup";
  const r = await fetch(`${API}/v1/${signup ? "signup" : "auth/login"}`, {
    method: "POST", headers: { "content-type": "application/json", "x-forwarded-for": req.headers.get("x-forwarded-for") ?? "" },
    body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) return NextResponse.json(j, { status: r.status });
  (await cookies()).set("vx_session", j.token, { httpOnly: true, sameSite: "lax", secure, path: "/", maxAge: 8 * 3600 });
  return NextResponse.json({ tenant_id: j.tenant_id, role: j.role });
}

export async function DELETE() {
  (await cookies()).delete("vx_session");
  return NextResponse.json({ ok: true });
}

export async function GET() {
  const token = (await cookies()).get("vx_session")?.value;
  if (!token) return NextResponse.json({ detail: "not signed in" }, { status: 401 });
  const r = await fetch(`${API}/v1/me`, { headers: { authorization: `Bearer ${token}` }, cache: "no-store" });
  return NextResponse.json(await r.json(), { status: r.status });
}
