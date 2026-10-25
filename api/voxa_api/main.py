"""Voxa API: auth/RBAC, tenant config, knowledge base, voices, widget tokens, conversations, analytics, handoffs."""
from __future__ import annotations

import asyncio
import io
import ipaddress
import logging
import json
import os
import re
import secrets as pysecrets
import socket
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse

import numpy as np
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from livekit import api as lk
from pydantic import BaseModel, Field
from voxa_shared import voices as vc
from voxa_shared.db import create_pool, migrate_db, tenant_conn
from voxa_shared.kb import ingest as ing
from voxa_shared.kb.embed import Embedder, Reranker
from voxa_shared.kb.retrieve import Retriever
from voxa_shared.kb.retrieve import ingest_documents
from voxa_shared.passwords import hash_password, verify_password
from voxa_shared.secrets import DbSecretStore
from voxa_shared.tenant import Identity, TenantConfig

from .auth import Principal, api_key_hash, make_csat_token, make_jwt, principal, rate_limit, read_csat_token, require

MAX_UPLOAD = 20 * 1024 * 1024
AGENT_NAME = os.environ.get("AGENT_NAME", "voxa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.admin_dsn = os.environ.get("DATABASE_ADMIN_URL", "postgresql://voxa:voxa@postgres:5432/voxa")
    await migrate_db(app.state.admin_dsn)
    app.state.pool = await create_pool(os.environ.get("DATABASE_URL", "postgresql://voxa_app:voxa_app@postgres:5432/voxa"))
    app.state.admin_pool = await create_pool(app.state.admin_dsn, min_size=1, max_size=3)
    app.state.redis = aioredis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"))
    app.state.embedder = None
    app.state.voices = {}
    try:
        from voxa_shared import storage
        await asyncio.to_thread(storage.ensure_bucket)
    except Exception as e:  # noqa: BLE001 - recordings are optional; the API must still start
        logging.getLogger('voxa.api').warning('object storage not reachable (%s); recordings disabled', type(e).__name__)
    yield
    await app.state.pool.close(); await app.state.admin_pool.close(); await app.state.redis.aclose()


app = FastAPI(title="Voxa API", lifespan=lifespan)


async def require_admin_token(authorization: str = Header(...)) -> None:
    tok = os.environ.get("ADMIN_TOKEN", "")
    if not tok or not pysecrets.compare_digest(authorization.removeprefix("Bearer ").strip(), tok):
        raise HTTPException(401, "platform admin token required")


def _embedder() -> Embedder:
    if app.state.embedder is None:
        app.state.embedder = Embedder()
    return app.state.embedder


def _livekit_url() -> str:
    return os.environ.get("LIVEKIT_PUBLIC_URL", os.environ.get("LIVEKIT_URL", "ws://localhost:7880"))


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ============================================================ auth + users
class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=200)


class AuthOut(BaseModel):
    token: str
    tenant_id: str
    role: str


@app.post("/v1/auth/login", response_model=AuthOut)
async def login(body: LoginIn, request: Request) -> AuthOut:
    await rate_limit(request, f"login:{request.client.host if request.client else '-'}", 10, 60)
    async with app.state.pool.acquire() as c:
        row = await c.fetchrow("SELECT * FROM login_user($1)", body.email)
    # constant-ish work whether or not the user exists
    ok = verify_password(body.password, row["password_hash"] if row else hash_password("x"))
    if not row or not ok:
        raise HTTPException(401, "invalid email or password")
    return AuthOut(token=make_jwt(str(row["id"]), row["tenant_id"], row["role"]), tenant_id=row["tenant_id"], role=row["role"])


class SignupIn(BaseModel):
    org_name: str = Field(min_length=2, max_length=120)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254)
    password: str = Field(min_length=10, max_length=200)


@app.post("/v1/signup", response_model=AuthOut)
async def signup(body: SignupIn, request: Request) -> AuthOut:
    """Self-serve onboarding (create an agent). Disabled unless SIGNUP_ENABLED=1; IP rate-limited."""
    if os.environ.get("SIGNUP_ENABLED", "0") != "1":
        raise HTTPException(403, "signup is disabled")
    await rate_limit(request, f"signup:{request.client.host if request.client else '-'}", 5, 3600)
    slug = re.sub(r"[^a-z0-9]+", "_", body.org_name.lower()).strip("_")[:24] or "org"
    tid = f"{slug}_{uuid.uuid4().hex[:6]}"
    cfg = TenantConfig(tenant_id=tid, identity=Identity(org_name=body.org_name))
    async with app.state.admin_pool.acquire() as c:
        if await c.fetchval("SELECT 1 FROM users WHERE lower(email)=lower($1)", body.email):
            raise HTTPException(409, "email already registered")
        async with c.transaction():
            await c.execute("INSERT INTO tenants(id,name,api_key_hash,config,config_version) VALUES($1,$2,$3,$4::jsonb,1)",
                            tid, body.org_name, api_key_hash("vx_" + pysecrets.token_urlsafe(32)), cfg.model_dump_json())
            uid = await c.fetchval("INSERT INTO users(tenant_id,email,password_hash,role) VALUES($1,$2,$3,'admin') RETURNING id",
                                   tid, body.email, hash_password(body.password))
    return AuthOut(token=make_jwt(str(uid), tid, "admin"), tenant_id=tid, role="admin")


class UserIn(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254)
    password: str = Field(min_length=10, max_length=200)
    role: str = Field(pattern="^(admin|editor|viewer)$")


@app.get("/v1/me")
async def me(p: Principal = Depends(require("viewer"))) -> dict[str, Any]:
    return {"tenant_id": p.tenant, "role": p.role, "user_id": p.user_id}


@app.get("/v1/users")
async def list_users(p: Principal = Depends(require("admin"))) -> list[dict[str, Any]]:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        rows = await c.fetch("SELECT id::text, email, role, created_at FROM users ORDER BY created_at")
    return [dict(r) for r in rows]


@app.post("/v1/users", status_code=201)
async def create_user(body: UserIn, p: Principal = Depends(require("admin"))) -> dict[str, str]:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        try:
            uid = await c.fetchval("INSERT INTO users(tenant_id,email,password_hash,role) VALUES($1,$2,$3,$4) RETURNING id",
                                   p.tenant, body.email, hash_password(body.password), body.role)
        except Exception as e:  # noqa: BLE001 - unique violation
            if "users_email_ci" in str(e):
                raise HTTPException(409, "email already registered")
            raise
    return {"id": str(uid)}


@app.delete("/v1/users/{uid}", status_code=204)
async def delete_user(uid: uuid.UUID, p: Principal = Depends(require("admin"))) -> None:
    if p.user_id == str(uid):
        raise HTTPException(400, "cannot delete yourself")
    async with tenant_conn(app.state.pool, p.tenant) as c:
        await c.execute("DELETE FROM users WHERE id=$1", uid)


# ============================================================ platform admin (operator)
class TenantCreated(BaseModel):
    tenant_id: str
    api_key: str = Field(description="Shown once; only its hash is stored.")


@app.post("/v1/admin/tenants", response_model=TenantCreated, dependencies=[Depends(require_admin_token)])
async def create_tenant(cfg: TenantConfig, api_key: str | None = None) -> TenantCreated:
    key = api_key or "vx_" + pysecrets.token_urlsafe(32)
    async with app.state.admin_pool.acquire() as c:
        if await c.fetchval("SELECT 1 FROM tenants WHERE id=$1", cfg.tenant_id):
            raise HTTPException(409, "tenant exists")
        await c.execute("INSERT INTO tenants(id,name,api_key_hash,config,config_version) VALUES($1,$2,$3,$4::jsonb,$5)",
                        cfg.tenant_id, cfg.identity.org_name, api_key_hash(key), cfg.model_dump_json(), cfg.version)
    return TenantCreated(tenant_id=cfg.tenant_id, api_key=key)


@app.put("/v1/admin/tenants/{tenant_id}/config", dependencies=[Depends(require_admin_token)])
async def admin_update_config(tenant_id: str, cfg: TenantConfig) -> dict[str, int]:
    if cfg.tenant_id != tenant_id:
        raise HTTPException(400, "tenant_id mismatch")
    return {"version": await _save_config(app.state.admin_pool, tenant_id, cfg)}


class SecretIn(BaseModel):
    value: str = Field(min_length=1, max_length=4096)


@app.put("/v1/admin/tenants/{tenant_id}/secrets/{name}", dependencies=[Depends(require_admin_token)], status_code=204)
async def admin_put_secret(tenant_id: str, name: str, body: SecretIn) -> None:
    await DbSecretStore(app.state.pool).put(tenant_id, name, body.value)


@app.put("/v1/admin/tenants/{tenant_id}/users", dependencies=[Depends(require_admin_token)], status_code=204)
async def admin_upsert_user(tenant_id: str, body: UserIn) -> None:
    async with app.state.admin_pool.acquire() as c:
        await c.execute("INSERT INTO users(tenant_id,email,password_hash,role) VALUES($1,$2,$3,$4) "
                        "ON CONFLICT (lower(email)) DO UPDATE SET password_hash=EXCLUDED.password_hash, role=EXCLUDED.role "
                        "WHERE users.tenant_id=EXCLUDED.tenant_id",
                        tenant_id, body.email, hash_password(body.password), body.role)


async def _save_config(pool, tenant_id: str, cfg: TenantConfig, scoped: bool = False) -> int:
    """scoped=True uses the RLS-confined app role (tenant-initiated); False the admin pool (operator)."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def conn():
        if scoped:
            async with tenant_conn(pool, tenant_id) as c:
                yield c
        else:
            async with pool.acquire() as c:
                yield c

    async with conn() as c:
        cur = await c.fetchval("SELECT config_version FROM tenants WHERE id=$1", tenant_id)
        if cur is None:
            raise HTTPException(404, "unknown tenant")
        new = max(cfg.version, cur + 1)            # versions only move forward
        cfg = cfg.model_copy(update={"version": new})
        await c.execute("UPDATE tenants SET config=$2::jsonb, config_version=$3, name=$4, updated_at=now() WHERE id=$1",
                        tenant_id, cfg.model_dump_json(), new, cfg.identity.org_name)
    return new


def _check_tool_urls(cfg: TenantConfig) -> None:
    """Tenant-supplied tool URLs are fetched server-side by the agent: refuse non-http(s) and (unless the
    deployment opts in) private/loopback targets. DNS is re-checked at call time in phase 7 hardening."""
    allow_private = os.environ.get("TOOL_ALLOW_PRIVATE", "0") == "1"
    for t in cfg.tools:
        if t.kind != "webhook":
            continue
        u = urlparse(re.sub(r"\{[^}]*\}", "x", t.url))
        if u.scheme not in ("http", "https") or not u.hostname:
            raise HTTPException(422, f"tool {t.name}: url must be http(s)")
        if allow_private:
            continue
        try:
            for info in socket.getaddrinfo(u.hostname, None):
                ip = ipaddress.ip_address(info[4][0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise HTTPException(422, f"tool {t.name}: url resolves to a private address")
        except socket.gaierror:
            raise HTTPException(422, f"tool {t.name}: host does not resolve")


# ============================================================ tenant config
@app.get("/v1/config")
async def get_config(p: Principal = Depends(require("viewer"))) -> dict[str, Any]:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        raw = json.loads(await c.fetchval("SELECT config FROM tenants WHERE id=$1", p.tenant))
    # normalise through the schema so fields added after the tenant was created appear with their defaults
    return TenantConfig.model_validate(raw).model_dump(mode="json")


@app.put("/v1/config")
async def put_config(cfg: TenantConfig, p: Principal = Depends(require("editor"))) -> dict[str, int]:
    if cfg.tenant_id != p.tenant:
        raise HTTPException(400, "tenant_id mismatch")
    _check_tool_urls(cfg)
    return {"version": await _save_config(app.state.pool, p.tenant, cfg, scoped=True)}


class SecretNameIn(BaseModel):
    value: str = Field(min_length=1, max_length=4096)


@app.put("/v1/secrets/{name}", status_code=204)
async def put_secret(name: str, body: SecretNameIn, p: Principal = Depends(require("admin"))) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
        raise HTTPException(422, "bad secret name")
    await DbSecretStore(app.state.pool).put(p.tenant, name, body.value)


@app.get("/v1/secrets")
async def list_secrets(p: Principal = Depends(require("editor"))) -> list[str]:
    """Names only; values are never returned."""
    async with tenant_conn(app.state.pool, p.tenant) as c:
        return [r["name"] for r in await c.fetch("SELECT name FROM secrets ORDER BY name")]


# ============================================================ voices
@app.get("/v1/voices")
async def voices(p: Principal = Depends(require("viewer"))) -> list[dict[str, str]]:
    return [{"id": v.id, "name": v.name, "description": v.description, "lang": v.lang} for v in vc.VOICES.values()]


class PreviewIn(BaseModel):
    voice_id: str = "amy"
    text: str = Field(default="Hi, thanks for calling. How can I help you today?", min_length=1, max_length=300)
    rate: float = Field(default=1.0, ge=0.6, le=1.6)


@app.post("/v1/voices/preview")
async def preview_voice(body: PreviewIn, request: Request, p: Principal = Depends(require("viewer"))) -> Response:
    await rate_limit(request, f"preview:{p.tenant}", 30, 60)
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    v = vc.resolve(body.voice_id)
    if v.id not in app.state.voices:
        app.state.voices[v.id] = PiperVoice.load(vc.path(v))
    voice = app.state.voices[v.id]
    syn = SynthesisConfig(length_scale=1.0 / body.rate)

    def synth() -> bytes:
        pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(body.text, syn_config=syn))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(voice.config.sample_rate); w.writeframes(pcm)
        return buf.getvalue()

    import asyncio
    return Response(await asyncio.to_thread(synth), media_type="audio/wav")


# ============================================================ rooms + widget
class TokenResponse(BaseModel):
    url: str
    room: str
    token: str
    csat_token: str | None = None


def _room_token(tenant: str, identity_prefix: str = "user", ttl_s: int | None = None, room: str | None = None,
                dispatch_agent: bool = True) -> TokenResponse:
    ttl = ttl_s or int(os.environ.get("ROOM_TOKEN_TTL_SECONDS", "600"))
    room = room or f"{tenant}-{uuid.uuid4().hex[:12]}"
    token = (lk.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
             .with_identity(f"{identity_prefix}-{uuid.uuid4().hex[:8]}").with_ttl(timedelta(seconds=ttl))
             .with_metadata(json.dumps({"tenant_id": tenant}))
             .with_grants(lk.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True,
                                         can_publish_data=True)))
    if dispatch_agent:   # the voice agent joins when the caller does (named agent, so it can also be re-dispatched)
        token = token.with_room_config(lk.RoomConfiguration(agents=[lk.RoomAgentDispatch(agent_name=AGENT_NAME)]))
    return TokenResponse(url=_livekit_url(), room=room, token=token.to_jwt())


@app.post("/v1/rooms/token", response_model=TokenResponse)
async def mint_token(request: Request, p: Principal = Depends(require("editor"))) -> TokenResponse:
    """Test-call token for dashboard users / server integrations."""
    await rate_limit(request, f"room:{p.tenant}", 60, 60)
    return _room_token(p.tenant)


@app.post("/v1/widget/token", response_model=TokenResponse)
async def widget_token(request: Request, x_widget_key: str = Header(...), origin: str | None = Header(None),
                       x_embed_origin: str | None = Header(None)) -> TokenResponse:
    """Public endpoint for the embeddable widget. The widget key is publishable; abuse is bounded by the
    origin allow-list (browsers) and per-key + per-IP rate limits (everything else)."""
    ip = request.client.host if request.client else "-"
    await rate_limit(request, f"wtok-ip:{ip}", 20, 60)
    async with app.state.pool.acquire() as c:
        row = await c.fetchrow("SELECT * FROM widget_tenant($1)", x_widget_key)
    if not row:
        raise HTTPException(401, "invalid widget key")
    await rate_limit(request, f"wtok-key:{x_widget_key}", 120, 60)
    o = (origin or x_embed_origin or "").rstrip("/")
    if row["allowed_origins"] and o not in row["allowed_origins"] and "*" not in row["allowed_origins"]:
        raise HTTPException(403, "origin not allowed for this widget key")
    if not row["allowed_origins"]:
        raise HTTPException(403, "no allowed origins configured for this widget key")
    out = _room_token(row["id"], ttl_s=300)
    out.csat_token = make_csat_token(row["id"], out.room)
    return out


class CsatIn(BaseModel):
    token: str
    score: int = Field(ge=1, le=5)


@app.post("/v1/widget/csat", status_code=204)
async def widget_csat(body: CsatIn, request: Request) -> None:
    """Post-call rating. The room-bound token (minted with the call) proves this browser made the call."""
    await rate_limit(request, f"csat:{request.client.host if request.client else '-'}", 20, 60)
    try:
        tenant, room = read_csat_token(body.token)
    except Exception:  # noqa: BLE001
        raise HTTPException(401, "invalid or expired rating token")
    async with tenant_conn(app.state.pool, tenant) as c:
        await c.execute("UPDATE conversations SET csat=$2 WHERE room=$1", room, body.score)


class WidgetSettings(BaseModel):
    allowed_origins: list[str] = Field(default_factory=list, max_length=50)


@app.get("/v1/widget")
async def get_widget(p: Principal = Depends(require("viewer"))) -> dict[str, Any]:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        r = await c.fetchrow("SELECT widget_key, allowed_origins FROM tenants WHERE id=$1", p.tenant)
    return {"widget_key": r["widget_key"], "allowed_origins": list(r["allowed_origins"])}


@app.put("/v1/widget")
async def put_widget(body: WidgetSettings, p: Principal = Depends(require("admin"))) -> dict[str, Any]:
    origins = []
    for o in body.allowed_origins:
        o = o.strip().rstrip("/")
        u = urlparse(o)
        if o != "*" and (u.scheme not in ("http", "https") or not u.netloc or u.path):
            raise HTTPException(422, f"origin must look like https://example.com: {o!r}")
        origins.append(o)
    async with tenant_conn(app.state.pool, p.tenant) as c:
        await c.execute("UPDATE tenants SET allowed_origins=$2 WHERE id=$1", p.tenant, origins)
    return {"allowed_origins": origins}


@app.post("/v1/widget/rotate")
async def rotate_widget_key(p: Principal = Depends(require("admin"))) -> dict[str, str]:
    key = "wk_" + uuid.uuid4().hex
    async with tenant_conn(app.state.pool, p.tenant) as c:
        await c.execute("UPDATE tenants SET widget_key=$2 WHERE id=$1", p.tenant, key)
    return {"widget_key": key}


# ============================================================ knowledge base
@app.post("/v1/kb/documents")
async def upload_document(file: UploadFile = File(...), p: Principal = Depends(require("editor"))) -> dict[str, int]:
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (20 MB max)")
    name = file.filename or "upload"
    ext = name.rsplit(".", 1)[-1].lower()
    parsers = {"pdf": ing.parse_pdf, "docx": ing.parse_docx, "csv": ing.parse_csv, "txt": ing.parse_text, "md": ing.parse_text}
    if ext not in parsers:
        raise HTTPException(415, f"unsupported type .{ext}")
    try:
        doc = parsers[ext](data, name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"could not parse {name}: {type(e).__name__}")
    stats = await ingest_documents(app.state.pool, _embedder(), p.tenant, [doc])
    await _flush_kb_cache(p.tenant)
    return stats


class CrawlIn(BaseModel):
    url: str = Field(pattern=r"^https?://")
    max_pages: int = Field(default=10, ge=1, le=50)


@app.post("/v1/kb/crawl")
async def crawl(body: CrawlIn, p: Principal = Depends(require("editor"))) -> dict[str, int]:
    docs = await ing.crawl_url(body.url, body.max_pages)
    stats = await ingest_documents(app.state.pool, _embedder(), p.tenant, docs)
    await _flush_kb_cache(p.tenant)
    return stats


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)


@app.post("/v1/kb/search")
async def kb_search(body: SearchIn, request: Request, p: Principal = Depends(require("editor"))) -> dict[str, Any]:
    """'What would the agent retrieve for this question?' - the same hybrid search the voice agent uses."""
    await rate_limit(request, f"kbsearch:{p.tenant}", 60, 60)
    if getattr(app.state, "retriever", None) is None:
        app.state.retriever = Retriever(app.state.pool, _embedder(), Reranker(), None)
    hits = await app.state.retriever.search(p.tenant, body.query)
    return {"latency_ms": round(app.state.retriever.last_ms, 1),
            "hits": [{"content": h.content, "score": round(h.score, 3)} for h in hits]}


@app.get("/v1/kb/documents")
async def list_documents(p: Principal = Depends(require("viewer"))) -> list[dict[str, Any]]:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        rows = await c.fetch("SELECT d.id::text, d.title, d.source, d.created_at, count(k.id) AS chunks "
                             "FROM kb_documents d LEFT JOIN kb_chunks k ON k.document_id=d.id GROUP BY d.id ORDER BY d.created_at DESC")
    return [dict(r) for r in rows]


@app.delete("/v1/kb/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: uuid.UUID, p: Principal = Depends(require("editor"))) -> None:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        await c.execute("DELETE FROM kb_documents WHERE id=$1", doc_id)
    await _flush_kb_cache(p.tenant)


async def _flush_kb_cache(tenant: str) -> None:
    keys = [k async for k in app.state.redis.scan_iter(f"kb:{tenant}:*")]
    if keys:
        await app.state.redis.delete(*keys)


# ============================================================ conversations + analytics + handoffs
@app.get("/v1/conversations")
async def conversations(limit: int = 50, p: Principal = Depends(require("viewer"))) -> list[dict[str, Any]]:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        rows = await c.fetch(
            "SELECT c.id::text, c.room, c.started_at, c.ended_at, c.handoff, c.csat, c.summary, "
            "(SELECT count(*) FROM turns t WHERE t.conversation_id=c.id AND t.role='user') AS user_turns, "
            "(SELECT count(*) FROM turns t WHERE t.conversation_id=c.id AND t.interrupted) AS interruptions, "
            "(SELECT count(*) FROM turns t WHERE t.conversation_id=c.id AND t.role='tool') AS tool_calls, "
            "(SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY (t.metrics->'latency'->>'voice_to_voice_ms')::float) "
            "   FROM turns t WHERE t.conversation_id=c.id AND t.metrics->'latency'->>'voice_to_voice_ms' IS NOT NULL) AS p50_v2v_ms "
            "FROM conversations c ORDER BY c.started_at DESC LIMIT $1", min(limit, 200))
    return [dict(r) for r in rows]


@app.get("/v1/conversations/{cid}")
async def conversation(cid: uuid.UUID, p: Principal = Depends(require("viewer"))) -> dict[str, Any]:
    async with tenant_conn(app.state.pool, p.tenant) as c:
        conv = await c.fetchrow("SELECT id::text, room, started_at, ended_at, handoff, csat, summary, consent_recording "
                                "FROM conversations WHERE id=$1", cid)
        if not conv:
            raise HTTPException(404)
        turns = await c.fetch("SELECT idx, role, text, interrupted, metrics, created_at FROM turns "
                              "WHERE conversation_id=$1 ORDER BY idx", cid)
    return {**dict(conv), "turns": [{**dict(t), "metrics": json.loads(t["metrics"]) if t["metrics"] else None} for t in turns]}


@app.get("/v1/analytics/summary")
async def analytics(days: int = 7, p: Principal = Depends(require("viewer"))) -> dict[str, Any]:
    days = max(1, min(days, 90))
    async with tenant_conn(app.state.pool, p.tenant) as c:
        tot = await c.fetchrow(
            "SELECT count(*) AS calls, count(*) FILTER (WHERE handoff) AS handoffs, avg(csat)::float AS csat "
            "FROM conversations WHERE started_at > now() - make_interval(days => $1)", days)
        lat = await c.fetchrow(
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY v) AS p50, percentile_cont(0.95) WITHIN GROUP (ORDER BY v) AS p95, "
            "count(*) AS n FROM (SELECT (metrics->'latency'->>'voice_to_voice_ms')::float AS v FROM turns "
            "WHERE created_at > now() - make_interval(days => $1) AND metrics->'latency'->>'voice_to_voice_ms' IS NOT NULL) s", days)
        ints = await c.fetchrow(
            "SELECT count(*) FILTER (WHERE interrupted) AS interrupted, count(*) FILTER (WHERE role='assistant') AS replies, "
            "count(*) FILTER (WHERE role='tool') AS tool_calls FROM turns WHERE created_at > now() - make_interval(days => $1)", days)
        series = await c.fetch(
            "SELECT date_trunc('day', started_at)::date AS day, count(*) AS calls FROM conversations "
            "WHERE started_at > now() - make_interval(days => $1) GROUP BY 1 ORDER BY 1", days)
    return {"days": days, "calls": tot["calls"], "handoffs": tot["handoffs"], "csat_avg": tot["csat"],
            "v2v_p50_ms": lat["p50"], "v2v_p95_ms": lat["p95"], "v2v_samples": lat["n"],
            "interrupted_replies": ints["interrupted"], "replies": ints["replies"], "tool_calls": ints["tool_calls"],
            "series": [{"day": str(r["day"]), "calls": r["calls"]} for r in series]}


@app.get("/v1/handoffs")
async def handoffs(p: Principal = Depends(require("viewer"))) -> list[dict[str, Any]]:
    raw = await app.state.redis.hgetall(f"handoff:{p.tenant}")
    return sorted((json.loads(v) for v in raw.values()), key=lambda r: r["ts"])


@app.post("/v1/handoffs/{room}/join", response_model=TokenResponse)
async def join_handoff(room: str, p: Principal = Depends(require("editor"))) -> TokenResponse:
    """Token for a human agent to join a waiting caller's room (warm transfer)."""
    if not room.startswith(f"{p.tenant}-") or not await app.state.redis.hexists(f"handoff:{p.tenant}", room):
        raise HTTPException(404, "no such pending handoff")
    return _room_token(p.tenant, "human", 600, room, dispatch_agent=False)


# ============================================================ privacy: export, erasure, recordings (GDPR / DPDP)
def _delete_objects(keys: list[str]) -> None:
    from voxa_shared import storage

    for k in keys:
        try:
            storage.delete(k)
        except Exception:  # noqa: BLE001 - an already-missing object is fine; the DB row is the source of truth
            pass


@app.get("/v1/conversations/{cid}/recording")
async def recording_link(cid: uuid.UUID, p: Principal = Depends(require("editor"))) -> dict[str, Any]:
    """Short-lived signed URL. Only exists if the caller explicitly consented to recording."""
    from voxa_shared import storage

    async with tenant_conn(app.state.pool, p.tenant) as c:
        r = await c.fetchrow("SELECT object_key, duration_s FROM recordings WHERE conversation_id=$1 ORDER BY id DESC LIMIT 1", cid)
    if not r:
        raise HTTPException(404, "no recording for this conversation")
    return {"url": storage.presign_get(r["object_key"], 300), "duration_s": r["duration_s"], "expires_in": 300}


@app.get("/v1/conversations/{cid}/export")
async def export_conversation(cid: uuid.UUID, p: Principal = Depends(require("admin"))) -> dict[str, Any]:
    """Data-subject access request: everything stored about one conversation."""
    d = await conversation(cid, p)
    async with tenant_conn(app.state.pool, p.tenant) as c:
        d["recordings"] = [dict(r) for r in await c.fetch(
            "SELECT object_key, duration_s, bytes, created_at FROM recordings WHERE conversation_id=$1", cid)]
        d["tool_calls"] = [dict(r) for r in await c.fetch(
            "SELECT name, ok, duration_ms, created_at FROM tool_calls WHERE conversation_id=$1 ORDER BY id", cid)]
    return d


@app.delete("/v1/conversations/{cid}", status_code=204)
async def erase_conversation(cid: uuid.UUID, p: Principal = Depends(require("admin"))) -> None:
    """Right to erasure for one conversation: transcript, turns, tool-call log and recording audio."""
    async with tenant_conn(app.state.pool, p.tenant) as c:
        keys = [r["object_key"] for r in await c.fetch("SELECT object_key FROM recordings WHERE conversation_id=$1", cid)]
        n = await c.fetchval("WITH d AS (DELETE FROM conversations WHERE id=$1 RETURNING 1) SELECT count(*) FROM d", cid)
    if not n:
        raise HTTPException(404)
    _delete_objects(keys)


@app.delete("/v1/conversations")
async def erase_before(before: str, p: Principal = Depends(require("admin"))) -> dict[str, int]:
    """Bulk erasure: every conversation that started before the given ISO timestamp."""
    from datetime import datetime

    try:
        ts = datetime.fromisoformat(before)
    except ValueError:
        raise HTTPException(422, "before must be an ISO-8601 timestamp")
    async with tenant_conn(app.state.pool, p.tenant) as c:
        keys = [r["object_key"] for r in await c.fetch(
            "SELECT r.object_key FROM recordings r JOIN conversations v ON v.id=r.conversation_id WHERE v.started_at < $1", ts)]
        n = await c.fetchval("WITH d AS (DELETE FROM conversations WHERE started_at < $1 RETURNING 1) SELECT count(*) FROM d", ts)
    _delete_objects(keys)
    return {"deleted": n, "recordings_deleted": len(keys)}


@app.delete("/v1/admin/tenants/{tenant_id}", status_code=204, dependencies=[Depends(require_admin_token)])
async def offboard_tenant(tenant_id: str) -> None:
    """Delete a tenant and everything it owns (cascades to KB, conversations, users, secrets); purge recordings."""
    async with app.state.admin_pool.acquire() as c:
        keys = [r["object_key"] for r in await c.fetch("SELECT object_key FROM recordings WHERE tenant_id=$1", tenant_id)]
        await c.execute("DELETE FROM tenants WHERE id=$1", tenant_id)
    _delete_objects(keys)
    await app.state.redis.delete(f"handoff:{tenant_id}")


class ResumeIn(BaseModel):
    token: str


@app.post("/v1/widget/resume", response_model=TokenResponse)
async def widget_resume(body: ResumeIn, request: Request) -> TokenResponse:
    """The agent process died mid-call: put a fresh agent into the same room (it restores the conversation from
    Redis, 60 s TTL) and give the caller a new join token. Authorised by the rating token minted with the call."""
    await rate_limit(request, f"resume:{request.client.host if request.client else '-'}", 6, 60)
    try:
        tenant, room = read_csat_token(body.token)
    except Exception:  # noqa: BLE001
        raise HTTPException(401, "invalid or expired token")
    if not await app.state.redis.exists(f"session:{room}"):
        raise HTTPException(410, "this call can no longer be resumed")
    async with lk.LiveKitAPI(os.environ.get("LIVEKIT_URL"), os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]) as api_:
        await api_.agent_dispatch.create_dispatch(lk.CreateAgentDispatchRequest(room=room, agent_name=AGENT_NAME))
    return _room_token(tenant, ttl_s=300, room=room, dispatch_agent=False)
