"""Authentication + RBAC: per-tenant API keys (service, admin role) or dashboard user JWTs."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, Request

ROLE_LEVEL = {"viewer": 1, "editor": 2, "admin": 3}
JWT_TTL_S = 8 * 3600


def jwt_secret() -> str:
    s = os.environ.get("JWT_SECRET") or os.environ.get("VOXA_KMS_KEY")
    if not s:
        raise RuntimeError("JWT_SECRET (or VOXA_KMS_KEY) must be set")
    return s


def api_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def make_jwt(user_id: str, tenant_id: str, role: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user_id, "tid": tenant_id, "role": role, "iat": now, "exp": now + JWT_TTL_S},
                      jwt_secret(), algorithm="HS256")


@dataclass(frozen=True)
class Principal:
    tenant: str
    role: str
    user_id: str | None = None      # None => API-key service principal


async def principal(request: Request, authorization: str | None = Header(None),
                    x_api_key: str | None = Header(None)) -> Principal:
    if authorization and authorization.lower().startswith("bearer "):
        try:
            c = jwt.decode(authorization[7:].strip(), jwt_secret(), algorithms=["HS256"])
        except jwt.PyJWTError:
            raise HTTPException(401, "invalid or expired token")
        return Principal(c["tid"], c["role"], c["sub"])
    if x_api_key:
        async with request.app.state.pool.acquire() as conn:
            tid = await conn.fetchval("SELECT auth_tenant($1)", api_key_hash(x_api_key))
        if tid:
            return Principal(tid, "admin")
    raise HTTPException(401, "authentication required")


def require(role: str):
    need = ROLE_LEVEL[role]

    async def dep(p: Principal = Depends(principal)) -> Principal:
        if ROLE_LEVEL.get(p.role, 0) < need:
            raise HTTPException(403, f"requires role {role}")
        return p
    return dep


async def rate_limit(request: Request, key: str, limit: int, window_s: int) -> None:
    r = request.app.state.redis
    k = f"rl:{key}:{int(time.time() // window_s)}"
    n = await r.incr(k)
    if n == 1:
        await r.expire(k, window_s + 1)
    if n > limit:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(window_s)})


def make_csat_token(tenant: str, room: str) -> str:
    now = int(time.time())
    return jwt.encode({"typ": "csat", "tid": tenant, "room": room, "iat": now, "exp": now + 3600}, jwt_secret(), algorithm="HS256")


def read_csat_token(token: str) -> tuple[str, str]:
    c = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
    if c.get("typ") != "csat":
        raise ValueError("wrong token type")
    return c["tid"], c["room"]
