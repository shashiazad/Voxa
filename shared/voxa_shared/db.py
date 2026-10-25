"""asyncpg helpers: tenant-scoped connections (RLS) and the migration runner."""
from __future__ import annotations

import importlib.resources
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from pgvector.asyncpg import register_vector


async def _init(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def create_pool(dsn: str, min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init)


@asynccontextmanager
async def tenant_conn(pool: asyncpg.Pool, tenant_id: str) -> AsyncIterator[asyncpg.Connection]:
    """Connection whose every statement is confined to `tenant_id` by RLS. The setting is
    transaction-local (set_config(..., true)), so it can never leak to the next pool user."""
    async with pool.acquire() as c:
        async with c.transaction():
            await c.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            yield c


async def migrate_db(admin_dsn: str) -> list[str]:
    """Apply migrations/*.sql in order (idempotent). Needs the admin role."""
    conn = await asyncpg.connect(admin_dsn)
    applied: list[str] = []
    try:
        await conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations(name text PRIMARY KEY)")
        done = {r["name"] for r in await conn.fetch("SELECT name FROM schema_migrations")}
        files = sorted(f for f in importlib.resources.files("voxa_shared.migrations").iterdir()
                       if f.name.endswith(".sql"))
        for f in files:
            if f.name in done:
                continue
            async with conn.transaction():
                await conn.execute(f.read_text())
                await conn.execute("INSERT INTO schema_migrations VALUES($1)", f.name)
            applied.append(f.name)
    finally:
        await conn.close()
    return applied
