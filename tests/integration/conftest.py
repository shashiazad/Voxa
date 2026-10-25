import asyncio
import os
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

from voxa_shared.db import create_pool, migrate_db
from voxa_shared.kb.embed import Embedder, Reranker
from voxa_shared.tenant import load_config

ADMIN = os.environ.get("TEST_ADMIN_DSN", "postgresql://voxa:voxa@localhost:5432/voxa_test")
APP = os.environ.get("TEST_APP_DSN", "postgresql://voxa_app:voxa_app@localhost:5432/voxa_test")
ROOT = Path(__file__).parents[2]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pools():
    try:
        sys = await asyncpg.connect(ADMIN.rsplit("/", 1)[0] + "/postgres", timeout=3)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres not reachable: {e}")
    await sys.execute("DROP DATABASE IF EXISTS voxa_test WITH (FORCE)")
    await sys.execute("CREATE DATABASE voxa_test")
    await sys.close()
    await migrate_db(ADMIN)
    admin = await asyncpg.create_pool(ADMIN, min_size=1, max_size=2)
    app = await create_pool(APP)
    yield admin, app
    await app.close(); await admin.close()


@pytest.fixture(scope="session")
def models():
    return Embedder(), Reranker()


async def make_tenant(admin, tid: str, key: str = "k") -> None:
    cfg = load_config((ROOT / "samples/acme/tenant.yaml").read_text())
    cfg = cfg.model_copy(update={"tenant_id": tid})
    await admin.execute(
        "INSERT INTO tenants(id,name,api_key_hash,config) VALUES($1,$2,$3,$4::jsonb) ON CONFLICT DO NOTHING",
        tid, tid, f"hash-{tid}-{key}", cfg.model_dump_json())
