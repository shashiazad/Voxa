"""Retention sweep, export, erasure and tenant offboarding (GDPR / DPDP), including recording objects."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from conftest import make_tenant
from test_api import client, mk  # noqa: F401  (fixtures/helpers)
from voxa_shared.db import tenant_conn

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture
def storage_spy(monkeypatch):
    deleted = []
    from voxa_shared import storage
    monkeypatch.setattr(storage, "delete", lambda k: deleted.append(k))
    monkeypatch.setattr(storage, "presign_get", lambda k, e=300: f"https://s3.test/{k}?sig=1")
    return deleted


async def add_conv(app_pool, tid, room, started, ended=None, recording=True):
    async with tenant_conn(app_pool, tid) as c:
        cid = await c.fetchval("INSERT INTO conversations(tenant_id, room, started_at, ended_at, consent_recording) "
                               "VALUES($1,$2,$3,$4,$5) RETURNING id", tid, room, started, ended, recording)
        await c.execute("INSERT INTO turns(tenant_id, conversation_id, idx, role, text) VALUES($1,$2,0,'user','hi')", tid, cid)
        if recording:
            await c.execute("INSERT INTO recordings(tenant_id, conversation_id, object_key, duration_s, bytes) VALUES($1,$2,$3,1.0,10)",
                            tid, cid, f"recordings/{tid}/{room}.wav")
    return cid


async def test_export_recording_link_and_single_erasure(client, pools, storage_spy):
    admin_pool, app_pool = pools
    h = await mk(client, "t_priv1")
    now = datetime.now(timezone.utc)
    cid = await add_conv(app_pool, "t_priv1", "t_priv1-a", now)
    plain = await add_conv(app_pool, "t_priv1", "t_priv1-b", now, recording=False)
    exp = (await client.get(f"/v1/conversations/{cid}/export", headers=h)).json()
    assert exp["turns"][0]["text"] == "hi" and exp["recordings"][0]["object_key"].endswith("t_priv1-a.wav")
    assert (await client.get(f"/v1/conversations/{cid}/recording", headers=h)).json()["url"].startswith("https://s3.test/recordings/t_priv1/")
    assert (await client.get(f"/v1/conversations/{plain}/recording", headers=h)).status_code == 404
    viewer = await mk(client, "t_priv1", "viewer")
    assert (await client.get(f"/v1/conversations/{cid}/export", headers=viewer)).status_code == 403      # admins only
    assert (await client.delete(f"/v1/conversations/{cid}", headers=viewer)).status_code == 403
    assert (await client.delete(f"/v1/conversations/{cid}", headers=h)).status_code == 204
    assert storage_spy == ["recordings/t_priv1/t_priv1-a.wav"]                                         # audio object removed too
    assert (await client.get(f"/v1/conversations/{cid}/export", headers=h)).status_code == 404
    async with tenant_conn(app_pool, "t_priv1") as c:
        assert await c.fetchval("SELECT count(*) FROM turns WHERE conversation_id=$1", cid) == 0        # cascade


async def test_cannot_erase_or_read_another_tenants_conversation(client, pools, storage_spy):
    admin_pool, app_pool = pools
    a, b = await mk(client, "t_priv2a"), await mk(client, "t_priv2b")
    cid = await add_conv(app_pool, "t_priv2a", "t_priv2a-x", datetime.now(timezone.utc))
    assert (await client.delete(f"/v1/conversations/{cid}", headers=b)).status_code == 404
    assert (await client.get(f"/v1/conversations/{cid}/export", headers=b)).status_code == 404
    assert storage_spy == []


async def test_bulk_erase_before_date(client, pools, storage_spy):
    admin_pool, app_pool = pools
    h = await mk(client, "t_priv3")
    now = datetime.now(timezone.utc)
    await add_conv(app_pool, "t_priv3", "t_priv3-old1", now - timedelta(days=40))
    await add_conv(app_pool, "t_priv3", "t_priv3-old2", now - timedelta(days=35))
    keep = await add_conv(app_pool, "t_priv3", "t_priv3-new", now - timedelta(days=1))
    r = await client.delete("/v1/conversations", params={"before": (now - timedelta(days=30)).isoformat()}, headers=h)
    assert r.json() == {"deleted": 2, "recordings_deleted": 2} and len(storage_spy) == 2
    assert [c["id"] for c in (await client.get("/v1/conversations", headers=h)).json()] == [str(keep)]
    assert (await client.delete("/v1/conversations", params={"before": "garbage"}, headers=h)).status_code == 422


async def test_retention_sweep_honours_each_tenants_policy(client, pools, storage_spy):
    admin_pool, app_pool = pools
    from voxa_api.retention import sweep
    from conftest import ADMIN
    keep7 = await mk(client, "t_ret7")
    keep0 = await mk(client, "t_ret0")
    for tid, days in (("t_ret7", 7), ("t_ret0", 0)):
        cfg = (await client.get("/v1/config", headers=keep7 if tid == "t_ret7" else keep0)).json()
        cfg["guardrails"]["retention_days"] = days
        assert (await client.put("/v1/config", headers=keep7 if tid == "t_ret7" else keep0, json=cfg)).status_code == 200
    now = datetime.now(timezone.utc)
    old7 = await add_conv(app_pool, "t_ret7", "t_ret7-old", now - timedelta(days=9), now - timedelta(days=9))
    new7 = await add_conv(app_pool, "t_ret7", "t_ret7-new", now - timedelta(days=3), now - timedelta(days=3))
    done0 = await add_conv(app_pool, "t_ret0", "t_ret0-done", now - timedelta(hours=1), now - timedelta(minutes=30))
    live0 = await add_conv(app_pool, "t_ret0", "t_ret0-live", now - timedelta(minutes=2), None)             # still in progress
    totals = await sweep(ADMIN)
    assert totals["conversations"] >= 2
    ids7 = {c["id"] for c in (await client.get("/v1/conversations", headers=keep7)).json()}
    ids0 = {c["id"] for c in (await client.get("/v1/conversations", headers=keep0)).json()}
    assert str(old7) not in ids7 and str(new7) in ids7
    assert str(done0) not in ids0 and str(live0) in ids0
    assert "recordings/t_ret7/t_ret7-old.wav" in storage_spy and "recordings/t_ret0/t_ret0-done.wav" in storage_spy


async def test_tenant_offboarding_deletes_everything(client, pools, storage_spy):
    admin_pool, app_pool = pools
    await mk(client, "t_gone")
    await add_conv(app_pool, "t_gone", "t_gone-1", datetime.now(timezone.utc))
    r = await client.delete("/v1/admin/tenants/t_gone", headers={"Authorization": "Bearer adm"})
    assert r.status_code == 204
    for table in ("tenants", "conversations", "turns", "recordings", "users", "kb_chunks", "secrets"):
        col = "id" if table == "tenants" else "tenant_id"
        assert await admin_pool.fetchval(f"SELECT count(*) FROM {table} WHERE {col}='t_gone'") == 0, table
    assert storage_spy == ["recordings/t_gone/t_gone-1.wav"]
    assert (await client.delete("/v1/admin/tenants/t_gone", headers={"Authorization": "Bearer nope"})).status_code == 401
