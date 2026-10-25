"""API behaviour: RBAC matrix, login, widget origin policy, tenant isolation, rate limits, voices, analytics."""
import json
import os

import httpx
import pytest
import pytest_asyncio

from conftest import ADMIN, APP, ROOT, make_tenant

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client(pools):
    os.environ.update(DATABASE_URL=APP, DATABASE_ADMIN_URL=ADMIN, LIVEKIT_API_KEY="devkey", ADMIN_TOKEN="adm",
                      LIVEKIT_API_SECRET="x" * 40, JWT_SECRET="j" * 40, SIGNUP_ENABLED="1", TOOL_ALLOW_PRIVATE="1",
                      REDIS_URL="redis://localhost:6379/6", VOICE_DIR=str(ROOT / "models"))
    from voxa_api.main import app
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            yield c


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def _fresh_rate_limits(client):
    from voxa_api.main import app
    await app.state.redis.flushdb()


async def mk(client, tid, role="admin", email=None, pw="correct-horse-123"):
    cfg = json.loads((ROOT / "samples/acme/tenant.yaml").read_text() and
                     __import__("voxa_shared.tenant", fromlist=["load_config"]).load_config(
                         (ROOT / "samples/acme/tenant.yaml").read_text()).model_copy(update={"tenant_id": tid}).model_dump_json())
    r = await client.post("/v1/admin/tenants", params={"api_key": f"key-{tid}"}, json=cfg, headers={"Authorization": "Bearer adm"})
    assert r.status_code in (200, 409), r.text
    email = email or f"{role}@{tid}.test"
    r = await client.put(f"/v1/admin/tenants/{tid}/users", json={"email": email, "password": pw, "role": role},
                         headers={"Authorization": "Bearer adm"})
    assert r.status_code == 204, r.text
    r = await client.post("/v1/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


async def test_login_rejects_bad_credentials_and_unknown_users(client):
    await mk(client, "t_login")
    for body in ({"email": "admin@t_login.test", "password": "wrong"}, {"email": "ghost@x.test", "password": "whatever"}):
        assert (await client.post("/v1/auth/login", json=body)).status_code == 401


async def test_rbac_matrix(client):
    tid = "t_rbac"
    admin = await mk(client, tid, "admin")
    editor = await mk(client, tid, "editor")
    viewer = await mk(client, tid, "viewer")
    cfg = (await client.get("/v1/config", headers=viewer)).json()
    cases = [  # method, path, kwargs, min role
        ("GET", "/v1/config", {}, "viewer"), ("GET", "/v1/conversations", {}, "viewer"),
        ("GET", "/v1/analytics/summary", {}, "viewer"), ("GET", "/v1/kb/documents", {}, "viewer"),
        ("PUT", "/v1/config", {"json": cfg}, "editor"), ("POST", "/v1/rooms/token", {}, "editor"),
        ("GET", "/v1/users", {}, "admin"), ("PUT", "/v1/widget", {"json": {"allowed_origins": ["https://a.com"]}}, "admin"),
        ("POST", "/v1/widget/rotate", {}, "admin"),
    ]
    rank = {"viewer": 1, "editor": 2, "admin": 3}
    for method, path, kw, need in cases:
        for role, h in (("viewer", viewer), ("editor", editor), ("admin", admin)):
            r = await client.request(method, path, headers=h, **kw)
            if rank[role] >= rank[need]:
                assert r.status_code < 300, (method, path, role, r.text)
            else:
                assert r.status_code == 403, (method, path, role, r.status_code)
    assert (await client.get("/v1/config")).status_code == 401                       # anonymous
    assert (await client.get("/v1/config", headers={"Authorization": "Bearer junk"})).status_code == 401


async def test_tenants_cannot_see_each_others_data_through_the_api(client):
    a = await mk(client, "t_iso_a")
    b = await mk(client, "t_iso_b")
    r = await client.put("/v1/config", headers=a, json={**(await client.get("/v1/config", headers=a)).json(), "tenant_id": "t_iso_b"})
    assert r.status_code == 400                                                       # cannot write another tenant's config
    assert (await client.get("/v1/config", headers=a)).json()["tenant_id"] == "t_iso_a"
    assert (await client.get("/v1/users", headers=a)).json()[0]["email"] == "admin@t_iso_a.test"
    assert all("t_iso_b" not in u["email"] for u in (await client.get("/v1/users", headers=a)).json())
    r = await client.post("/v1/handoffs/t_iso_b-abc/join", headers=a)
    assert r.status_code == 404


async def test_config_validation_errors_are_422_and_ssrf_guard(client, monkeypatch):
    h = await mk(client, "t_cfg")
    cfg = (await client.get("/v1/config", headers=h)).json()
    bad = json.loads(json.dumps(cfg)); bad["turn"]["vad_threshold"] = 7
    assert (await client.put("/v1/config", headers=h, json=bad)).status_code == 422
    monkeypatch.setenv("TOOL_ALLOW_PRIVATE", "0")
    ssrf = json.loads(json.dumps(cfg)); ssrf["tools"][0]["url"] = "http://127.0.0.1:5432/{order_id}"
    r = await client.put("/v1/config", headers=h, json=ssrf)
    assert r.status_code == 422 and "private" in r.text
    ok = (await client.put("/v1/config", headers=h, json=cfg))
    assert ok.status_code in (200, 422)     # acme-mock host does not resolve in tests: guard says so, not a crash
    monkeypatch.setenv("TOOL_ALLOW_PRIVATE", "1")
    v1 = (await client.put("/v1/config", headers=h, json=cfg)).json()["version"]
    v2 = (await client.put("/v1/config", headers=h, json=cfg)).json()["version"]
    assert v2 == v1 + 1                                                               # versions only move forward


async def test_widget_token_origin_policy(client):
    h = await mk(client, "t_widget")
    key = (await client.get("/v1/widget", headers=h)).json()["widget_key"]
    wk = {"X-Widget-Key": key}
    assert (await client.post("/v1/widget/token", headers={**wk, "Origin": "https://shop.com"})).status_code == 403   # none configured
    assert (await client.put("/v1/widget", headers=h, json={"allowed_origins": ["https://shop.com/"]})).json()["allowed_origins"] == ["https://shop.com"]
    ok = await client.post("/v1/widget/token", headers={**wk, "Origin": "https://shop.com"})
    assert ok.status_code == 200 and ok.json()["room"].startswith("t_widget-")
    assert (await client.post("/v1/widget/token", headers={**wk, "Origin": "https://evil.com"})).status_code == 403
    assert (await client.post("/v1/widget/token", headers={"X-Widget-Key": "wk_nope", "Origin": "https://shop.com"})).status_code == 401
    assert (await client.put("/v1/widget", headers=h, json={"allowed_origins": ["javascript:alert(1)"]})).status_code == 422
    new = (await client.post("/v1/widget/rotate", headers=h)).json()["widget_key"]
    assert (await client.post("/v1/widget/token", headers={**wk, "Origin": "https://shop.com"})).status_code == 401    # old key dead
    assert (await client.post("/v1/widget/token", headers={"X-Widget-Key": new, "Origin": "https://shop.com"})).status_code == 200


async def test_widget_token_is_rate_limited(client):
    h = await mk(client, "t_rl")
    await client.put("/v1/widget", headers=h, json={"allowed_origins": ["https://x.com"]})
    key = (await client.get("/v1/widget", headers=h)).json()["widget_key"]
    codes = [(await client.post("/v1/widget/token", headers={"X-Widget-Key": key, "Origin": "https://x.com"})).status_code
             for _ in range(25)]
    assert 429 in codes and codes[0] == 200


async def test_signup_creates_isolated_tenant_with_admin(client):
    r = await client.post("/v1/signup", json={"org_name": "Globex Corp", "email": "ceo@globex.test", "password": "long-enough-pass"})
    assert r.status_code == 200 and r.json()["role"] == "admin" and r.json()["tenant_id"].startswith("globex_corp_")
    h = {"Authorization": "Bearer " + r.json()["token"]}
    assert (await client.get("/v1/config", headers=h)).json()["identity"]["org_name"] == "Globex Corp"
    assert (await client.post("/v1/signup", json={"org_name": "Other", "email": "CEO@globex.test", "password": "long-enough-pass"})).status_code == 409
    assert (await client.post("/v1/signup", json={"org_name": "Short", "email": "a@b.co", "password": "short"})).status_code == 422


async def test_voices_list_and_preview_returns_wav(client):
    h = await mk(client, "t_voice", "viewer")
    names = [v["id"] for v in (await client.get("/v1/voices", headers=h)).json()]
    assert {"amy", "ryan", "lessac"} <= set(names)
    r = await client.post("/v1/voices/preview", headers=h, json={"voice_id": "ryan", "text": "Hello there", "rate": 1.2})
    assert r.status_code == 200 and r.content[:4] == b"RIFF" and len(r.content) > 8000


async def test_analytics_and_conversation_endpoints_over_logged_data(client, pools):
    admin_pool, app_pool = pools
    h = await mk(client, "t_ana")
    from voxa_shared.db import tenant_conn
    async with tenant_conn(app_pool, "t_ana") as c:
        cid = await c.fetchval("INSERT INTO conversations(tenant_id,room,handoff,csat) VALUES('t_ana','t_ana-r1',true,5) RETURNING id")
        for i, (role, text, m, intr) in enumerate([("user", "hi", None, False),
                                                   ("assistant", "hello", {"latency": {"voice_to_voice_ms": 900.0}}, False),
                                                   ("user", "wait", None, False),
                                                   ("assistant", "ok [interrupted]", {"latency": {"voice_to_voice_ms": 1300.0}}, True),
                                                   ("tool", "order_status ok=True", {"tool": "order_status"}, False)]):
            await c.execute("INSERT INTO turns(tenant_id,conversation_id,idx,role,text,interrupted,metrics) VALUES('t_ana',$1,$2,$3,$4,$5,$6::jsonb)",
                            cid, i, role, text, intr, json.dumps(m) if m else None)
    s = (await client.get("/v1/analytics/summary", headers=h)).json()
    assert s["calls"] == 1 and s["handoffs"] == 1 and s["csat_avg"] == 5 and s["interrupted_replies"] == 1
    assert s["tool_calls"] == 1 and 900 <= s["v2v_p50_ms"] <= 1300 and s["v2v_samples"] == 2
    lst = (await client.get("/v1/conversations", headers=h)).json()
    assert lst[0]["interruptions"] == 1 and lst[0]["tool_calls"] == 1 and lst[0]["p50_v2v_ms"] == 1100
    d = (await client.get(f"/v1/conversations/{cid}", headers=h)).json()
    assert [t["role"] for t in d["turns"]] == ["user", "assistant", "user", "assistant", "tool"]
    other = await mk(client, "t_ana2")
    assert (await client.get(f"/v1/conversations/{cid}", headers=other)).status_code == 404


async def test_kb_upload_validation(client):
    h = await mk(client, "t_kb", "editor")
    r = await client.post("/v1/kb/documents", headers=h, files={"file": ("x.exe", b"MZ")})
    assert r.status_code == 415
    r = await client.post("/v1/kb/documents", headers=h, files={"file": ("faq.csv", b'question,answer\n"Where?","Here."\n')})
    assert r.status_code == 200 and r.json()["chunks"] == 1
    docs = (await client.get("/v1/kb/documents", headers=h)).json()
    assert docs[0]["title"] == "faq.csv"
    assert (await client.delete(f"/v1/kb/documents/{docs[0]['id']}", headers=h)).status_code == 204


async def test_csat_token_flow(client, pools):
    admin_pool, app_pool = pools
    h = await mk(client, "t_csat")
    await client.put("/v1/widget", headers=h, json={"allowed_origins": ["https://x.com"]})
    key = (await client.get("/v1/widget", headers=h)).json()["widget_key"]
    tok = (await client.post("/v1/widget/token", headers={"X-Widget-Key": key, "Origin": "https://x.com"})).json()
    from voxa_shared.db import tenant_conn
    async with tenant_conn(app_pool, "t_csat") as c:
        await c.execute("INSERT INTO conversations(tenant_id, room) VALUES('t_csat', $1)", tok["room"])
    assert (await client.post("/v1/widget/csat", json={"token": tok["csat_token"], "score": 4})).status_code == 204
    assert (await client.post("/v1/widget/csat", json={"token": tok["csat_token"], "score": 9})).status_code == 422
    assert (await client.post("/v1/widget/csat", json={"token": "junk", "score": 4})).status_code == 401
    assert (await client.post("/v1/widget/csat", json={"token": tok["token"], "score": 4})).status_code == 401   # a room token is not a rating token
    assert (await client.get("/v1/conversations", headers=h)).json()[0]["csat"] == 4
