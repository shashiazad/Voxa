import asyncio
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from voxa_agent.escalation import EscalationEngine
from voxa_agent.tools import ToolRuntime
from voxa_shared.secrets import EnvSecretStore
from voxa_shared.tenant import load_config

ACME = Path(__file__).parents[2] / "samples/acme/tenant.yaml"


@pytest.fixture
def cfg():
    return load_config(ACME.read_text())


@pytest.fixture(autouse=True)
def secret(monkeypatch):
    monkeypatch.setenv("VOXA_SECRET__ACME__ACME_API_TOKEN", "tok123")
    monkeypatch.setenv("TOOL_ALLOW_PRIVATE", "1")   # tests use the in-cluster host name acme-mock


def runtime(cfg, handler):
    return ToolRuntime(cfg, EnvSecretStore(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_schemas_are_openai_function_format(cfg):
    rt = runtime(cfg, lambda r: httpx.Response(200, json={}))
    s = {x["function"]["name"]: x for x in rt.schemas()}
    assert set(s) == {"order_status", "create_ticket"} and s["order_status"]["type"] == "function"
    assert s["order_status"]["function"]["parameters"]["required"] == ["order_id"]


async def test_get_tool_substitutes_path_and_resolves_secret_header(cfg):
    seen = {}

    def h(req: httpx.Request):
        seen["url"], seen["tok"] = str(req.url), req.headers["x-acme-token"]
        return httpx.Response(200, json={"order_id": "AC48213", "status": "shipped"})

    out = await runtime(cfg, h).execute("order_status", '{"order_id": "AC48213"}')
    assert out.ok and out.result["status"] == "shipped"
    assert seen == {"url": "http://acme-mock:8100/orders/AC48213", "tok": "tok123"}


async def test_path_param_is_percent_encoded(cfg):
    seen = {}
    def h(req):
        seen["u"] = str(req.url); return httpx.Response(200, json={})
    await runtime(cfg, h).execute("order_status", {"order_id": "../admin?x=1"})
    assert "/orders/..%2Fadmin%3Fx%3D1" in seen["u"]


async def test_404_and_server_error_and_timeout_give_graceful_results(cfg):
    out = await runtime(cfg, lambda r: httpx.Response(404)).execute("order_status", {"order_id": "ZZ1"})
    assert out.ok and out.result == {"error": "not_found"}
    out = await runtime(cfg, lambda r: httpx.Response(500)).execute("order_status", {"order_id": "ZZ1"})
    assert not out.ok and out.result["spoken"] == cfg.scripts.tool_failure

    async def slow(req):
        await asyncio.sleep(1); return httpx.Response(200, json={})
    cfg2 = cfg.model_copy(deep=True); cfg2.tools[0].timeout_s = 0.1
    t0 = asyncio.get_event_loop().time()
    out = await runtime(cfg2, slow).execute("order_status", {"order_id": "AC1"})
    assert not out.ok and out.result["error"] == "timeout" and asyncio.get_event_loop().time() - t0 < 0.5


async def test_missing_required_and_bad_args_and_unknown_tool(cfg):
    rt = runtime(cfg, lambda r: httpx.Response(200, json={}))
    assert "missing required: order_id" in (await rt.execute("order_status", "{}")).result["error"]
    assert not (await rt.execute("order_status", "not json")).ok
    assert not (await rt.execute("nope", "{}")).ok


async def test_side_effect_runs_only_after_question_and_explicit_yes(cfg):
    calls = []
    def h(req):
        calls.append(json.loads(req.content)); return httpx.Response(200, json={"ticket_id": "T-1"})
    rt = runtime(cfg, h)
    args = {"subject": "Broken earbuds"}
    # 1. nothing asked yet: the model tries to act straight away
    rt.new_user_turn(); rt.set_context("my earbuds are broken, open a ticket", "")
    first = await rt.execute("create_ticket", args)
    assert first.needs_confirmation and "open a support ticket titled Broken earbuds" in first.result["instruction"]
    # 2. the model skips the caller's yes: user text is not an affirmation
    assert (await rt.execute("create_ticket", args)).needs_confirmation and calls == []
    # 3. agent asked a question, but the caller said no / hedged
    for u in ("no", "wait, actually don't", "not yet", "hmm"):
        rt.new_user_turn(); rt.set_context(u, "I will open a ticket. Is that okay?")
        assert (await rt.execute("create_ticket", args)).needs_confirmation
    # 4. caller says yes to a question (arguments may differ from what the model first drafted)
    rt.new_user_turn(); rt.set_context("yes please do", "I'll open a ticket for the broken earbuds. Is that okay?")
    ok = await rt.execute("create_ticket", {"subject": "Earbuds stopped working", "order_id": "AC1"})
    assert ok.ok and not ok.needs_confirmation and len(calls) == 1
    # 5. one action per confirmation
    assert (await rt.execute("create_ticket", args)).needs_confirmation and len(calls) == 1


async def test_yes_without_a_preceding_question_is_not_confirmation(cfg):
    rt = runtime(cfg, lambda r: httpx.Response(200, json={}))
    rt.new_user_turn(); rt.set_context("yes", "Your order has shipped.")
    assert (await rt.execute("create_ticket", {"subject": "x y z"})).needs_confirmation


async def test_openapi_tool_is_derived_from_spec(cfg):
    spec = {"servers": [{"url": "https://api.x.com/v1"}], "paths": {"/users/{id}/balance": {"get": {
        "operationId": "getBalance", "summary": "Balance", "parameters": [
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "currency", "in": "query", "schema": {"type": "string"}}]}}}}
    c = cfg.model_copy(deep=True)
    c.tools = [type(c.tools[0]).model_validate({
        "name": "get_balance", "description": "", "kind": "openapi", "openapi": {"spec": spec, "operation_id": "getBalance"}})]
    seen = {}
    def h(req): seen["u"] = str(req.url); return httpx.Response(200, json={"balance": 5})
    rt = ToolRuntime(c, EnvSecretStore(), httpx.AsyncClient(transport=httpx.MockTransport(h)))
    assert rt.schemas()[0]["function"]["parameters"]["required"] == ["id"]
    out = await rt.execute("get_balance", {"id": "u1", "currency": "usd"})
    assert out.ok and seen["u"] == "https://api.x.com/v1/users/u1/balance?currency=usd"


async def test_identity_verification_gate(cfg):
    c = cfg.model_copy(deep=True); c.tools[0].verify_identity = True
    rt = runtime(c, lambda r: httpx.Response(200, json={}))
    assert "not verified" in (await rt.execute("order_status", {"order_id": "A1"})).result["error"]
    rt.verified_identity = True
    assert (await rt.execute("order_status", {"order_id": "A1"})).ok


NOON_MON = datetime(2026, 10, 26, 12, 0, tzinfo=ZoneInfo("America/New_York"))
NIGHT_MON = datetime(2026, 10, 26, 23, 0, tzinfo=ZoneInfo("America/New_York"))
SUN = datetime(2026, 10, 25, 12, 0, tzinfo=ZoneInfo("America/New_York"))


@pytest.mark.parametrize("text", ["I want to talk to a human", "can you transfer me to a real person",
                                  "let me speak with a manager", "get me a representative"])
def test_human_request(cfg, text):
    d = EscalationEngine(cfg).evaluate(text, NOON_MON)
    assert d and d.reason == "human_request" and d.handoff


def test_after_hours_offers_ticket_instead_of_transfer(cfg):
    for t in (NIGHT_MON, SUN):
        d = EscalationEngine(cfg).evaluate("I need a human", t)
        assert d and d.reason == "human_request" and not d.handoff and d.line == cfg.escalation.after_hours_line


def test_restricted_topic(cfg):
    d = EscalationEngine(cfg).evaluate("I'm going to contact my attorney", NOON_MON)
    assert d and d.reason == "restricted_topic"


def test_negative_sentiment_accumulates_but_neutral_does_not(cfg):
    e = EscalationEngine(cfg)
    assert e.evaluate("where is my order please", NOON_MON) is None
    assert e.evaluate("this is ridiculous", NOON_MON) is None            # 2 < 3
    d = e.evaluate("this is a terrible awful service!", NOON_MON)
    assert d and d.reason == "sentiment"
    assert EscalationEngine(cfg).evaluate("this is f***ing useless shit", NOON_MON) is not None


def test_repeated_failures_escalate_and_success_resets(cfg):
    e = EscalationEngine(cfg)
    for _ in range(2):
        e.record_failure()
    assert e.evaluate("hello", NOON_MON) is None
    e.record_failure()
    assert e.evaluate("hello", NOON_MON).reason == "failures"
    e.record_success()
    assert e.evaluate("hello", NOON_MON) is None


@pytest.mark.parametrize("url", ["http://127.0.0.1/x", "http://169.254.169.254/latest/meta-data", "http://localhost:5432/", "http://10.0.0.5/a"])
async def test_ssrf_guard_blocks_private_targets_at_call_time(cfg, monkeypatch, url):
    monkeypatch.setenv("TOOL_ALLOW_PRIVATE", "0")
    c = cfg.model_copy(deep=True); c.tools[0].url = url + "{order_id}" if "{" not in url else url
    called = []
    rt = runtime(c, lambda r: called.append(1) or httpx.Response(200, json={}))
    out = await rt.execute("order_status", {"order_id": "A1"})
    assert not out.ok and out.result["error"] == "tool_failed" and called == []


async def test_ssrf_guard_off_in_dev(cfg, monkeypatch):
    monkeypatch.setenv("TOOL_ALLOW_PRIVATE", "1")
    assert (await runtime(cfg, lambda r: httpx.Response(200, json={"ok": 1})).execute("order_status", {"order_id": "A1"})).ok
