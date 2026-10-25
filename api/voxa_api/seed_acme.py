"""Seed the sample tenant 'acme' (config, secret, knowledge base). Idempotent: re-running replaces the KB."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
from voxa_shared.tenant import load_config

BASE = os.environ.get("API_URL", "http://localhost:8000")
ADMIN = {"Authorization": "Bearer " + os.environ.get("ADMIN_TOKEN", "dev-admin-token"), "Content-Type": "application/json"}
KEY = os.environ.get("ACME_API_KEY", "acme_dev_key_change_me")
SAMPLES = Path(os.environ.get("SAMPLES_DIR", "/app/samples/acme"))


async def main() -> None:
    cfg = load_config((SAMPLES / "tenant.yaml").read_text())
    async with httpx.AsyncClient(base_url=BASE, timeout=120) as c:
        for _ in range(60):
            try:
                if (await c.get("/healthz")).status_code == 200:
                    break
            except httpx.HTTPError:
                await asyncio.sleep(1)
        r = await c.post("/v1/admin/tenants", params={"api_key": KEY}, headers=ADMIN, content=cfg.model_dump_json())
        if r.status_code == 409:
            r = await c.put(f"/v1/admin/tenants/{cfg.tenant_id}/config", headers=ADMIN, content=cfg.model_dump_json())
        r.raise_for_status()
        (await c.put(f"/v1/admin/tenants/{cfg.tenant_id}/secrets/acme_api_token", headers=ADMIN,
                     json={"value": os.environ.get("ACME_API_TOKEN", "acme-dev-token")})).raise_for_status()
        pw = os.environ.get("ACME_ADMIN_PASSWORD", "acme-admin-pass-123")
        for email, role in (("admin@acme.test", "admin"), ("editor@acme.test", "editor"), ("viewer@acme.test", "viewer")):
            (await c.put(f"/v1/admin/tenants/{cfg.tenant_id}/users", headers=ADMIN,
                         json={"email": email, "password": pw, "role": role})).raise_for_status()
        h = {"X-API-Key": KEY}
        origins = os.environ.get("ACME_WIDGET_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
        (await c.put("/v1/widget", headers=h, json={"allowed_origins": origins})).raise_for_status()
        for d in (await c.get("/v1/kb/documents", headers=h)).json():
            await c.delete(f"/v1/kb/documents/{d['id']}", headers=h)
        for k in cfg.knowledge:
            p = SAMPLES / k.uri
            (await c.post("/v1/kb/documents", headers=h, files={"file": (p.name, p.read_bytes())})).raise_for_status()
        docs = (await c.get("/v1/kb/documents", headers=h)).json()
        wk = (await c.get("/v1/widget", headers=h)).json()["widget_key"]
        print(f"widget key: {wk}")
        print(f"seeded tenant {cfg.tenant_id}: {len(docs)} documents, {sum(d['chunks'] for d in docs)} chunks")


if __name__ == "__main__":
    asyncio.run(main())
