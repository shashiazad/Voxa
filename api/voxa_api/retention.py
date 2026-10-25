"""Per-tenant data-retention sweep: deletes conversations (and their recordings) older than each tenant's
`guardrails.retention_days`. Run daily (compose: service `retention`; Helm: CronJob). Idempotent."""
from __future__ import annotations

import asyncio
import json
import logging
import os

import asyncpg

from voxa_shared import storage

log = logging.getLogger("voxa.retention")


async def sweep(dsn: str) -> dict[str, int]:
    c = await asyncpg.connect(dsn)
    totals = {"conversations": 0, "recordings": 0}
    try:
        for t in await c.fetch("SELECT id, config FROM tenants"):
            days = json.loads(t["config"]).get("guardrails", {}).get("retention_days", 30)
            # 0 = keep nothing beyond the call: purge once a conversation has been over for 10 minutes
            cutoff = "now() - interval '10 minutes'" if days == 0 else f"now() - make_interval(days => {int(days)})"
            where = f"c.tenant_id=$1 AND COALESCE(c.ended_at, c.started_at) < {cutoff}"
            keys = [r["object_key"] for r in await c.fetch(
                f"SELECT r.object_key FROM recordings r JOIN conversations c ON c.id=r.conversation_id WHERE {where}", t["id"])]
            n = await c.fetchval(f"WITH d AS (DELETE FROM conversations c WHERE {where} RETURNING 1) SELECT count(*) FROM d", t["id"])
            for k in keys:
                try:
                    storage.delete(k)
                except Exception:  # noqa: BLE001
                    log.warning("could not delete object %s", k)
            totals["conversations"] += n; totals["recordings"] += len(keys)
            if n:
                log.info("tenant %s: purged %s conversations, %s recordings (retention %s d)", t["id"], n, len(keys), days)
    finally:
        await c.close()
    return totals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dsn = os.environ.get("DATABASE_ADMIN_URL", "postgresql://voxa:voxa@postgres:5432/voxa")
    loop_s = int(os.environ.get("RETENTION_INTERVAL_S", "0"))     # 0 = run once (CronJob), >0 = loop (compose)
    import time
    while True:
        try:
            print(asyncio.run(sweep(dsn)), flush=True)
        except Exception as e:  # noqa: BLE001 - DB still migrating / storage down: try again next interval
            log.warning("retention sweep failed: %s", e)
            if not loop_s:
                raise
        if not loop_s:
            break
        time.sleep(loop_s if loop_s else 0)
