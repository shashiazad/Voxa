"""Conversation persistence (redacted) without ever blocking the audio path: a single writer task drains a queue."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from voxa_shared.db import tenant_conn

log = logging.getLogger("voxa.convlog")


class ConversationLog:
    def __init__(self, pool, tenant_id: str, room: str, consent_recording: bool = False) -> None:
        self.pool, self.tenant_id, self.room, self.consent = pool, tenant_id, room, consent_recording
        self.id: str | None = None
        self._q: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue(maxsize=500)
        self._idx = 0
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        async with tenant_conn(self.pool, self.tenant_id) as c:
            self.id = str(await c.fetchval(
                "INSERT INTO conversations(tenant_id, room, consent_recording) VALUES($1,$2,$3) RETURNING id",
                self.tenant_id, self.room, self.consent))
        self._task = asyncio.create_task(self._writer(), name="convlog")

    def turn(self, rec: dict[str, Any]) -> None:
        try:
            self._q.put_nowait(("turn", rec))
        except asyncio.QueueFull:
            log.warning("conversation log queue full; dropping a turn")

    def finish(self, **fields: Any) -> None:
        self._q.put_nowait(("finish", fields))

    async def set_csat(self, score: int) -> None:
        async with tenant_conn(self.pool, self.tenant_id) as c:
            await c.execute("UPDATE conversations SET csat=$2 WHERE id=$1", self.id, score)

    async def set_consent(self, granted: bool) -> None:
        async with tenant_conn(self.pool, self.tenant_id) as c:
            await c.execute("UPDATE conversations SET consent_recording=$2 WHERE id=$1", self.id, granted)

    async def add_recording(self, key: str, seconds: float, size: int) -> None:
        async with tenant_conn(self.pool, self.tenant_id) as c:
            await c.execute("INSERT INTO recordings(tenant_id, conversation_id, object_key, duration_s, bytes) VALUES($1,$2,$3,$4,$5)",
                            self.tenant_id, self.id, key, seconds, size)

    async def close(self) -> None:
        await self._q.put(None)
        if self._task:
            await asyncio.wait_for(self._task, 5)

    async def _writer(self) -> None:
        while (item := await self._q.get()) is not None:
            kind, rec = item
            try:
                async with tenant_conn(self.pool, self.tenant_id) as c:
                    if kind == "turn":
                        metrics = {k: v for k, v in rec.items() if k not in ("role", "text", "interrupted")}
                        await c.execute(
                            "INSERT INTO turns(tenant_id, conversation_id, idx, role, text, interrupted, metrics) "
                            "VALUES($1,$2,$3,$4,$5,$6,$7::jsonb)", self.tenant_id, self.id, self._idx,
                            rec["role"], rec["text"], bool(rec.get("interrupted")), json.dumps(metrics))
                        self._idx += 1
                    else:
                        await c.execute(
                            "UPDATE conversations SET ended_at=now(), handoff=COALESCE($2,handoff), "
                            "summary=COALESCE($3,summary) WHERE id=$1",
                            self.id, rec.get("handoff"), rec.get("summary"))
            except Exception:  # noqa: BLE001
                log.exception("conversation log write failed")
