"""Session resume: conversation state lives in Redis with a 60 s TTL that a heartbeat keeps fresh while the call is
alive. If the agent process dies, the API re-dispatches a worker into the same room (`/v1/widget/resume`); the new
job finds this state, skips the greeting and carries on with the history."""
from __future__ import annotations

import asyncio
import json
import logging
import time

log = logging.getLogger("voxa.resume")
TTL_S = 60


class SessionStore:
    def __init__(self, redis, room: str, tenant_id: str) -> None:
        self.r, self.key, self.tenant_id = redis, f"session:{room}", tenant_id
        self._task: asyncio.Task[None] | None = None
        self._snapshot: dict = {}

    async def load(self) -> dict | None:
        raw = await self.r.get(self.key)
        if not raw:
            return None
        d = json.loads(raw)
        return d if d.get("tenant") == self.tenant_id else None      # never resume across tenants

    def update(self, history: list[dict[str, str]], consent: bool | None = None) -> None:
        self._snapshot = {"tenant": self.tenant_id, "history": history[-30:], "ts": time.time(), "consent": consent}

    async def _save(self) -> None:
        if self._snapshot:
            await self.r.set(self.key, json.dumps(self._snapshot), ex=TTL_S)

    def start_heartbeat(self, every_s: float = 15.0) -> None:
        async def run() -> None:
            while True:
                try:
                    await self._save()
                except Exception:  # noqa: BLE001 - Redis hiccup must never affect the call
                    log.warning("session state write failed")
                await asyncio.sleep(every_s)
        self._task = asyncio.create_task(run())

    async def save_now(self) -> None:
        try:
            await self._save()
        except Exception:  # noqa: BLE001
            pass

    async def close(self, clean_end: bool) -> None:
        if self._task:
            self._task.cancel()
        if clean_end:                        # a finished call must not be resumable
            await self.r.delete(self.key)
