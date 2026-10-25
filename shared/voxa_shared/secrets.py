"""Per-tenant secret storage. Secrets are Fernet-encrypted at rest; the key comes from a KMS/env
(VOXA_KMS_KEY). Plaintext never reaches logs: callers get it only at request time."""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from cryptography.fernet import Fernet

_REF = re.compile(r"\{\{secret:([A-Za-z0-9_.-]+)\}\}")


class SecretStore(ABC):
    @abstractmethod
    async def get(self, tenant_id: str, name: str) -> str | None: ...

    async def resolve(self, tenant_id: str, value: str) -> str:
        """Replace {{secret:NAME}} references; unknown secrets raise (never send a literal placeholder)."""
        out, pos = [], 0
        for m in _REF.finditer(value):
            s = await self.get(tenant_id, m.group(1))
            if s is None:
                raise KeyError(f"secret {m.group(1)!r} not found")
            out += [value[pos:m.start()], s]
            pos = m.end()
        out.append(value[pos:])
        return "".join(out)


class EnvSecretStore(SecretStore):
    """Dev/CI: VOXA_SECRET__<TENANT>__<NAME> environment variables."""

    async def get(self, tenant_id: str, name: str) -> str | None:
        return os.environ.get(f"VOXA_SECRET__{tenant_id.upper()}__{name.upper()}")


class DbSecretStore(SecretStore):
    def __init__(self, pool, key: str | None = None) -> None:
        from .db import tenant_conn

        self._pool, self._tc = pool, tenant_conn
        k = key or os.environ.get("VOXA_KMS_KEY")
        if not k:
            raise RuntimeError("VOXA_KMS_KEY is required")
        self._f = Fernet(k.encode())

    async def get(self, tenant_id: str, name: str) -> str | None:
        async with self._tc(self._pool, tenant_id) as c:
            row = await c.fetchrow("SELECT ciphertext FROM secrets WHERE name=$1", name)
        return self._f.decrypt(bytes(row["ciphertext"])).decode() if row else None

    async def put(self, tenant_id: str, name: str, value: str) -> None:
        async with self._tc(self._pool, tenant_id) as c:
            await c.execute(
                "INSERT INTO secrets(tenant_id,name,ciphertext) VALUES($1,$2,$3) "
                "ON CONFLICT (tenant_id,name) DO UPDATE SET ciphertext=EXCLUDED.ciphertext",
                tenant_id, name, self._f.encrypt(value.encode()))
