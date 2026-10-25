# Adding a tenant

1. Write a config (YAML or JSON) that validates against `voxa_shared.tenant.TenantConfig`
   (see `samples/acme/tenant.yaml`; unknown fields are rejected; `tenant_id` is `[a-z0-9_]{2,40}`).
2. Create it (admin token), keep the returned API key (shown once, only its hash is stored):
   ```
   curl -X POST localhost:8000/v1/admin/tenants -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H 'content-type: application/json' --data @config.json
   ```
3. Store tool secrets (Fernet-encrypted at rest; referenced in config as `{{secret:NAME}}`):
   `PUT /v1/admin/tenants/<id>/secrets/<NAME>` with `{"value": "..."}`.
4. Upload knowledge: `POST /v1/kb/documents` (PDF, DOCX, CSV FAQ, TXT/MD) or `POST /v1/kb/crawl`
   with the tenant's `X-API-Key`.
5. Mint a call: `POST /v1/rooms/token` with `X-API-Key` -> `{url, room, token}`; the agent worker
   picks the tenant from the room name prefix and loads its config on each call.

Config changes (`PUT /v1/admin/tenants/<id>/config`) apply to the next call; versions only move forward.
The sample tenant is seeded by `docker compose up` (service `seed`), key `ACME_API_KEY` in `.env`.
