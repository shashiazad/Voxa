CREATE TABLE IF NOT EXISTS recordings (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  object_key text NOT NULL,
  duration_s real,
  bytes bigint,
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE recordings ENABLE ROW LEVEL SECURITY;
ALTER TABLE recordings FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON recordings;
CREATE POLICY tenant_isolation ON recordings USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
GRANT SELECT, INSERT, UPDATE, DELETE ON recordings TO voxa_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO voxa_app;
-- the retention job (admin role) needs object keys across tenants: ordinary admin SELECT, no function needed
