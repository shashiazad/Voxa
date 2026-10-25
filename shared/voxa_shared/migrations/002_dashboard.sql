ALTER TABLE tenants ADD COLUMN IF NOT EXISTS widget_key text UNIQUE;
UPDATE tenants SET widget_key = 'wk_' || replace(gen_random_uuid()::text, '-', '') WHERE widget_key IS NULL;
ALTER TABLE tenants ALTER COLUMN widget_key SET DEFAULT ('wk_' || replace(gen_random_uuid()::text, '-', ''));
ALTER TABLE tenants ALTER COLUMN widget_key SET NOT NULL;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS allowed_origins text[] NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  email text NOT NULL,
  password_hash text NOT NULL,
  role text NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS users_email_ci ON users (lower(email));
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON users;
CREATE POLICY tenant_isolation ON users USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
GRANT SELECT, INSERT, UPDATE, DELETE ON users TO voxa_app;

-- Pre-tenant lookups: login (email -> tenant) and the public widget token endpoint (key -> tenant + origins).
CREATE OR REPLACE FUNCTION login_user(e text) RETURNS TABLE(id uuid, tenant_id text, role text, password_hash text)
  LANGUAGE sql SECURITY DEFINER SET search_path = public AS
  $$ SELECT id, tenant_id, role, password_hash FROM users WHERE lower(email) = lower(e) $$;
CREATE OR REPLACE FUNCTION widget_tenant(k text) RETURNS TABLE(id text, allowed_origins text[])
  LANGUAGE sql SECURITY DEFINER SET search_path = public AS
  $$ SELECT id, allowed_origins FROM tenants WHERE widget_key = k $$;
REVOKE ALL ON FUNCTION login_user(text), widget_tenant(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION login_user(text), widget_tenant(text) TO voxa_app;
