CREATE EXTENSION IF NOT EXISTS vector;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'voxa_app') THEN
    CREATE ROLE voxa_app LOGIN PASSWORD 'voxa_app' NOSUPERUSER NOBYPASSRLS;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS tenants (
  id text PRIMARY KEY,
  name text NOT NULL,
  api_key_hash text UNIQUE NOT NULL,
  config jsonb NOT NULL,
  config_version int NOT NULL DEFAULT 1,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  source text NOT NULL,
  title text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_chunks (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  document_id uuid NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
  ord int NOT NULL,
  content text NOT NULL,
  embedding vector(384) NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS kb_chunks_tenant ON kb_chunks(tenant_id);
CREATE INDEX IF NOT EXISTS kb_chunks_tsv ON kb_chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS kb_chunks_vec ON kb_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS secrets (
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  name text NOT NULL,
  ciphertext bytea NOT NULL,
  PRIMARY KEY (tenant_id, name)
);

CREATE TABLE IF NOT EXISTS conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  room text NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  csat int,
  consent_recording boolean NOT NULL DEFAULT false,
  handoff boolean NOT NULL DEFAULT false,
  summary text
);
CREATE TABLE IF NOT EXISTS turns (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  idx int NOT NULL,
  role text NOT NULL,
  text text NOT NULL,
  interrupted boolean NOT NULL DEFAULT false,
  metrics jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS tool_calls (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  conversation_id uuid REFERENCES conversations(id) ON DELETE CASCADE,
  name text NOT NULL,
  arguments jsonb,
  result jsonb,
  ok boolean NOT NULL,
  duration_ms int NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Row-level security: every tenant-owned table is filtered by app.tenant_id, even for the table owner.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['tenants','kb_documents','kb_chunks','secrets','conversations','turns','tool_calls'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
    IF t = 'tenants' THEN
      EXECUTE 'CREATE POLICY tenant_isolation ON tenants USING (id = current_setting(''app.tenant_id'', true)) WITH CHECK (id = current_setting(''app.tenant_id'', true))';
    ELSE
      EXECUTE format('CREATE POLICY tenant_isolation ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))', t);
    END IF;
  END LOOP;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO voxa_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO voxa_app;

-- Pre-tenant lookups (API-key auth, worker prewarm) cannot know the tenant yet; narrow SECURITY DEFINER
-- functions owned by the admin role expose exactly what is needed and nothing else.
CREATE OR REPLACE FUNCTION auth_tenant(h text) RETURNS text
  LANGUAGE sql SECURITY DEFINER SET search_path = public AS
  $$ SELECT id FROM tenants WHERE api_key_hash = h $$;
CREATE OR REPLACE FUNCTION all_tenant_configs() RETURNS TABLE(id text, config jsonb)
  LANGUAGE sql SECURITY DEFINER SET search_path = public AS
  $$ SELECT id, config FROM tenants $$;
REVOKE ALL ON FUNCTION auth_tenant(text), all_tenant_configs() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION auth_tenant(text), all_tenant_configs() TO voxa_app;
