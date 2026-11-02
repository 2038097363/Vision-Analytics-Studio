-- PostgreSQL 16+ draft. Apply through a versioned migration tool in production.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email_normalized text NOT NULL UNIQUE,
  display_name text NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 80),
  password_hash text NOT NULL,
  email_verified_at timestamptz,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','active','locked','deleting')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz
);

CREATE TABLE devices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  display_name text NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 120),
  platform text NOT NULL,
  last_seen_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE refresh_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  family_id uuid NOT NULL,
  token_hash bytea NOT NULL UNIQUE,
  access_token_hash bytea NOT NULL UNIQUE,
  access_expires_at timestamptz NOT NULL,
  expires_at timestamptz NOT NULL,
  rotated_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE category_preferences (
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  category_id text NOT NULL,
  enabled boolean NOT NULL,
  manifest_version integer NOT NULL,
  version bigint NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, category_id)
);

CREATE TABLE photo_records (
  id uuid PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  client_photo_id uuid NOT NULL,
  safe_category_id text,
  classification_source text CHECK (classification_source IN ('local_model','manual')),
  classification_revision text,
  cloud_sync_enabled boolean NOT NULL DEFAULT false,
  version bigint NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  UNIQUE (user_id, device_id, client_photo_id)
);

CREATE TABLE cloud_objects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  photo_id uuid NOT NULL REFERENCES photo_records(id) ON DELETE CASCADE,
  object_key text NOT NULL UNIQUE,
  sha256 bytea NOT NULL,
  size_bytes bigint NOT NULL CHECK (size_bytes > 0),
  mime_type text NOT NULL,
  status text NOT NULL CHECK (status IN ('upload_requested','uploading','uploaded','delete_requested','deleted')),
  created_at timestamptz NOT NULL DEFAULT now(),
  verified_at timestamptz,
  deleted_at timestamptz
);

CREATE TABLE audit_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id uuid REFERENCES users(id) ON DELETE SET NULL,
  request_id uuid NOT NULL,
  event_type text NOT NULL,
  outcome text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE deletion_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  object_id uuid,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','succeeded','failed')),
  attempt_count integer NOT NULL DEFAULT 0,
  next_attempt_at timestamptz NOT NULL DEFAULT now(),
  last_error_code text,
  created_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz
);

CREATE INDEX idx_devices_user ON devices(user_id, revoked_at);
CREATE INDEX idx_refresh_sessions_user ON refresh_sessions(user_id, revoked_at, expires_at);
CREATE INDEX idx_photo_records_sync ON photo_records(user_id, version) WHERE deleted_at IS NULL;
CREATE INDEX idx_cloud_objects_status ON cloud_objects(status, created_at);
CREATE INDEX idx_deletion_jobs_due ON deletion_jobs(status, next_attempt_at);

-- The runtime role must not own these tables and must not have BYPASSRLS.
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE devices FORCE ROW LEVEL SECURITY;
ALTER TABLE refresh_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE refresh_sessions FORCE ROW LEVEL SECURITY;
ALTER TABLE category_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE category_preferences FORCE ROW LEVEL SECURITY;
ALTER TABLE photo_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE photo_records FORCE ROW LEVEL SECURITY;
ALTER TABLE cloud_objects ENABLE ROW LEVEL SECURITY;
ALTER TABLE cloud_objects FORCE ROW LEVEL SECURITY;

CREATE POLICY devices_owner ON devices
  USING (user_id = current_setting('app.user_id', true)::uuid)
  WITH CHECK (user_id = current_setting('app.user_id', true)::uuid);
CREATE POLICY refresh_sessions_owner ON refresh_sessions
  USING (user_id = current_setting('app.user_id', true)::uuid)
  WITH CHECK (user_id = current_setting('app.user_id', true)::uuid);
CREATE POLICY category_preferences_owner ON category_preferences
  USING (user_id = current_setting('app.user_id', true)::uuid)
  WITH CHECK (user_id = current_setting('app.user_id', true)::uuid);
CREATE POLICY photo_records_owner ON photo_records
  USING (user_id = current_setting('app.user_id', true)::uuid)
  WITH CHECK (user_id = current_setting('app.user_id', true)::uuid);
CREATE POLICY cloud_objects_owner ON cloud_objects
  USING (user_id = current_setting('app.user_id', true)::uuid)
  WITH CHECK (user_id = current_setting('app.user_id', true)::uuid);

-- users, audit_events, and deletion_jobs require narrow service functions/roles rather than general runtime grants.
