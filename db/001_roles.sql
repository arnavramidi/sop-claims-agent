-- The three accounts. Runs as the superuser, before the schema is rebuilt.
-- Roles belong to the whole server, so they survive the schema drop: create each one only if missing.
-- Every run re-states its attributes and password, so a stale password or a role created by hand with
-- more rights (superuser, bypassing row rules) is brought back in line.
-- Passwords arrive as session settings `claims.<role>_password`, set by migrate.apply from the environment.
DO $$
DECLARE
  r text;
BEGIN
  FOREACH r IN ARRAY ARRAY['claims_owner', 'claims_loader', 'claims_app'] LOOP
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
      EXECUTE format('CREATE ROLE %I', r);
    END IF;
    EXECUTE format(
      'ALTER ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD %L',
      r, current_setting('claims.' || r || '_password'));
  END LOOP;
END $$;
