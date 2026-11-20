-- The three accounts. Runs as the superuser, before the schema is rebuilt.
-- Roles belong to the whole server, so they survive the schema drop: create each one only if missing.
-- Every run re-states its attributes and password, so a stale password or a role created by hand with
-- more rights (superuser, bypassing row rules) is brought back in line.
-- Passwords arrive as session settings `claims.<role>_password`, set by migrate.apply from the environment.
--
-- On a managed database the administrative account is not a superuser, only allowed to create roles. Postgres
-- refuses to let such an account name the attributes only a superuser may grant (even to switch them off), so it
-- checks instead that no account holds one, and stops if one does. It also takes membership of claims_owner, which
-- a superuser does not need, so it can build the schema as that account.
DO $$
DECLARE
  r text;
  as_superuser boolean := (SELECT rolsuper FROM pg_roles WHERE rolname = current_user);
BEGIN
  FOREACH r IN ARRAY ARRAY['claims_owner', 'claims_loader', 'claims_app'] LOOP
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
      EXECUTE format('CREATE ROLE %I', r);
    END IF;
    IF as_superuser THEN
      EXECUTE format(
        'ALTER ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD %L',
        r, current_setting('claims.' || r || '_password'));
    ELSE
      EXECUTE format('ALTER ROLE %I LOGIN NOCREATEROLE PASSWORD %L', r, current_setting('claims.' || r || '_password'));
      IF EXISTS (SELECT FROM pg_roles WHERE rolname = r
                 AND (rolsuper OR rolcreatedb OR rolreplication OR rolbypassrls)) THEN
        RAISE EXCEPTION 'account % holds a power only a superuser can take away; remove it, then run setup again', r;
      END IF;
    END IF;
  END LOOP;
  IF NOT as_superuser THEN
    EXECUTE format('GRANT claims_owner TO %I WITH SET TRUE, INHERIT TRUE', current_user);
  END IF;
END $$;
