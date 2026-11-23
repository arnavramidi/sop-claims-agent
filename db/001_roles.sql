-- The accounts. Runs as the superuser, before the schema is rebuilt.
-- Roles belong to the whole server, so they survive the schema drop: create each one only if missing.
-- Every run re-states its attributes and password, so a stale password or a role created by hand with
-- more rights (superuser, bypassing row rules) is brought back in line.
-- Passwords arrive as session settings `claims.<role>_password`, set by migrate.apply from the environment. The demo
-- account's may be empty: it then has no password and cannot log in (the claim-change control is off).
-- A password statement that fails is reported with a fixed message: the statement itself holds the password, and an
-- error's context would carry it into setup's output and a deploy log.
-- No account may be a member of another role: a membership would hand it that role's rights and row rules (the
-- server's account inside the loader's role would read every row). Any membership stops the run.
--
-- On a managed database the administrative account is not a superuser, only allowed to create roles. Postgres
-- refuses to let such an account name the attributes only a superuser may grant (even to switch them off), so it
-- checks instead that no account holds one, and stops if one does. It also takes membership of claims_owner, which
-- a superuser does not need, so it can build the schema as that account; migrate.apply takes that membership back
-- before it commits.
DO $$
DECLARE
  r text;
  pw text;
  as_superuser boolean := (SELECT rolsuper FROM pg_roles WHERE rolname = current_user);
BEGIN
  FOREACH r IN ARRAY ARRAY['claims_owner', 'claims_loader', 'claims_app', 'claims_demo'] LOOP
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
      EXECUTE format('CREATE ROLE %I', r);
    END IF;
    pw := NULLIF(current_setting('claims.' || r || '_password'), '');
    BEGIN
      IF as_superuser THEN
        EXECUTE format(
          'ALTER ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD %L', r, pw);
      ELSE
        EXECUTE format('ALTER ROLE %I LOGIN NOCREATEROLE PASSWORD %L', r, pw);
      END IF;
    EXCEPTION WHEN OTHERS THEN
      RAISE EXCEPTION 'the attributes and password of account % could not be set', r;
    END;
    IF NOT as_superuser AND EXISTS (SELECT FROM pg_roles WHERE rolname = r
                                    AND (rolsuper OR rolcreatedb OR rolreplication OR rolbypassrls)) THEN
      RAISE EXCEPTION 'account % holds a power only a superuser can take away; remove it, then run setup again', r;
    END IF;
  END LOOP;
  SELECT format('%s is a member of %s', pg_get_userbyid(m.member), pg_get_userbyid(m.roleid)) INTO r
  FROM pg_auth_members m
  WHERE pg_get_userbyid(m.member) IN ('claims_owner', 'claims_loader', 'claims_app', 'claims_demo')
  ORDER BY 1 LIMIT 1;
  IF r IS NOT NULL THEN
    RAISE EXCEPTION 'account % (an account must hold no other role''s rights); revoke that membership, then run setup again', r;
  END IF;
  IF NOT as_superuser THEN
    EXECUTE format('GRANT claims_owner TO %I WITH SET TRUE, INHERIT TRUE', current_user);
  END IF;
END $$;
