"""Shared by tests/conftest.py and tests/db/conftest.py (two files named conftest cannot import each other).

The test database is migrated once per test session, by one session at a time (an advisory lock), and the
result is checked against the state a migration should leave (`EXPECTED_*` below) before it is trusted.
Before each database test:
  1. every other client session on the test database is ended, so a connection a previous test left open can
     neither hold a lock the reset waits on nor carry session state into the next test;
  2. a fingerprint of the schema, the roles and the database's settings is compared with the one taken right
     after the session's migration; if a test changed any of them, the schema is migrated again;
  3. every table is emptied and every sequence restarted, in one statement, as the superuser.
After each database test the fingerprint is compared again: a change is undone there (migrated again), and a
change a migration cannot undo fails THAT test with `SchemaLeak`. The same compare-and-rebuild runs once more
when the session ends.
Tests never rebuild the schema themselves: a test that needs a freshly migrated database asks for the
`fresh_test_db` fixture.

Not covered by the fingerprint (no test touches them): ALTER SYSTEM, permission to set server parameters,
large objects, table inheritance, and objects created in the system catalog schema.
"""

import os
import time
from dataclasses import dataclass, field

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from claims_agent.db import migrate
from claims_agent.db.pool import AppDb

# A reset or a rebuild waits at most this long for a lock, then fails instead of hanging the run.
LOCK_TIMEOUT_MS = 10_000

# Held by the session's own connection for the whole session: two sessions would end each other's connections.
SESSION_LOCK = "SELECT pg_try_advisory_lock(hashtext('claims_agent tests: one session per test database'))"

# What a migration leaves, checked once per session right after its first migration and before the session's
# fingerprint is taken, so a leftover a migration does not undo can never become a session's starting point.
# Each value comes from the file named; change it here when that file changes.
EXPECTED_ACCOUNTS = ("claims_owner", "claims_loader", "claims_app")                  # db/001_roles.sql
EXPECTED_ACCOUNT_ATTRIBUTES = {                                                     # db/001_roles.sql ALTER ROLE
    "rolcanlogin": True, "rolsuper": False, "rolcreatedb": False, "rolcreaterole": False,
    "rolreplication": False, "rolbypassrls": False,
}
EXPECTED_MEMBERSHIPS: frozenset[tuple[str, str]] = frozenset()                       # db/001_roles.sql grants none
EXPECTED_ROLE_SETTINGS: frozenset[str] = frozenset()      # none of the accounts', none on the test database
EXPECTED_SCHEMAS = frozenset({"public"})                  # src/claims_agent/db/migrate.py recreates only public
EXPECTED_FORCED_ROW_SECURITY = frozenset({"claims", "claim_documents"})              # db/003_security.sql
EXPECTED_DEFAULT_PRIVILEGES = frozenset({                 # src/claims_agent/db/migrate.py ALTER DEFAULT PRIVILEGES
    "claims_owner * f {claims_owner=X/claims_owner}",
})

# The namespaces that hold the database's own objects: every schema but the system ones and the per-session
# temporary ones (a temporary schema outlives the session that made it, but its objects do not).
_USER_NAMESPACES = """
  SELECT oid, nspname FROM pg_namespace
  WHERE nspname NOT IN ('pg_catalog', 'information_schema')
    AND nspname NOT LIKE 'pg\\_toast%' AND nspname NOT LIKE 'pg\\_temp\\_%'
"""

# One line per catalog row that defines the schema, the roles or the database's settings; never an oid
# (a rebuild gives every object a new one) and never a data value. Role passwords are covered only as
# set / not set: their stored hashes are salted afresh by every migration, and every test logs in with the
# passwords from the environment, so a changed password fails loudly rather than silently.
FINGERPRINT = f"""
WITH ns AS ({_USER_NAMESPACES}),
lines AS (
  SELECT 'database ' || concat_ws(' ', datname, pg_get_userbyid(datdba), datacl::text, datallowconn,
                                  datconnlimit, datistemplate) AS line
  FROM pg_database WHERE datname = current_database()
  UNION ALL
  SELECT 'setting ' || concat_ws(' ', coalesce(r.rolname, '*'), CASE WHEN s.setdatabase = 0 THEN '*' ELSE 'db' END,
                                 s.setconfig::text)
  FROM pg_db_role_setting s LEFT JOIN pg_roles r ON r.oid = s.setrole
  WHERE s.setdatabase IN (0, (SELECT oid FROM pg_database WHERE datname = current_database()))
  UNION ALL
  SELECT 'role ' || concat_ws(' ', rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin,
                              rolreplication, rolbypassrls, rolconnlimit, rolvaliduntil, rolpassword IS NOT NULL)
  FROM pg_authid WHERE rolname NOT LIKE 'pg\\_%'
  UNION ALL
  SELECT 'member ' || concat_ws(' ', pg_get_userbyid(roleid), pg_get_userbyid(member), pg_get_userbyid(grantor),
                                admin_option, inherit_option, set_option)
  FROM pg_auth_members
  UNION ALL
  SELECT 'default_acl ' || concat_ws(' ', pg_get_userbyid(d.defaclrole), coalesce(n.nspname, '*'), d.defaclobjtype,
                                     d.defaclacl::text)
  FROM pg_default_acl d LEFT JOIN pg_namespace n ON n.oid = d.defaclnamespace
  UNION ALL
  SELECT 'schema ' || concat_ws(' ', n.nspname, pg_get_userbyid(n.nspowner), n.nspacl::text)
  FROM pg_namespace n JOIN ns ON ns.oid = n.oid
  UNION ALL
  SELECT 'extension ' || concat_ws(' ', e.extname, e.extversion, n.nspname, pg_get_userbyid(e.extowner))
  FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace
  UNION ALL
  SELECT 'event_trigger ' || concat_ws(' ', evtname, evtevent, evtenabled, evtfoid::regproc::text, evttags::text)
  FROM pg_event_trigger
  UNION ALL
  SELECT 'relation ' || concat_ws(' ', ns.nspname, c.relname, c.relkind, c.relpersistence,
                                  pg_get_userbyid(c.relowner), c.relacl::text, c.relrowsecurity,
                                  c.relforcerowsecurity, c.reloptions::text,
                                  CASE WHEN c.relkind IN ('v', 'm') THEN pg_get_viewdef(c.oid) END)
  FROM pg_class c JOIN ns ON ns.oid = c.relnamespace
  UNION ALL
  SELECT 'column ' || concat_ws(' ', ns.nspname, c.relname, a.attnum, a.attname, format_type(a.atttypid, a.atttypmod),
                                a.attnotnull, a.attidentity, a.attgenerated, a.attacl::text,
                                pg_get_expr(d.adbin, d.adrelid))
  FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid JOIN ns ON ns.oid = c.relnamespace
  LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
  WHERE a.attnum > 0 AND NOT a.attisdropped
  UNION ALL
  SELECT 'sequence ' || concat_ws(' ', ns.nspname, c.relname, format_type(s.seqtypid, NULL), s.seqstart,
                                  s.seqincrement, s.seqmin, s.seqmax, s.seqcache, s.seqcycle)
  FROM pg_sequence s JOIN pg_class c ON c.oid = s.seqrelid JOIN ns ON ns.oid = c.relnamespace
  UNION ALL
  SELECT 'constraint ' || concat_ws(' ', ns.nspname, co.conname, co.conrelid::regclass::text,
                                    co.contypid::regtype::text, pg_get_constraintdef(co.oid))
  FROM pg_constraint co JOIN ns ON ns.oid = co.connamespace
  UNION ALL
  SELECT 'index ' || pg_get_indexdef(i.indexrelid)
  FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid JOIN ns ON ns.oid = c.relnamespace
  UNION ALL
  SELECT 'trigger ' || concat_ws(' ', pg_get_triggerdef(t.oid), t.tgenabled)
  FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid JOIN ns ON ns.oid = c.relnamespace
  WHERE NOT t.tgisinternal
  UNION ALL
  SELECT 'rule ' || pg_get_ruledef(r.oid)
  FROM pg_rewrite r JOIN pg_class c ON c.oid = r.ev_class JOIN ns ON ns.oid = c.relnamespace
  WHERE r.rulename <> '_RETURN'
  UNION ALL
  SELECT 'policy ' || concat_ws(' ', p.schemaname, p.tablename, p.policyname, p.permissive, p.roles::text, p.cmd,
                                p.qual, p.with_check)
  FROM pg_policies p WHERE p.schemaname IN (SELECT nspname FROM ns)
  UNION ALL
  SELECT 'function ' || concat_ws(' ', ns.nspname, p.proname, p.prokind, pg_get_function_identity_arguments(p.oid),
                                  pg_get_userbyid(p.proowner), p.proacl::text, p.prosecdef, p.proconfig::text,
                                  CASE WHEN p.prokind IN ('f', 'p') THEN pg_get_functiondef(p.oid) ELSE p.prosrc END)
  FROM pg_proc p JOIN ns ON ns.oid = p.pronamespace
  UNION ALL
  SELECT 'type ' || concat_ws(' ', ns.nspname, t.typname, t.typtype, pg_get_userbyid(t.typowner), t.typacl::text,
                              (SELECT string_agg(e.enumlabel, ',' ORDER BY e.enumsortorder)
                               FROM pg_enum e WHERE e.enumtypid = t.oid))
  FROM pg_type t JOIN ns ON ns.oid = t.typnamespace
  UNION ALL
  SELECT 'operator ' || concat_ws(' ', ns.nspname, o.oprname, o.oprleft::regtype::text, o.oprright::regtype::text,
                                  o.oprcode::regproc::text)
  FROM pg_operator o JOIN ns ON ns.oid = o.oprnamespace
)
SELECT md5(string_agg(line, E'\\n' ORDER BY line)) FROM lines
"""

# Every table in the database's own schemas, emptied in one statement with their identities restarted; then
# every sequence (an identity's or a free-standing one) back to its start. TRUNCATE is not subject to row
# rules (forced or not), and the superuser may truncate and restart everything whoever owns it.
# A materialized view would keep its rows through a TRUNCATE of its tables, so one refuses the reset until this
# statement is taught what a fresh migration leaves in it.
RESET = f"""
DO $$
DECLARE
  tables text;
  seq text;
  views text;
BEGIN
  SELECT string_agg(format('%I.%I', ns.nspname, c.relname), ', ') INTO views
  FROM pg_class c JOIN ({_USER_NAMESPACES}) ns ON ns.oid = c.relnamespace
  WHERE c.relkind = 'm';
  IF views IS NOT NULL THEN
    RAISE EXCEPTION 'materialized view(s) % are not reset by tests/dbsupport.py RESET; teach it first', views;
  END IF;
  SELECT string_agg(format('%I.%I', ns.nspname, c.relname), ', ') INTO tables
  FROM pg_class c JOIN ({_USER_NAMESPACES}) ns ON ns.oid = c.relnamespace
  WHERE c.relkind IN ('r', 'p');
  IF tables IS NOT NULL THEN
    EXECUTE 'TRUNCATE ' || tables || ' RESTART IDENTITY';
  END IF;
  FOR seq IN SELECT format('%I.%I', ns.nspname, c.relname)
             FROM pg_class c JOIN ({_USER_NAMESPACES}) ns ON ns.oid = c.relnamespace WHERE c.relkind = 'S' LOOP
    EXECUTE 'ALTER SEQUENCE ' || seq || ' RESTART';
  END LOOP;
END $$
"""

# Row counts of every table in the database's own schemas, in one query.
ROW_COUNTS = f"""
SELECT ns.nspname || '.' || c.relname,
       (xpath('/row/n/text()', query_to_xml(format('SELECT count(*) AS n FROM %I.%I', ns.nspname, c.relname),
                                            false, true, '')))[1]::text::bigint
FROM pg_class c JOIN ({_USER_NAMESPACES}) ns ON ns.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p')
ORDER BY 1
"""

# The other client sessions on this database. Ending one waits up to a second for it to be gone; its result
# is not trusted (a session that was already closing is reported as not ended), so the sessions are counted
# again afterwards.
_OTHER_SESSIONS = """
FROM pg_stat_activity
WHERE datname = current_database() AND pid <> pg_backend_pid() AND backend_type = 'client backend'
"""
END_OTHER_SESSIONS = f"SELECT count(pg_terminate_backend(pid, 1000)) {_OTHER_SESSIONS}"
COUNT_OTHER_SESSIONS = f"SELECT count(*) {_OTHER_SESSIONS}"


def ensure_database(superuser_url: str) -> None:
    """Create the database the URL names when it is missing, connecting as the superuser to `claims`."""
    name = conninfo_to_dict(superuser_url)["dbname"]
    with psycopg.connect(make_conninfo(superuser_url, dbname="claims"), autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", [name]).fetchone() is None:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


@dataclass
class TestDb:
    """A migrated test database. `.app` is the server's own account (opened on first use)."""

    __test__ = False  # not a test class, whatever pytest thinks of the name

    superuser_url: str
    owner_url: str
    loader_url: str
    app_url: str
    _app: AppDb | None = field(default=None, repr=False)

    @property
    def app(self) -> AppDb:
        if self._app is None:
            self._app = AppDb(self.app_url)
        return self._app

    async def loader_count(self, table: str) -> int:
        async with await psycopg.AsyncConnection.connect(self.loader_url) as conn:
            row = await (await conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))).fetchone()
        return row[0]

    async def close(self) -> None:
        if self._app is not None:
            await self._app.close()


class SchemaLeak(RuntimeError):
    """The test database or the server's accounts are not in the state a migration leaves, and a migration does
    not put them back (a new role or membership, a role or database setting, a schema other than public)."""


class SessionBusy(RuntimeError):
    """Another test session is using the test database."""


_KNOWN_GOOD_QUERIES = {
    "account": """
        SELECT a.name, CASE WHEN r.rolname IS NOT NULL THEN to_jsonb(r) - 'rolname' END
        FROM unnest(%(accounts)s::text[]) a(name)
        LEFT JOIN (SELECT rolname, rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls
                   FROM pg_roles) r ON r.rolname = a.name""",
    "membership": """
        SELECT pg_get_userbyid(roleid) || ' -> ' || pg_get_userbyid(member) FROM pg_auth_members
        WHERE pg_get_userbyid(roleid) = ANY(%(accounts)s) OR pg_get_userbyid(member) = ANY(%(accounts)s)""",
    "setting": """
        SELECT concat_ws(' ', coalesce(pg_get_userbyid(nullif(s.setrole, 0)), '(every role)'),
                         coalesce((SELECT datname FROM pg_database WHERE oid = s.setdatabase), '(every database)'),
                         s.setconfig::text)
        FROM pg_db_role_setting s
        WHERE pg_get_userbyid(s.setrole) = ANY(%(accounts)s)
           OR s.setdatabase = (SELECT oid FROM pg_database WHERE datname = current_database())""",
    "schema": f"SELECT nspname FROM ({_USER_NAMESPACES}) ns",
    "forced row security": """
        SELECT c.relname FROM pg_class c
        WHERE c.relnamespace = 'public'::regnamespace AND c.relrowsecurity AND c.relforcerowsecurity""",
    "default privileges": """
        SELECT concat_ws(' ', pg_get_userbyid(d.defaclrole), coalesce(n.nspname, '*'), d.defaclobjtype,
                         d.defaclacl::text)
        FROM pg_default_acl d LEFT JOIN pg_namespace n ON n.oid = d.defaclnamespace""",
}


class TestDatabase:
    """The session's test database: migrated once, then reset (and migrated again when needed) around each test."""

    __test__ = False

    def __init__(self) -> None:
        self.superuser_url = os.environ["SUPERUSER_DATABASE_URL"]
        self.owner_url = os.environ["OWNER_DATABASE_URL"]
        self.loader_url = os.environ["LOADER_DATABASE_URL"]
        self.app_url = os.environ["DATABASE_URL"]
        self._guarded_url = make_conninfo(self.superuser_url, options=f"-c lock_timeout={LOCK_TIMEOUT_MS}")
        self.rebuilds = 0   # how many times a change made by a test forced the schema to be migrated again
        self.last_test: str | None = None   # the test that ran last, named when a leak is found after it
        self.rebuilt_after: str | None = None   # the last test whose own change was undone after it
        ensure_database(self.superuser_url)
        self._conn = psycopg.connect(self._guarded_url, autocommit=True)
        try:
            if not self._conn.execute(SESSION_LOCK).fetchone()[0]:
                raise SessionBusy("another test session is using the test database; wait for it to finish "
                                  "(two sessions would end each other's connections)")
            self.end_other_sessions()
            migrate.apply(self._guarded_url)
            problems = self.known_good_problems()
            if problems:
                raise SchemaLeak("the test database or its accounts are not in the state a migration leaves, and "
                                 "a migration does not put this back; fix it by hand before running tests "
                                 "(expected values: tests/dbsupport.py EXPECTED_*):\n  " + "\n  ".join(problems))
            self.baseline = self.fingerprint()
            # A reset leaves every table empty, which is what a migration leaves only while no migration seeds
            # rows. If one ever does, every reset migrates instead (the slower, always-right path).
            self.seeds_rows = any(self.row_counts().values())
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def test_db(self) -> TestDb:
        return TestDb(superuser_url=self.superuser_url, owner_url=self.owner_url, loader_url=self.loader_url,
                      app_url=self.app_url)

    def fingerprint(self) -> str:
        return self._conn.execute(FINGERPRINT).fetchone()[0]

    def row_counts(self) -> dict[str, int]:
        return dict(self._conn.execute(ROW_COUNTS).fetchall())

    def known_good_problems(self) -> list[str]:
        """Every way the server's accounts and the test database differ from what a migration leaves."""
        def rows(kind):
            query = _KNOWN_GOOD_QUERIES[kind]
            params = {"accounts": list(EXPECTED_ACCOUNTS)} if "%(accounts)s" in query else None
            return self._conn.execute(query, params).fetchall()

        problems = []
        for name, attributes in rows("account"):
            if attributes is None:
                problems.append(f"account {name} does not exist")
            for key, want in EXPECTED_ACCOUNT_ATTRIBUTES.items():
                if attributes is not None and attributes[key] != want:
                    problems.append(f"account {name}: {key} is {attributes[key]}, expected {want}")
        expected_sets = {
            "membership": {" -> ".join(m) for m in EXPECTED_MEMBERSHIPS},
            "setting": set(EXPECTED_ROLE_SETTINGS),
            "schema": set(EXPECTED_SCHEMAS),
            "forced row security": set(EXPECTED_FORCED_ROW_SECURITY),
            "default privileges": set(EXPECTED_DEFAULT_PRIVILEGES),
        }
        for kind, want in expected_sets.items():
            have = {r[0] for r in rows(kind)}
            problems += [f"{kind}: unexpected {x}" for x in sorted(have - want)]
            problems += [f"{kind}: missing {x}" for x in sorted(want - have)]
        return problems

    def end_other_sessions(self) -> None:
        deadline = time.monotonic() + 5
        while True:
            self._conn.execute(END_OTHER_SESSIONS)
            left = self._conn.execute(COUNT_OTHER_SESSIONS).fetchone()[0]
            if not left:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"{left} session(s) on the test database did not end within 5 seconds")
            time.sleep(0.01)

    def reset_data(self) -> None:
        """Every table empty, every sequence at its start (one statement)."""
        self._conn.execute(RESET)

    def start_test(self) -> TestDb:
        """Before a test: no other session left, the session's schema, every table empty."""
        self.end_other_sessions()
        return self.reset()

    def reset(self) -> TestDb:
        """The session's schema (migrated again if a test changed it), every table empty, every sequence at
        its start. Ends no session, so a test that loads a second data set keeps the connections it holds."""
        if self.seeds_rows or self.fingerprint() != self.baseline:
            return self.rebuild(end_sessions=False)
        self.reset_data()
        return self.test_db()

    def after_test(self, test: str) -> None:
        """After a test: a change it made to the schema or the roles is undone now; one a migration cannot undo
        fails this test, by name."""
        self.last_test = test
        if self.fingerprint() != self.baseline:
            self.rebuild()
            self.rebuilt_after = test

    def finish(self) -> None:
        """At the end of the session: whatever the last test did, leave the session's own schema and roles."""
        self.end_other_sessions()
        if self.fingerprint() != self.baseline:
            self.rebuild(end_sessions=False)

    def rebuild(self, *, end_sessions: bool = True) -> TestDb:
        """Migrate again, then check the result is the session's own schema and roles."""
        if end_sessions:
            self.end_other_sessions()
        migrate.apply(self._guarded_url)
        if not self.seeds_rows:
            self.reset_data()   # a table outside the migrated schema would otherwise keep its rows
        self.rebuilds += 1
        if self.fingerprint() != self.baseline:
            after = f" (left by {self.last_test})" if self.last_test else ""
            problems = self.known_good_problems()
            raise SchemaLeak(f"a test changed something a migration does not put back{after}; database tests cannot "
                             "trust this database until it is put right by hand" +
                             ("".join(f"\n  {p}" for p in problems) if problems else
                              " (a role, a setting or a schema outside what tests/dbsupport.py EXPECTED_* checks)"))
        return self.test_db()
