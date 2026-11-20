"""The test database's own support: whatever one test does to the database never reaches the next.

Each test makes its change and then runs the support code's own step (the start-of-test reset, the after-test
check, the end-of-session check) inside the same test, so it proves its point when run alone. One pair at the
top proves the fixtures are wired to that step.

No test here gives an account more power than a migration gives it for longer than the test itself: a change to
a role or to the database's settings is undone in the test's own `finally`, as well as by the support code.
"""

import psycopg
import pytest
from dbsupport import SchemaLeak, SessionBusy, TestDatabase

pytestmark = pytest.mark.db

PROBE = "00000000-0000-0000-0000-0000000000aa"
_left: dict = {}   # what the first test of the wiring pair left behind


async def _one(url: str, query: str, params=None):
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as c:
        return await (await c.execute(query, params)).fetchone()


async def _do(url: str, statement: str) -> None:
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as c:
        await c.execute(statement)


def _rebuilt_by_the_next_reset(session_db) -> None:
    """The change is seen, the start-of-test step migrates again once, and the session's schema is back."""
    assert session_db.fingerprint() != session_db.baseline
    before = session_db.rebuilds
    session_db.start_test()
    assert session_db.rebuilds == before + 1
    assert session_db.fingerprint() == session_db.baseline


# ---- the fixtures are wired to the reset (a pair, run in file order) ----------------------------------------

async def test_wiring_a_row_left_by_one_test(test_db):
    await _do(test_db.loader_url, f"INSERT INTO conversations (id, state) VALUES ('{PROBE}', 'VERIFY_ID')")
    _left["row"] = True


async def test_wiring_the_next_test_starts_with_every_table_empty(test_db, session_db):
    assert set(session_db.row_counts().values()) == {0}
    assert await _one(test_db.loader_url, "SELECT count(*) FROM conversations WHERE id = %s", [PROBE]) == (0,)


async def test_wiring_a_schema_change_left_by_one_test(test_db, request):
    await _do(test_db.owner_url, "CREATE TABLE public.probe_leftover (a int)")
    _left["schema"] = request.node.nodeid


async def test_wiring_is_undone_after_that_test_and_not_before_the_next(test_db, session_db):
    if "schema" in _left:
        assert session_db.rebuilt_after == _left.pop("schema")
    assert session_db.fingerprint() == session_db.baseline


# ---- data ----------------------------------------------------------------------------------------------------

async def test_a_reset_empties_every_table_and_restarts_identities(test_db, session_db):
    async with await psycopg.AsyncConnection.connect(test_db.loader_url, autocommit=True) as c:
        await c.execute("INSERT INTO conversations (id, state) VALUES (%s, 'VERIFY_ID')", [PROBE])
        for turn_no in range(3):
            await c.execute("INSERT INTO turns (conversation_id, turn_no) VALUES (%s, %s)", [PROBE, turn_no])
        assert [i for (i,) in await (await c.execute("SELECT id FROM turns ORDER BY id")).fetchall()] == [1, 2, 3]
    session_db.start_test()
    assert set(session_db.row_counts().values()) == {0}
    async with await psycopg.AsyncConnection.connect(test_db.loader_url, autocommit=True) as c:
        await c.execute("INSERT INTO conversations (id, state) VALUES (%s, 'VERIFY_ID')", [PROBE])
        row = await (await c.execute(
            "INSERT INTO turns (conversation_id, turn_no) VALUES (%s, 0) RETURNING id", [PROBE])).fetchone()
    assert row == (1,)


async def test_a_loaded_data_set_is_gone_after_a_reset(load_dataset, session_db):
    db = await load_dataset("starter")
    assert await db.loader_count("claims") == 5
    session_db.start_test()
    assert set(session_db.row_counts().values()) == {0}


async def test_a_second_data_set_in_the_same_test_replaces_the_first(load_dataset):
    """As when each load migrated: the second load starts from empty tables with identities at their start."""
    first = await load_dataset("bad_data")
    assert await first.loader_count("load_report") > 0
    second = await load_dataset("starter")
    assert await second.loader_count("load_report") == 0
    async with await psycopg.AsyncConnection.connect(second.loader_url, autocommit=True) as c:
        await c.execute("INSERT INTO load_report (level, reason) VALUES ('row', 'probe')")
        assert (await (await c.execute("SELECT max(id) FROM load_report")).fetchone()) == (1,)


async def test_a_materialized_view_refuses_the_reset_until_it_is_taught(test_db, session_db):
    """Its rows would survive a TRUNCATE of its tables, so a reset refuses rather than leave them."""
    await _do(test_db.owner_url, "CREATE MATERIALIZED VIEW public.probe_view AS SELECT 1 AS a")
    with pytest.raises(psycopg.errors.RaiseException, match="probe_view"):
        session_db.reset_data()
    _rebuilt_by_the_next_reset(session_db)
    assert await _one(test_db.superuser_url, "SELECT to_regclass('public.probe_view')") == (None,)


# ---- schema, grants and rules: detected and migrated again ---------------------------------------------------

async def test_a_created_table_is_detected_and_removed(test_db, session_db):
    await _do(test_db.owner_url, "CREATE TABLE public.probe_leftover (a int)")
    _rebuilt_by_the_next_reset(session_db)
    assert await _one(test_db.superuser_url, "SELECT to_regclass('public.probe_leftover')") == (None,)


async def test_a_changed_grant_is_detected_and_restored(test_db, session_db):
    await _do(test_db.owner_url, "REVOKE SELECT ON vocabulary FROM claims_app")   # less power, never more
    _rebuilt_by_the_next_reset(session_db)
    assert await _one(test_db.superuser_url, "SELECT has_table_privilege('claims_app', 'vocabulary', 'SELECT')") \
        == (True,)


async def test_a_changed_function_body_is_detected_and_restored(test_db, session_db):
    await _do(test_db.owner_url,
              "CREATE OR REPLACE FUNCTION public.name_matches(candidates text[], stored text) RETURNS boolean "
              "LANGUAGE sql SET search_path = pg_catalog, public, pg_temp AS 'SELECT false'")   # stricter
    _rebuilt_by_the_next_reset(session_db)
    (body,) = await _one(test_db.superuser_url, "SELECT prosrc FROM pg_proc WHERE proname = 'name_matches'")
    assert body.strip() != "SELECT false"


async def test_a_dropped_policy_is_detected_and_restored(test_db, session_db):
    await _do(test_db.owner_url, "DROP POLICY app_read ON claims")   # the app then sees fewer rows, not more
    _rebuilt_by_the_next_reset(session_db)
    assert await _one(test_db.superuser_url,
                      "SELECT count(*) FROM pg_policies WHERE tablename = 'claims' AND policyname = 'app_read'") \
        == (1,)


async def test_forced_row_security_turned_off_is_detected_and_restored(test_db, session_db):
    try:
        await _do(test_db.owner_url, "ALTER TABLE claims NO FORCE ROW LEVEL SECURITY")
        _rebuilt_by_the_next_reset(session_db)
        assert await _one(test_db.superuser_url,
                          "SELECT relforcerowsecurity FROM pg_class WHERE oid = 'public.claims'::regclass") == (True,)
    finally:
        await _do(test_db.superuser_url, "ALTER TABLE public.claims FORCE ROW LEVEL SECURITY")


async def test_a_changed_role_attribute_is_detected_and_restored(test_db, session_db):
    """CREATEDB opens no data; the migration states NOCREATEDB, so a rebuild must put it back."""
    try:
        await _do(test_db.superuser_url, "ALTER ROLE claims_app CREATEDB")
        _rebuilt_by_the_next_reset(session_db)
        assert await _one(test_db.superuser_url, "SELECT rolcreatedb FROM pg_roles WHERE rolname = 'claims_app'") \
            == (False,)
    finally:
        await _do(test_db.superuser_url, "ALTER ROLE claims_app NOCREATEDB")


# ---- changes a migration cannot undo: loud, and blamed on the test that made them -----------------------------

async def test_a_database_setting_fails_loudly(test_db, session_db):
    try:
        await _do(test_db.superuser_url, "ALTER DATABASE claims_test SET work_mem = '8MB'")
        with pytest.raises(SchemaLeak, match="work_mem"):
            session_db.start_test()
    finally:
        await _do(test_db.superuser_url, "ALTER DATABASE claims_test RESET work_mem")
    assert session_db.fingerprint() == session_db.baseline


async def test_a_table_in_another_schema_fails_loudly_and_loses_its_rows(test_db, session_db):
    try:
        await _do(test_db.superuser_url, "CREATE SCHEMA probe_schema; CREATE TABLE probe_schema.t (a int); "
                                         "INSERT INTO probe_schema.t VALUES (1)")
        with pytest.raises(SchemaLeak, match="probe_schema"):
            session_db.start_test()
        assert await _one(test_db.superuser_url, "SELECT count(*) FROM probe_schema.t") == (0,)
    finally:
        await _do(test_db.superuser_url, "DROP SCHEMA IF EXISTS probe_schema CASCADE")
    assert session_db.fingerprint() == session_db.baseline


async def test_the_check_after_a_test_names_that_test(test_db, session_db):
    try:
        await _do(test_db.superuser_url, "ALTER DATABASE claims_test SET work_mem = '8MB'")
        with pytest.raises(SchemaLeak, match=r"left by tests/probe\.py::test_leaky"):
            session_db.after_test("tests/probe.py::test_leaky")
    finally:
        await _do(test_db.superuser_url, "ALTER DATABASE claims_test RESET work_mem")
    assert session_db.fingerprint() == session_db.baseline


async def test_the_check_after_a_test_undoes_what_a_migration_can(test_db, session_db):
    await _do(test_db.owner_url, "CREATE TABLE public.probe_leftover (a int)")
    before = session_db.rebuilds
    session_db.after_test("tests/probe.py::test_tidy")
    assert session_db.rebuilds == before + 1
    assert session_db.fingerprint() == session_db.baseline


async def test_the_end_of_the_session_undoes_the_last_tests_change(test_db, session_db):
    await _do(test_db.owner_url, "CREATE TABLE public.probe_leftover (a int)")
    session_db.finish()
    assert session_db.fingerprint() == session_db.baseline
    assert await _one(test_db.superuser_url, "SELECT to_regclass('public.probe_leftover')") == (None,)


# ---- the state a session starts from ----------------------------------------------------------------------

async def test_a_migrated_database_is_in_the_known_good_state(test_db, session_db):
    assert session_db.known_good_problems() == []


async def test_a_powerful_account_is_not_known_good(test_db, session_db):
    try:
        await _do(test_db.superuser_url, "ALTER ROLE claims_app CREATEDB")
        assert session_db.known_good_problems() == ["account claims_app: rolcreatedb is True, expected False"]
    finally:
        await _do(test_db.superuser_url, "ALTER ROLE claims_app NOCREATEDB")
    assert session_db.known_good_problems() == []


async def test_a_role_setting_or_membership_is_not_known_good(test_db, session_db):
    try:
        await _do(test_db.superuser_url,
                  "ALTER ROLE claims_loader SET work_mem = '8MB'; GRANT claims_loader TO claims_app")
        problems = session_db.known_good_problems()
        assert any(p.startswith("setting: unexpected claims_loader") for p in problems), problems
        assert "membership: unexpected claims_loader -> claims_app" in problems
    finally:
        await _do(test_db.superuser_url,
                  "ALTER ROLE claims_loader RESET work_mem; REVOKE claims_loader FROM claims_app")
    assert session_db.known_good_problems() == []


# ---- fixtures and sessions ----------------------------------------------------------------------------------

@pytest.fixture
def before_fresh(session_db):
    return session_db.rebuilds, session_db._conn.execute("SELECT 'public'::regnamespace::oid").fetchone()[0]


async def test_the_fresh_schema_fixture_migrates_again(before_fresh, fresh_test_db, session_db):
    rebuilds, public = before_fresh
    assert session_db.rebuilds > rebuilds
    assert await _one(fresh_test_db.superuser_url, "SELECT 'public'::regnamespace::oid") != (public,)
    assert session_db.fingerprint() == session_db.baseline
    assert set(session_db.row_counts().values()) == {0}


async def test_a_second_session_on_the_same_database_is_refused(test_db, session_db):
    with pytest.raises(SessionBusy):
        TestDatabase()
    assert session_db.fingerprint() == session_db.baseline   # its own connection was not ended


# ---- connections --------------------------------------------------------------------------------------------

async def test_a_connection_left_open_holding_a_lock_neither_blocks_the_reset_nor_survives_it(test_db, session_db):
    conn = psycopg.connect(test_db.owner_url)   # a transaction left open, as a careless test might
    try:
        conn.execute("LOCK TABLE claims IN ACCESS EXCLUSIVE MODE")
        conn.execute("SELECT set_config('app.conversation_id', %s, false)", [PROBE])
        pid = conn.info.backend_pid
        session_db.start_test()   # with the lock held this would wait, then fail after the lock timeout
        with pytest.raises(psycopg.OperationalError):
            conn.execute("SELECT 1")
    finally:
        conn.close()
    assert await _one(test_db.superuser_url, "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", [pid]) == (0,)
    assert await _one(test_db.loader_url, "SELECT count(*) FROM claims") == (0,)
