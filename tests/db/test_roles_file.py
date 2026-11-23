"""The roles file: a failed password statement never shows the password, and an account that is a member of another
role stops the run. Each test runs the file inside a transaction that is rolled back, so nothing it does stays."""

import uuid

import psycopg
import pytest

from claims_agent.db import migrate

pytestmark = pytest.mark.db

SENTINEL = "pw-SENTINEL-5be1"


def _run_roles_file(superuser_url: str, *, read_only: bool = False):
    """The roles file, run as the superuser with every password set to the sentinel; always rolled back."""
    with psycopg.connect(superuser_url) as conn:
        conn.read_only = read_only
        try:
            for role in migrate.ROLE_PASSWORDS:
                conn.execute("SELECT set_config(%s, %s, true)", [f"claims.{role}_password", SENTINEL])
            conn.execute((migrate.SQL_DIR / migrate.ROLES_FILE).read_text())
        finally:
            conn.rollback()


def _everything_said(e: psycopg.Error) -> str:
    d = e.diag
    return "\n".join(str(x) for x in (e, d.message_primary, d.message_detail, d.message_hint, d.context,
                                      d.internal_query, d.statement_position))


def test_a_failed_password_statement_is_reported_without_the_password(test_db):
    # In a read-only transaction every ALTER ROLE fails, as a refused password statement would.
    with pytest.raises(psycopg.errors.RaiseException) as e:
        _run_roles_file(test_db.superuser_url, read_only=True)
    assert e.value.diag.message_primary == "the attributes and password of account claims_owner could not be set"
    assert SENTINEL not in _everything_said(e.value)


def test_an_account_that_is_a_member_of_another_role_stops_the_run(test_db, session_db):
    suffix = uuid.uuid4().hex[:12]
    try:
        with psycopg.connect(test_db.superuser_url, autocommit=True) as su:
            su.execute(f"CREATE ROLE claims_test_tmp_{suffix} NOLOGIN")
            su.execute(f"GRANT claims_test_tmp_{suffix} TO claims_app")
        with pytest.raises(psycopg.errors.RaiseException, match=f"claims_app is a member of claims_test_tmp_{suffix}"):
            _run_roles_file(test_db.superuser_url)
    finally:
        with psycopg.connect(test_db.superuser_url, autocommit=True) as su:
            su.execute(f"DROP ROLE IF EXISTS claims_test_tmp_{suffix}")
    assert session_db.known_good_problems() == []


def test_the_roles_file_runs_cleanly_on_the_known_good_accounts(test_db, session_db):
    _run_roles_file(test_db.superuser_url)
    assert session_db.known_good_problems() == []
