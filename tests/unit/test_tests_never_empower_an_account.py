"""No file under tests/ may hold SQL that gives one of the three database accounts more power than a migration does.

Roles belong to the whole database server, and no migration takes back a role membership, so a test that grants
a role to an account, or raises one of its attributes, and is stopped before it undoes that leaves the account
with that power. Tests prove the support code's checks with a throwaway role instead (named with
the throwaway prefix, no login, no privileges): granting one of those to an account gives the account nothing.

The files are read as plain text, so a statement is caught wherever it sits: in a string, an f-string, a string
split over several lines, or a .sql file. This file is skipped, because its own samples must be caught.
"""

import re
from pathlib import Path

import pytest
from dbsupport import EXPECTED_ACCOUNTS as ACCOUNTS
from dbsupport import THROWAWAY_ROLE_PREFIX as THROWAWAY_PREFIX

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
MIGRATIONS = sorted((ROOT / "db").glob("*.sql"))
POWER_ATTRIBUTES = ("SUPERUSER", "BYPASSRLS", "CREATEROLE", "CREATEDB", "REPLICATION")

_ACCOUNT = r'"?(?P<account>' + "|".join(ACCOUNTS) + r')"?\b'
_NAME = r'[\w"{}$%.]+'                              # a role name, a quoted one, or a placeholder for one
_NAMES_THEN_ACCOUNT = rf"(?:{_NAME}\s*,\s*)*{_ACCOUNT}"
# One statement's clauses: never past its end (`;`) or past the end of the call that holds it (`)`); no role
# statement's options contain either.
_CLAUSES = r"[^;)]{0,300}?"

# GRANT <roles> TO ... <account>: a role grant names no object, so it has no ON (every privilege grant has one).
# The words may sit in two strings Python joins ("GRANT claims_loader " "TO claims_app"), so quotes count as space.
ROLE_GRANT = re.compile(
    rf"""\bGRANT\s+(?P<roles>(?:(?!\bON\b)[^;)]){{1,300}}?)[\s"']+TO[\s"']+{_NAMES_THEN_ACCOUNT}""", re.IGNORECASE)
# CREATE ROLE <role> ... ROLE|USER|ADMIN <account>, and ALTER GROUP <role> ADD USER <account>: the account
# becomes a member of the new role, as a GRANT would make it.
ROLE_GRANT_ON_CREATE = re.compile(
    rf"\bCREATE\s+(?:ROLE|USER|GROUP)\s+(?P<roles>{_NAME}){_CLAUSES}\b(?:ROLE|USER|ADMIN)\s+{_NAMES_THEN_ACCOUNT}",
    re.IGNORECASE)
ROLE_GRANT_BY_GROUP = re.compile(rf"\bALTER\s+GROUP\s+(?P<roles>{_NAME})\s+ADD\s+USER\s+{_NAMES_THEN_ACCOUNT}",
                                 re.IGNORECASE)
# ALTER|CREATE ROLE|USER <account> ... SUPERUSER (the word itself: NOSUPERUSER takes the power away).
POWER_ATTRIBUTE = re.compile(
    rf"\b(?:ALTER|CREATE)\s+(?:ROLE|USER)\s+{_ACCOUNT}{_CLAUSES}\b(?P<attribute>{'|'.join(POWER_ATTRIBUTES)})\b",
    re.IGNORECASE)


def _only_throwaway_roles(roles: str) -> bool:
    names = [n.strip().strip("\"'").strip() for n in re.split(r",|\s+", roles) if n.strip().strip("\"'").strip()]
    return bool(names) and all(n.lower().startswith(THROWAWAY_PREFIX) for n in names)


def account_escalations(text: str) -> list[str]:
    """Every statement in `text` that grants a role (other than a throwaway one) to one of the three accounts, or
    gives one of them a power attribute; each as the text that matched."""
    found = []
    for pattern in (ROLE_GRANT, ROLE_GRANT_ON_CREATE, ROLE_GRANT_BY_GROUP):
        found += [m.group(0) for m in pattern.finditer(text) if not _only_throwaway_roles(m.group("roles"))]
    found += [m.group(0) for m in POWER_ATTRIBUTE.finditer(text)]
    return found


CAUGHT = [
    "GRANT claims_loader TO claims_app",
    "ALTER ROLE claims_loader SET work_mem = '8MB'; GRANT claims_loader TO claims_app",
    'await _do(url, "GRANT claims_loader "\n              "TO claims_app")',
    "grant pg_read_all_data to claims_test_tmp_a, claims_app with inherit true",
    'GRANT {} TO claims_app',
    "GRANT claims_test_tmp_a, claims_loader TO claims_app",
    "ALTER ROLE claims_app CREATEDB",
    "ALTER USER claims_owner WITH SUPERUSER",
    'ALTER ROLE "claims_loader" LOGIN BYPASSRLS',
    "CREATE ROLE claims_app CREATEROLE",
    "CREATE ROLE probe ROLE claims_app",
    "ALTER GROUP claims_loader ADD USER claims_app",
]

NOT_CAUGHT = [
    "ALTER ROLE claims_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD 'x'",
    "ALTER ROLE claims_app NOLOGIN",
    "ALTER ROLE claims_loader SET work_mem = '8MB'",
    "GRANT SELECT, INSERT ON turns, escalations, feedback TO claims_app",
    "GRANT INSERT (id, state) ON conversations TO claims_app",
    "GRANT TEMPORARY ON DATABASE {} TO claims_app",
    "ALTER DEFAULT PRIVILEGES FOR ROLE claims_owner GRANT SELECT ON TABLES TO claims_app",
    "GRANT claims_test_tmp_0a1b TO claims_app",
    "GRANT claims_test_tmp_{suffix} TO claims_app",
    "GRANT claims_app TO claims_test_tmp_0a1b",
    "CREATE ROLE claims_test_tmp_0a1b NOLOGIN",
    "ALTER ROLE claims_test_tmp_0a1b CREATEDB",
    "REVOKE claims_loader FROM claims_app",
]


@pytest.mark.parametrize("text", CAUGHT)
def test_the_guard_catches(text):
    assert account_escalations(text), text


@pytest.mark.parametrize("text", NOT_CAUGHT)
def test_the_guard_lets_pass(text):
    assert account_escalations(text) == []


@pytest.mark.parametrize("path", MIGRATIONS, ids=lambda p: p.name)
def test_the_migrations_do_not_trip_the_guard(path):
    assert account_escalations(path.read_text(encoding="utf-8")) == []


def test_no_file_under_tests_gives_an_account_more_power():
    here = Path(__file__).resolve()
    found = []
    for path in sorted(p for p in TESTS.rglob("*") if p.is_file() and p.resolve() != here):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue   # not text, so no SQL
        found += [f"{path.relative_to(ROOT)}: {s!r}" for s in account_escalations(text)]
    assert found == [], ("a test must never give one of the three accounts more power than a migration gives it, "
                         "not even briefly; use a throwaway role named " + THROWAWAY_PREFIX + "...:\n  "
                         + "\n  ".join(found))
