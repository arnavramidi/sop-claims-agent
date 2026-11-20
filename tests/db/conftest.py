import uuid
from dataclasses import dataclass

import psycopg
import pytest

from claims_agent.db.pool import AppDb
from dbsupport import TestDb


@pytest.fixture
async def test_db(clean_db):
    """The test database with the session's schema and every table empty (see `clean_db`)."""
    yield clean_db
    await clean_db.close()


@pytest.fixture
async def fresh_test_db(session_db, clean_db):
    """For a test that wants the schema freshly migrated, not reset: migrates again before the test."""
    db = session_db.rebuild()
    yield db
    await db.close()


@pytest.fixture
async def owner_conn(test_db):
    async with await psycopg.AsyncConnection.connect(test_db.owner_url) as conn:
        yield conn


@pytest.fixture
async def loader_conn(test_db):
    async with await psycopg.AsyncConnection.connect(test_db.loader_url) as conn:
        yield conn


@pytest.fixture
async def app_db(test_db):
    db = AppDb(test_db.app_url)
    yield db
    await db.close()


@dataclass(frozen=True)
class Seed:
    unverified: uuid.UUID
    verified_p12: uuid.UUID


@pytest.fixture
async def seed(loader_conn, owner_conn) -> Seed:
    s = Seed(unverified=uuid.uuid4(), verified_p12=uuid.uuid4())
    await loader_conn.execute(
        "INSERT INTO customers (party_id, name, policy_number, dob, id_type, id_last4, phone, email) VALUES "
        "('P9', 'Margaret Chen', 'POL-9921', '1985-03-15', 'ssn_last4', '4472', '+16505212836', 'margaret@email.com'), "
        "('P12', 'Ma Tian', 'POL-8836', '1964-09-10', 'national_id_last4', '6688', '+16502088799', 'matian@example.com')"
    )
    await loader_conn.execute(
        "INSERT INTO claims (case_id, party_id, case_type, created_at, status, summary) VALUES "
        "('CL-2048', 'P9', 'healthcare', '2026-01-12', 'denied', 'Healthcare claim denied'), "
        "('CL-2011', 'P9', 'healthcare', '2025-01-28', 'closed', 'Healthcare claim settled'), "
        "('CL-3001', 'P12', 'healthcare', '2026-03-01', 'denied', 'Healthcare claim denied')"
    )
    await loader_conn.execute(
        "INSERT INTO claim_documents (case_id, party_id, position, name) VALUES "
        "('CL-2048', 'P9', 1, 'pathology report'), ('CL-2048', 'P9', 2, 'office note'), "
        "('CL-3001', 'P12', 1, 'diagnosis report')"
    )
    await loader_conn.execute(
        "INSERT INTO conversations (id, state, today) VALUES (%s, 'VERIFY_ID', '2026-03-05'), (%s, 'VERIFY_ID', '2026-03-05')",
        [s.unverified, s.verified_p12],
    )
    await loader_conn.commit()
    await owner_conn.execute("UPDATE conversations SET verified_party_id = 'P12' WHERE id = %s", [s.verified_p12])
    await owner_conn.commit()
    return s
