"""A few slow turns cannot stall the app: every connection carries a lock wait limit and an idle-in-transaction limit,
the pool's size and wait come from the settings, a message for a conversation whose turn is still running is answered
"busy" instead of hanging, and the health route reports the database."""

import asyncio
import time

import psycopg
import pytest
from fastapi.testclient import TestClient
from helpers import make_form
from test_api import Scripted

from claims_agent import api
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps
from claims_agent.settings import Database, load_settings, load_thresholds

pytestmark = pytest.mark.db


@pytest.fixture
async def api_db(load_dataset):
    return await load_dataset("starter")


def _client(api_db, limits: Database):
    scripted = Scripted()
    deps = Deps(settings=load_settings(), thresholds=load_thresholds(), db=AppDb(api_db.app_url, limits=limits),
                guidance=None, filler=scripted.filler, extractor=scripted.extractor, wording=scripted.wording)
    return TestClient(api.create_app(deps)), deps, scripted


def test_the_settings_name_the_limits():
    assert load_settings().database == Database(pool_size=20, pool_wait_seconds=10, lock_wait_seconds=10,
                                                idle_in_transaction_seconds=150)


async def test_every_connection_carries_the_lock_and_idle_limits(test_db):
    db = AppDb(test_db.app_url, limits=Database(pool_size=3, lock_wait_seconds=2.5, idle_in_transaction_seconds=40))
    try:
        async with db.conversation(None) as c:
            row = await (await c.execute("SELECT current_setting('lock_timeout'), "
                                         "current_setting('idle_in_transaction_session_timeout')")).fetchone()
        assert row == ("2500ms", "40s")
        assert db._pool.max_size == 3
    finally:
        await db.close()


async def test_a_transaction_left_paused_is_ended_by_the_database(test_db):
    db = AppDb(test_db.app_url, limits=Database(idle_in_transaction_seconds=0.5))
    try:
        with pytest.raises(psycopg.errors.IdleInTransactionSessionTimeout):
            async with db.conversation(None) as c:
                await c.execute("SELECT 1")
                await asyncio.sleep(2)
                await c.execute("SELECT 1")
    finally:
        await db.close()


def test_a_message_for_a_conversation_whose_turn_is_running_is_answered_busy(api_db):
    client, deps, scripted = _client(api_db, Database(lock_wait_seconds=0.5))
    with client as c:
        cid = c.post("/api/conversations", json={}).json()["conversation_id"]
        with psycopg.connect(api_db.loader_url) as running:  # stands in for a turn holding the conversation's lock
            running.execute("SELECT 1 FROM conversations WHERE id = %s FOR UPDATE", [cid])
            scripted.next(make_form())
            started = time.monotonic()
            r = c.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"})
            waited = time.monotonic() - started
        assert r.status_code == 503 and r.json() == {"error": "database_busy",
                                                      "message": "The database is busy. Try again in a moment."}
        assert waited < 5
        assert c.get(f"/api/conversations/{cid}").json()["turns"] == []
        c.portal.call(deps.db.close)


def test_health_says_busy_when_no_connection_is_free(api_db):
    client, deps, _ = _client(api_db, Database(pool_size=1, pool_wait_seconds=0.3))
    with client as c:
        assert c.get("/health").json() == {"ok": True, "database": "ok"}

        async def hold_the_only_connection():
            async with deps.db.conversation(None):
                r = await asyncio.to_thread(c.get, "/health")
                return r.status_code, r.json()
        assert c.portal.call(hold_the_only_connection) == (200, {"ok": True, "database": "busy"})
        c.portal.call(deps.db.close)
