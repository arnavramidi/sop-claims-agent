"""The demo control that changes a claim: off unless an admin database address is set, and then confined to the
claims of the customer verified in the conversation it is used from."""

import json

import psycopg
import pytest
from fastapi.testclient import TestClient
from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, make_form
from test_api import Scripted

from claims_agent import api
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps
from claims_agent.settings import load_settings, load_thresholds


@pytest.fixture
def scripted():
    return Scripted()


@pytest.fixture
async def api_db(load_dataset):
    return await load_dataset("starter")


def make_client(api_db, scripted, admin_url):
    deps = Deps(settings=load_settings(), thresholds=load_thresholds(), db=AppDb(api_db.app_url), guidance=None,
                filler=scripted.filler, extractor=scripted.extractor, wording=scripted.wording)
    return TestClient(api.create_app(deps, admin_database_url=admin_url)), deps


@pytest.fixture
def client(api_db, scripted, monkeypatch):
    monkeypatch.delenv("ADMIN_DATABASE_URL", raising=False)
    c, deps = make_client(api_db, scripted, None)
    with c:
        yield c
        c.portal.call(deps.db.close)


@pytest.fixture
def admin_client(api_db, scripted):
    c, deps = make_client(api_db, scripted, api_db.demo_url)
    with c:
        yield c
        c.portal.call(deps.db.close)


def verified_in_case(c, scripted) -> str:
    cid = c.post("/api/conversations", json={}).json()["conversation_id"]
    scripted.next(make_form(**GAVE, need="denial_question", claim_types={"healthcare": 0.95},
                            claim_statuses={"denied": 0.95}, months={1: 0.95}, how_many="one"), BRIEF_SAID)
    assert c.post(f"/api/conversations/{cid}/messages", json={"text": BRIEF_MSG}).status_code == 200
    return cid


def change(c, cid, case, body):
    return c.patch(f"/api/admin/claims/{cid}/{case}", json=body)


def read_claim(api_db, case):
    with psycopg.connect(api_db.loader_url) as conn:
        return conn.execute("SELECT status, appeal_deadline, updated_at FROM claims WHERE case_id = %s", [case]).fetchone()


def test_the_control_does_not_exist_by_default(client, scripted):
    cid = verified_in_case(client, scripted)
    assert client.get("/api/meta").json()["admin"] is False
    r = change(client, cid, "CL-2048", {"status": "open"})
    unknown = client.patch("/api/admin/nothing", json={})
    assert r.status_code == 404 and r.json() == unknown.json() == {"error": "not_found", "message": "There is nothing here."}


def test_meta_says_when_the_control_is_on(admin_client):
    assert admin_client.get("/api/meta").json()["admin"] is True


def test_a_change_shows_in_the_next_reply(admin_client, scripted, api_db):
    cid = verified_in_case(admin_client, scripted)
    before = read_claim(api_db, "CL-2048")
    r = change(admin_client, cid, "CL-2048", {"appeal_deadline": "2026-03-25", "status": "open"})
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["updated_at"]
    after = read_claim(api_db, "CL-2048")
    assert after[0] == "open" and str(after[1]) == "2026-03-25" and after[2] > before[2]
    scripted.next(make_form(answer_wanted="appeal_deadline"))
    body = admin_client.post(f"/api/conversations/{cid}/messages", json={"text": "what's my deadline?"}).json()
    parts = json.dumps(body["inspector"]["parts"])
    assert "2026-03-25" in parts and "20 days" in parts


def test_documents_needed_are_replaced_and_numbered(admin_client, scripted, api_db):
    cid = verified_in_case(admin_client, scripted)
    assert change(admin_client, cid, "CL-2048", {"documents_needed": ["a letter", "a bill"]}).status_code == 200
    with psycopg.connect(api_db.loader_url) as conn:
        rows = conn.execute("SELECT position, name FROM claim_documents WHERE case_id = 'CL-2048' ORDER BY position").fetchall()
    assert rows == [(1, "a letter"), (2, "a bill")]


def test_an_unknown_claim_is_a_404_and_bad_values_a_422(admin_client, scripted):
    cid = verified_in_case(admin_client, scripted)
    assert change(admin_client, cid, "CL-0000", {"status": "open"}).status_code == 404
    for bad in ({"status": ""}, {"status": "not a real status"}, {"appeal_deadline": "2026-02-30"},
                {"appeal_deadline": "tomorrow"}, {"documents_needed": [" "]}, {}, {"net_pay": "1.00"}, {"status": 5}):
        r = change(admin_client, cid, "CL-2048", bad)
        assert r.status_code == 422 and r.json() == {
            "error": "invalid_request", "message": "The request was not in the expected form."}, bad


def test_another_customers_claim_answers_like_a_missing_one(admin_client, scripted, api_db):
    cid = verified_in_case(admin_client, scripted)
    with psycopg.connect(api_db.loader_url) as conn:
        other = conn.execute("SELECT case_id, status FROM claims WHERE party_id <> 'P9' LIMIT 1").fetchone()
    missing = change(admin_client, cid, "CL-0000", {"status": "open"})
    r = change(admin_client, cid, other[0], {"status": "open"})
    assert r.status_code == 404 and r.json() == missing.json()
    assert read_claim(api_db, other[0])[0] == other[1]


def test_a_conversation_not_yet_verified_can_change_nothing(admin_client, api_db):
    cid = admin_client.post("/api/conversations", json={}).json()["conversation_id"]
    assert change(admin_client, cid, "CL-2048", {"status": "open"}).status_code == 404
    assert read_claim(api_db, "CL-2048")[0] != "open"


def test_an_unknown_conversation_is_a_404(admin_client):
    r = change(admin_client, "00000000-0000-0000-0000-000000000000", "CL-2048", {"status": "open"})
    assert r.status_code == 404 and r.json()["error"] == "conversation_not_found"


def test_a_body_that_is_too_large_is_refused(admin_client, scripted):
    cid = verified_in_case(admin_client, scripted)
    r = change(admin_client, cid, "CL-2048", {"documents_needed": ["x" * 100] * 1000})
    assert r.status_code == 413 and r.json()["error"] == "body_too_large"


async def test_the_apps_own_account_still_cannot_write_a_claim(api_db, admin_client):
    async with await psycopg.AsyncConnection.connect(api_db.app_url) as conn:
        for statement in ("UPDATE claims SET status = 'open'", "DELETE FROM claims", "UPDATE claim_documents SET name = 'x'",
                          "INSERT INTO claims (case_id, party_id, status) VALUES ('CL-9', 'P9', 'open')"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                await conn.execute(statement)
            await conn.rollback()


# ---- the demo account writes nothing else ---------------------------------------------------------------------

async def _as_demo(api_db, conversation_id, statement, params=()):
    async with await psycopg.AsyncConnection.connect(api_db.demo_url) as conn:
        if conversation_id is not None:
            await conn.execute("SELECT set_config('app.conversation_id', %s, true)", [str(conversation_id)])
        cur = await conn.execute(statement, params)
        return cur.rowcount if cur.description is None else await cur.fetchall()


async def test_the_demo_account_reads_no_claim_detail_and_no_customer(api_db):
    for statement in ("SELECT summary FROM claims", "SELECT denial_reason FROM claims", "SELECT net_pay FROM claims",
                      "SELECT name FROM claim_documents", "SELECT * FROM customers", "SELECT * FROM turns",
                      "SELECT memory FROM conversations", "SELECT * FROM outbox", "SELECT * FROM guidance"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await _as_demo(api_db, None, statement)


async def test_the_demo_account_changes_only_the_two_columns_and_documents(api_db):
    for statement in ("UPDATE claims SET net_pay = 1", "UPDATE claims SET party_id = 'P7'", "DELETE FROM claims",
                      "INSERT INTO claims (case_id, party_id, status) VALUES ('CL-9', 'P9', 'open')",
                      "UPDATE claim_documents SET name = 'x'", "UPDATE conversations SET verified_party_id = 'P9'"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await _as_demo(api_db, None, statement)


async def test_the_demo_account_reaches_only_the_bound_conversations_customer(admin_client, scripted, api_db):
    cid = verified_in_case(admin_client, scripted)  # verified as P9
    assert await _as_demo(api_db, None, "UPDATE claims SET status = 'open'") == 0
    assert await _as_demo(api_db, cid, "UPDATE claims SET status = 'open' WHERE party_id <> 'P9'") == 0
    with psycopg.connect(api_db.loader_url) as conn:
        other = conn.execute("SELECT case_id FROM claims WHERE party_id <> 'P9' LIMIT 1").fetchone()[0]
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        await _as_demo(api_db, cid, "INSERT INTO claim_documents (case_id, party_id, position, name) "
                                    "VALUES (%s, 'P7', 9, 'x')", [other])
    assert await _as_demo(api_db, cid, "DELETE FROM claim_documents WHERE party_id <> 'P9'") == 0
    assert await _as_demo(api_db, cid, "UPDATE claims SET status = 'open' WHERE party_id = 'P9'") > 0
