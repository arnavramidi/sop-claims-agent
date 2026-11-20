"""The HTTP API over real Postgres with scripted models: the routes, the error shape, the page and its headers."""

import json

import pytest
from fastapi.testclient import TestClient
from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, TurnExtractor, TurnFiller, make_form, stated
from psycopg_pool import PoolTimeout

from claims_agent import api
from claims_agent.db.identity import send_summary
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps, TurnFailed
from claims_agent.models.errors import ModelAccountProblem, ModelKeyRejected, ModelNotPermitted
from claims_agent.reply.wording import ScriptedWording
from claims_agent.settings import load_settings, load_thresholds


class Scripted:
    """The scripted models behind `client`: `.next(form, said)` queues one turn's form and stated values (whatever
    an earlier turn did not use is dropped first); `.fail_with(error)` makes the next model call fail."""

    def __init__(self) -> None:
        self.filler, self.extractor, self.wording = TurnFiller(), TurnExtractor(), ScriptedWording()

    def next(self, form, said=None) -> None:
        self.filler.discard_unused()
        self.extractor.discard_unused()
        self.filler.queue(form)
        self.extractor.queue(stated() if said is None else said)

    def fail_with(self, error: BaseException) -> None:
        self.filler.discard_unused()
        self.extractor.discard_unused()
        self.filler.fail_with(error)
        self.extractor.fail_with(error)


@pytest.fixture
def scripted():
    return Scripted()


@pytest.fixture
async def api_db(load_dataset):
    return await load_dataset("starter")


def _client(api_db, scripted, **kw):
    """A TestClient whose app has its own pool, opened and closed in the client's own event loop."""
    deps = Deps(settings=load_settings(), thresholds=load_thresholds(), db=AppDb(api_db.app_url), guidance=None,
                filler=scripted.filler, extractor=scripted.extractor, wording=scripted.wording)
    return TestClient(api.create_app(deps), **kw), deps


@pytest.fixture
def client(api_db, scripted):
    c, deps = _client(api_db, scripted)
    with c:
        yield c
        c.portal.call(deps.db.close)


def start_and_say(client, scripted, text) -> str:
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    scripted.next(make_form())
    assert client.post(f"/api/conversations/{cid}/messages", json={"text": text}).status_code == 200
    return cid


@pytest.fixture
def wrapped_and_sent(client, scripted):
    """A verified conversation with one summary email in the outbox. Written by the database's own send_summary
    inside a transaction bound to that conversation (the wrap-up state that sends it is another task's)."""
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    scripted.next(make_form(**GAVE), BRIEF_SAID)
    assert client.post(f"/api/conversations/{cid}/messages", json={"text": BRIEF_MSG}).status_code == 200
    db = client.app.state.deps.db

    async def send():
        async with db.conversation(cid) as conn:
            await send_summary(conn, cid, "Your claim summary", "Claim CL-2048: denied.")
    client.portal.call(send)
    return cid


# ---- the brief's tests ----------------------------------------------------------------------------------------

def test_a_conversation_runs_the_brief_message(client, scripted):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    scripted.next(make_form(**GAVE, need="denial_question", claim_types={"healthcare": 0.95}, claim_statuses={"denied": 0.95}, months={1: 0.95}, how_many="one"), BRIEF_SAID)
    body = client.post(f"/api/conversations/{cid}/messages", json={"text": BRIEF_MSG}).json()
    assert body["state"] == "PROCESS_CASE" and body["inspector"]["claims_visible"] == 4
    assert {"state_before", "state_after", "freedom", "filler", "form", "stated", "flags", "fired", "parts",
            "reply_source", "checks", "memory", "counts", "claims_visible", "outbox", "handover_note"} <= set(body["inspector"])
    # Plan correction: Task 11's pinned payload gives freedom as {"level", and the three switches}, not a bare name.
    assert body["inspector"]["freedom"]["level"] == "free"


def test_a_rejected_key_is_a_clear_401_and_nothing_moves(client, scripted):
    # Plan correction: model errors take the model's name (deviations, Task 11 helpers line); GET also returns the
    # conversation's controls (ruling for Part B).
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]; scripted.fail_with(ModelKeyRejected("scripted"))
    r = client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"})
    assert r.status_code == 401 and r.json() == {"error": "model_key_rejected", "message": "The model key was rejected. Check OPENAI_API_KEY."}
    assert client.get(f"/api/conversations/{cid}").json() == {
        "state": "VERIFY_ID", "turns": [], "controls": {"consent_script": "default", "today": "2026-03-05"}}


def test_reload_restores_the_transcript(client, scripted):
    cid = start_and_say(client, scripted, "hello there"); got = client.get(f"/api/conversations/{cid}").json()
    assert [t["message"] for t in got["turns"]] == ["hello there"] and got["turns"][0]["reply"]


def test_v11_the_inspector_masks_a_typed_full_id(client, scripted):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]; scripted.next(make_form(**GAVE), stated(id_digits="123454472"))
    assert "12345" not in json.dumps(client.post(f"/api/conversations/{cid}/messages", json={"text": "my ssn is 123454472"}).json()["inspector"])


def test_controls_set_the_consent_script_and_today(client):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    assert client.patch(f"/api/conversations/{cid}/controls", json={"consent_script": "timeout", "today": "2026-03-20"}).status_code == 200


def test_the_outbox_never_returns_an_address(client, wrapped_and_sent):
    assert client.get(f"/api/conversations/{wrapped_and_sent}/outbox").json()[0]["recipient"] == "the policyholder's email on file"


def test_meta_lists_scripts_and_the_load_report(client):
    m = client.get("/api/meta").json(); assert set(m["consent_scripts"]) == {"default", "timeout"} and m["load_report"] == []


# ---- shapes the page reads ------------------------------------------------------------------------------------

def test_create_get_and_controls_carry_the_controls(client, scripted):
    made = client.post("/api/conversations", json={"consent_script": "timeout", "today": "2026-02-01"}).json()
    assert made["state"] == "VERIFY_ID" and made["controls"] == {"consent_script": "timeout", "today": "2026-02-01"}
    cid = made["conversation_id"]
    r = client.patch(f"/api/conversations/{cid}/controls", json={"today": "2026-03-20"})
    assert r.json() == {"controls": {"consent_script": "timeout", "today": "2026-03-20"}}
    assert client.get(f"/api/conversations/{cid}").json()["controls"] == {"consent_script": "timeout", "today": "2026-03-20"}


def test_a_restored_turn_carries_its_record_as_inspector(client, scripted):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    scripted.next(make_form())
    sent = client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"}).json()
    assert set(sent) == {"turn_id", "reply", "state", "inspector"}
    turn = client.get(f"/api/conversations/{cid}").json()["turns"][0]
    assert set(turn) == {"turn_no", "message", "reply", "state_after", "inspector"}
    assert turn["inspector"] == sent["inspector"] and turn["reply"] == sent["reply"]


def test_the_outbox_holds_no_address(client, wrapped_and_sent):
    rows = client.get(f"/api/conversations/{wrapped_and_sent}/outbox").json()
    assert [set(r) for r in rows] == [{"subject", "body", "recipient"}] and "@" not in json.dumps(rows)


def test_no_endpoint_lists_conversations(client):
    r = client.get("/api/conversations")
    assert r.status_code == 405 and set(r.json()) == {"error", "message"}


# ---- every failure is {error, message} with fixed text -------------------------------------------------------

def _is_error(r, status, code):
    body = r.json()
    assert r.status_code == status and set(body) == {"error", "message"} and body["error"] == code, (r.status_code, body)
    return body


@pytest.mark.parametrize("error,status,code", [
    (ModelAccountProblem("scripted", "secret detail"), 402, "model_account_problem"),
    (ModelNotPermitted("scripted", "secret detail"), 403, "model_not_permitted"),
])
def test_operator_problems_are_distinct_and_nothing_moves(client, scripted, error, status, code):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]; scripted.fail_with(error)
    body = _is_error(client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"}), status, code)
    assert "secret" not in body["message"] and "scripted" not in body["message"]
    assert client.get(f"/api/conversations/{cid}").json()["turns"] == []


@pytest.mark.parametrize("error,status,code", [
    (TurnFailed(), 500, "turn_failed"), (PoolTimeout("secret detail"), 503, "database_busy")])
def test_a_failed_turn_and_a_busy_pool_have_their_own_messages(client, monkeypatch, error, status, code):
    async def fail(*a, **k):
        raise error
    monkeypatch.setattr(api, "run_turn", fail)
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    assert "secret" not in _is_error(client.post(f"/api/conversations/{cid}/messages", json={"text": "hi there"}),
                                     status, code)["message"]


def test_bad_requests_get_fixed_messages(client):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    _is_error(client.get("/api/conversations/not-a-uuid"), 422, "invalid_request")
    _is_error(client.get("/api/conversations/00000000-0000-4000-8000-000000000000"), 404, "conversation_not_found")
    _is_error(client.get("/nowhere"), 404, "not_found")
    _is_error(client.delete(f"/api/conversations/{cid}"), 405, "method_not_allowed")
    _is_error(client.post(f"/api/conversations/{cid}/messages", json={"txt": "<b>x</b>"}), 422, "invalid_request")
    _is_error(client.post(f"/api/conversations/{cid}/messages", content=b"{not json"), 422, "invalid_request")
    _is_error(client.post(f"/api/conversations/{cid}/messages", json={"text": "   "}), 422, "empty_message")
    _is_error(client.patch(f"/api/conversations/{cid}/controls", json={"today": "soon"}), 422, "invalid_request")
    for bad in ("<b>nope</b>", "DEFAULT"):
        body = _is_error(client.post("/api/conversations", json={"consent_script": bad}), 422, "unknown_consent_script")
        assert bad not in body["message"]
        _is_error(client.patch(f"/api/conversations/{cid}/controls", json={"consent_script": bad}), 422,
                  "unknown_consent_script")
    assert client.get(f"/api/conversations/{cid}").json()["controls"]["consent_script"] == "default"


def test_message_and_body_size_are_capped_before_anything_reads_them(client, scripted):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    cap = load_settings().limits.message_chars
    _is_error(client.post(f"/api/conversations/{cid}/messages", json={"text": "a" * (cap + 1)}), 413, "message_too_long")
    _is_error(client.post(f"/api/conversations/{cid}/messages", content=b"x" * (api.MAX_BODY_BYTES + 1),
                          headers={"content-type": "application/json"}), 413, "body_too_large")

    def chunks():  # no Content-Length: the size is counted as the body arrives
        for _ in range(api.MAX_BODY_BYTES // 1000 + 2):
            yield b"x" * 1000
    _is_error(client.post(f"/api/conversations/{cid}/messages", content=chunks()), 413, "body_too_large")
    assert client.get(f"/api/conversations/{cid}").json()["turns"] == []


def test_an_unexpected_error_is_a_fixed_500(api_db, scripted):
    class Broken:
        def conversation(self, cid):
            raise RuntimeError("secret detail from the database")
    c, deps = _client(api_db, scripted, raise_server_exceptions=False)
    c.app.state.deps.db = Broken()
    with c:
        body = _is_error(c.get("/api/meta"), 500, "internal_error")
    assert "secret" not in body["message"]


def test_a_missing_key_gives_the_rejected_key_body(client, scripted, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    deps = client.app.state.deps
    deps.filler, deps.extractor, deps.wording = api.model_clients(deps.settings)
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    r = client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"})
    assert r.status_code == 401 and r.json() == {"error": "model_key_rejected", "message": "The model key was rejected. Check OPENAI_API_KEY."}
    assert client.get("/api/meta").json()["filler"] == "decisions"


# ---- the page --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path,kind", [("/", "text/html"), ("/app.js", "text/javascript"), ("/styles.css", "text/css")])
def test_the_page_and_its_files_carry_the_policy(client, path, kind):
    r = client.get(path)
    assert r.status_code == 200 and r.headers["content-type"].startswith(kind)
    assert r.headers["content-security-policy"] == (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'; require-trusted-types-for 'script'")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["referrer-policy"] == "no-referrer"
