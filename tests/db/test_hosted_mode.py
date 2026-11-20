"""Hosted mode over real Postgres: the visitor's model key arrives with each message, builds that request's model
clients, and reaches nothing that outlives the request (no table, log line, response body, trace or event)."""

import json
import logging

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient
from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, ScriptedSettler, TurnExtractor, TurnFiller, make_form, stated
from psycopg import sql

from claims_agent import api, hosted
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps, TurnFailed
from claims_agent.models import client as client_module
from claims_agent.observability.turn_log import trace_event
from claims_agent.reply.wording import ScriptedWording
from claims_agent.settings import load_settings, load_thresholds

SENTINEL = "sk-test-SENTINEL-93f1"
KEY = {"X-Model-Key": SENTINEL}


class Scripted:
    """The scripted models a request's clients are built as: `.next(form, said)` queues one turn."""

    def __init__(self) -> None:
        self.filler, self.extractor, self.wording = TurnFiller(), TurnExtractor(), ScriptedWording()

    def next(self, form=None, said=None) -> None:
        self.filler.discard_unused()
        self.extractor.discard_unused()
        self.filler.queue(make_form() if form is None else form)
        self.extractor.queue(stated() if said is None else said)


class ClientFactory:
    """Stands in for building a request's model clients from its key, and records every key it was given. By
    default the clients are the scripted ones; after `answer_with(status, body)` the real clients are built over the
    real model SDK, whose every HTTP request is answered with that status and body (the headers it sent are kept)."""

    def __init__(self, scripted: Scripted, monkeypatch) -> None:
        self.keys: list[str] = []
        self.sent: list[httpx.Request] = []
        self.closed = 0
        self._answer: tuple[int, dict] | None = None
        build, make = hosted.request_clients, client_module.make_client

        def request_clients(settings, key):
            self.keys.append(key.reveal())
            if self._answer is None:
                return self, scripted.filler, scripted.extractor, scripted.wording, ScriptedSettler()
            return build(settings, key)

        def make_client(*, api_key=None, http_client=None):
            return make(api_key=api_key, http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._respond)))

        monkeypatch.setattr(hosted, "request_clients", request_clients)
        monkeypatch.setattr(hosted, "make_client", make_client)

    def answer_with(self, status: int, body: dict) -> None:
        self._answer = (status, body)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        status, body = self._answer
        return httpx.Response(status, json=body)

    async def close(self) -> None:
        self.closed += 1


class RecordingLog:
    """A turn log that keeps exactly what the Braintrust log would send (the masked trace event and ratings)."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def start(self, conversation_id, turn_no):
        return f"span-{conversation_id}-{turn_no}"

    def finish(self, span_id, *, message, reply, record, tags=()):
        self.sent.append(trace_event(message=message, reply=reply, record=record, tags=tags))

    def feedback(self, span_id, *, rating, comment):
        self.sent.append({"span": span_id, "rating": rating, "comment": comment})


@pytest.fixture
def scripted():
    s = Scripted()
    s.next()
    return s


@pytest.fixture
def client_factory(scripted, monkeypatch):
    return ClientFactory(scripted, monkeypatch)


@pytest.fixture
def trace():
    return RecordingLog()


@pytest.fixture
async def api_db(load_dataset):
    return await load_dataset("starter")


def _settings(**hosted_limits):
    """The settings with prepared replies switched ON, so hosted mode must switch them off itself."""
    s = load_settings()
    return s.model_copy(update={"prepared_replies": s.prepared_replies.model_copy(update={"enabled": True}),
                                "hosted": s.hosted.model_copy(update=hosted_limits)})


def _app(api_db, trace, *, hosted_mode=True, settings=None, **kw):
    stand_in = api._NoKey()  # as hosted mode's own dependencies: the server holds no model client of its own
    deps = Deps(settings=settings or _settings(), thresholds=load_thresholds(), db=AppDb(api_db.app_url), guidance=None,
                filler=stand_in, extractor=stand_in, wording=stand_in, settler=stand_in, turn_log=trace)
    return api.create_app(deps, hosted_mode=hosted_mode, **kw)


@pytest.fixture
def hosted_client(api_db, client_factory, trace):
    app = _app(api_db, trace)
    with TestClient(app) as c:
        yield c
        c.portal.call(app.state.deps.db.close)


def dump_all_tables(db) -> str:
    """Every row of every table in the schema, as JSON text, read as the superuser (no row rule hides anything)."""
    with psycopg.connect(db.superuser_url) as conn:
        names = [n for (n,) in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()]
        rows = []
        for name in names:
            rows += [r for (r,) in conn.execute(
                sql.SQL("SELECT to_jsonb(t)::text FROM public.{} t").format(sql.Identifier(name))).fetchall()]
    assert names and rows  # the dump really read the data
    return "\n".join(rows)


def _new(c) -> str:
    return c.post("/api/conversations", json={}).json()["conversation_id"]


# ---- the key builds the request's clients; no key, nothing moves ------------------------------------------------

def test_hosted_mode_builds_the_model_clients_from_the_request_key(hosted_client, client_factory, scripted):
    cid = hosted_client.post("/api/conversations", json={}).json()["conversation_id"]
    hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"}, headers={"X-Model-Key": SENTINEL})
    assert client_factory.keys == [SENTINEL]


def test_no_key_gives_a_clear_message_and_nothing_moves(hosted_client):
    cid = hosted_client.post("/api/conversations", json={}).json()["conversation_id"]
    r = hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"})
    assert r.status_code == 401 and r.json() == {"error": "model_key_needed", "message": "Enter a model key to start."}
    assert hosted_client.get(f"/api/conversations/{cid}").json()["turns"] == []


def test_the_key_never_reaches_the_database_or_the_logs(hosted_client, scripted, api_db, caplog):
    # The dump is read as the superuser, so no grant or row rule can hide a row from it.
    caplog.set_level(logging.DEBUG)
    cid = hosted_client.post("/api/conversations", json={}).json()["conversation_id"]
    hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"}, headers={"X-Model-Key": SENTINEL})
    assert SENTINEL not in dump_all_tables(api_db) and SENTINEL not in caplog.text


def test_nothing_is_prepared_in_hosted_mode(hosted_client): assert hosted_client.app.state.deps.settings.prepared_replies.enabled is False


# ---- the key reaches nothing that outlives the request ------------------------------------------------------------

def test_a_whole_verified_conversation_leaves_the_key_nowhere(hosted_client, client_factory, scripted, api_db, trace,
                                                              caplog):
    caplog.set_level(logging.DEBUG)
    bodies = []
    cid = _new(hosted_client)
    scripted.next(make_form(**GAVE, need="denial_question", claim_types={"healthcare": 0.95},
                            claim_statuses={"denied": 0.95}, months={1: 0.95}, how_many="one"), BRIEF_SAID)
    r = hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": BRIEF_MSG}, headers=KEY)
    assert r.status_code == 200 and r.json()["state"] == "PROCESS_CASE"
    bodies.append(r.text)
    scripted.next(make_form(answer_wanted="appeal_deadline"))
    r = hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": "when is the appeal deadline?"},
                           headers=KEY)
    assert r.status_code == 200
    bodies.append(r.text)
    got = hosted_client.get(f"/api/conversations/{cid}")
    bodies += [got.text, hosted_client.get(f"/api/conversations/{cid}/outbox").text, hosted_client.get("/api/meta").text]
    turn_id = got.json()["turns"][-1]["turn_id"]
    bodies.append(hosted_client.post(f"/api/turns/{turn_id}/feedback", json={"rating": 1, "comment": "fine"}).text)
    assert client_factory.keys == [SENTINEL, SENTINEL] and client_factory.closed == 2  # built per request, then closed
    events = [e for t in got.json()["turns"] for e in t["inspector"].get("events", [])]
    assert events, "the turns recorded events"
    assert len(trace.sent) == 3  # two turns and a rating were traced
    for where, text in [("tables", dump_all_tables(api_db)), ("logs", caplog.text), ("responses", "\n".join(bodies)),
                        ("trace", json.dumps(trace.sent)), ("events", json.dumps(events))]:
        assert SENTINEL not in text, where
        assert "SENTINEL" not in text, where


@pytest.mark.parametrize("status,body,code", [
    (401, {"error": {"message": f"Incorrect API key provided: {SENTINEL}", "type": "invalid_request_error",
                     "code": "invalid_api_key"}}, "model_key_rejected"),
    (429, {"error": {"message": f"You exceeded your current quota with key {SENTINEL}", "type": "insufficient_quota",
                     "code": "insufficient_quota"}}, "model_account_problem"),
    (403, {"error": {"message": f"Key {SENTINEL} may not use this model", "type": "permission_error",
                     "code": "model_not_permitted"}}, "model_not_permitted"),
])
def test_a_key_the_model_refuses_is_the_visitors_problem_and_nothing_is_echoed(hosted_client, client_factory, api_db,
                                                                              caplog, status, body, code):
    caplog.set_level(logging.DEBUG)
    client_factory.answer_with(status, body)  # the real SDK builds and sends the request; the provider echoes the key
    cid = _new(hosted_client)
    r = hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"}, headers=KEY)
    assert r.status_code == {401: 401, 429: 402, 403: 403}[status] and r.json()["error"] == code
    assert set(r.json()) == {"error", "message"} and "OPENAI_API_KEY" not in r.json()["message"]
    assert client_factory.sent and all(q.headers["authorization"] == f"Bearer {SENTINEL}" for q in client_factory.sent)
    assert hosted_client.get(f"/api/conversations/{cid}").json()["turns"] == []
    for text in (r.text, caplog.text, dump_all_tables(api_db)):
        assert "SENTINEL" not in text and "exceeded" not in text and "Incorrect" not in text


def test_a_rejected_key_tells_the_visitor_to_check_their_key(hosted_client, client_factory):
    client_factory.answer_with(401, {"error": {"message": "no", "type": "invalid_request_error", "code": "invalid_api_key"}})
    r = hosted_client.post(f"/api/conversations/{_new(hosted_client)}/messages", json={"text": "hello there"}, headers=KEY)
    assert r.status_code == 401 and r.json() == {"error": "model_key_rejected", "message": api.HOSTED_KEY_REJECTED}


@pytest.mark.parametrize("error,status,code", [(TurnFailed(), 500, "turn_failed"), (RuntimeError("boom"), 500, "internal_error")])
def test_a_failed_turn_never_shows_or_logs_the_key(api_db, client_factory, trace, monkeypatch, caplog, error, status, code):
    caplog.set_level(logging.DEBUG)

    async def fail(deps, *a, **k):
        raise error
    monkeypatch.setattr(api, "run_turn", fail)
    app = _app(api_db, trace)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post(f"/api/conversations/{_new(c)}/messages", json={"text": "hello there"}, headers=KEY)
        c.portal.call(app.state.deps.db.close)
    assert r.status_code == status and r.json()["error"] == code and SENTINEL not in r.text
    assert client_factory.closed == 1  # the key's client is closed even when the turn fails
    assert SENTINEL not in caplog.text


@pytest.mark.parametrize("value,code", [
    ("   ", "model_key_needed"),
    ("sk-has a space", "model_key_invalid"),
    (SENTINEL + "x" * 600, "model_key_invalid"),
    ("sk-caf\xe9".encode("latin-1"), "model_key_invalid"),
])
def test_a_header_that_is_no_usable_key_is_a_fixed_401_and_nothing_moves(hosted_client, client_factory, value, code):
    cid = _new(hosted_client)
    r = hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"}, headers={"X-Model-Key": value})
    assert r.status_code == 401 and r.json() == {"error": code, "message": api.ERRORS[code][1]} and "SENTINEL" not in r.text
    assert client_factory.keys == [] and hosted_client.get(f"/api/conversations/{cid}").json()["turns"] == []


def test_two_key_headers_are_refused(hosted_client, client_factory):
    cid = _new(hosted_client)
    r = hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": "hello there"},
                           headers=[("X-Model-Key", SENTINEL), ("X-Model-Key", "sk-other")])
    assert r.status_code == 401 and r.json()["error"] == "model_key_invalid" and client_factory.keys == []


# ---- what hosted mode switches off -------------------------------------------------------------------------------

def test_no_reply_is_prepared_in_the_background(hosted_client, client_factory, scripted, monkeypatch):
    scheduled = []
    monkeypatch.setattr(api.prepared, "schedule", lambda *a, **k: scheduled.append(a))
    cid = _new(hosted_client)
    scripted.next(make_form(**GAVE, need="denial_question", claim_types={"healthcare": 0.95},
                            claim_statuses={"denied": 0.95}, months={1: 0.95}, how_many="one"), BRIEF_SAID)
    r = hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": BRIEF_MSG}, headers=KEY)
    assert r.json()["state"] == "PROCESS_CASE" and scheduled == []


@pytest.mark.parametrize("from_env", [False, True])
def test_the_claim_change_control_never_exists_in_hosted_mode(api_db, client_factory, trace, monkeypatch, from_env):
    if from_env:
        monkeypatch.setenv("HOSTED", "1")
        monkeypatch.setenv("ADMIN_DATABASE_URL", api_db.loader_url)
        app = _app(api_db, trace, hosted_mode=None)
    else:
        app = _app(api_db, trace, admin_database_url=api_db.loader_url)
    with TestClient(app) as c:
        cid = _new(c)
        assert c.get("/api/meta").json()["admin"] is False
        r = c.patch(f"/api/admin/claims/{cid}/CL-2048", json={"status": "approved"})
        c.portal.call(app.state.deps.db.close)
    assert r.status_code == 404 and r.json()["error"] == "not_found"


async def test_hosted_dependencies_hold_no_model_key_and_ignore_braintrust_unless_opted_in(api_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", api_db.app_url)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-the-servers-own")
    monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-the-servers-own")
    monkeypatch.delenv("HOSTED_BRAINTRUST", raising=False)
    given = []
    monkeypatch.setattr(api, "make_turn_log", lambda s, key: given.append(("trace", key)) or RecordingLog())
    monkeypatch.setattr(api, "startup_examples", _record_examples(given))
    monkeypatch.setattr(api, "make_client", _no_client)
    deps = await api.hosted_deps()
    try:
        assert given == [("trace", None), ("examples", None)]
        assert isinstance(deps.filler, api._NoKey) and isinstance(deps.wording, api._NoKey)
        assert deps.settings.prepared_replies.enabled is False
        monkeypatch.setenv("HOSTED_BRAINTRUST", "1")
        given.clear()
        await (await api.hosted_deps()).db.close()
        assert given == [("trace", "bt-the-servers-own"), ("examples", "bt-the-servers-own")]
    finally:
        await deps.db.close()


def _record_examples(given):
    async def startup_examples(settings, key, db):
        given.append(("examples", key))
        return {}
    return startup_examples


def _no_client(*a, **k):
    raise AssertionError("hosted mode must not build a model client of the server's own")


# ---- strangers ----------------------------------------------------------------------------------------------------

def test_conversations_and_messages_are_rate_limited_per_address(api_db, client_factory, scripted, trace, monkeypatch):
    monkeypatch.setenv(hosted.CLIENT_ADDRESS_HEADER, "X-Visitor-Address")
    app = _app(api_db, trace, settings=_settings(conversations_per_minute=2, messages_per_minute=2))
    with TestClient(app) as c:
        cids = [_new(c), _new(c)]
        r = c.post("/api/conversations", json={})
        assert r.status_code == 429 and r.json() == {"error": "rate_limited", "message": api.ERRORS["rate_limited"][1]}
        other = {"X-Visitor-Address": "203.0.113.9"}
        assert c.post("/api/conversations", json={}, headers=other).status_code == 200  # another visitor
        for _ in range(2):
            scripted.next()
            assert c.post(f"/api/conversations/{cids[0]}/messages", json={"text": "hello there"}, headers=KEY).status_code == 200
        r = c.post(f"/api/conversations/{cids[1]}/messages", json={"text": "hello there"}, headers=KEY)
        assert r.status_code == 429 and r.json()["error"] == "rate_limited"
        assert c.get(f"/api/conversations/{cids[1]}").json()["turns"] == []
        c.portal.call(app.state.deps.db.close)


def test_no_route_lists_conversations_or_returns_an_email_address(hosted_client, scripted):
    r = hosted_client.get("/api/conversations")
    assert r.status_code == 405 and set(r.json()) == {"error", "message"}
    cid = _new(hosted_client)
    scripted.next(make_form(**GAVE), BRIEF_SAID)
    assert hosted_client.post(f"/api/conversations/{cid}/messages", json={"text": BRIEF_MSG}, headers=KEY).status_code == 200
    for path in ("/api/meta", f"/api/conversations/{cid}", f"/api/conversations/{cid}/outbox"):
        assert "@" not in hosted_client.get(path).text, path


def test_meta_says_hosted_and_the_page_keeps_its_policy(hosted_client):
    assert hosted_client.get("/api/meta").json()["hosted"] is True
    r = hosted_client.get("/")
    assert r.headers["content-security-policy"] == api.SECURITY_HEADERS["Content-Security-Policy"]
    assert "connect-src 'self'" in r.headers["content-security-policy"]


def test_outside_hosted_mode_the_header_is_ignored(api_db, client_factory, trace):
    app = _app(api_db, trace, hosted_mode=False)
    with TestClient(app) as c:
        r = c.post(f"/api/conversations/{_new(c)}/messages", json={"text": "hello there"}, headers=KEY)
        assert c.get("/api/meta").json()["hosted"] is False
        c.portal.call(app.state.deps.db.close)
    assert r.status_code == 401 and r.json()["error"] == "model_key_rejected"  # the server's own (absent) key
    assert client_factory.keys == []
