"""Turn logging and ratings: optional, never in a turn's way, and masked before anything is sent."""

import json
import sys
from collections.abc import Sequence
from uuid import UUID

import psycopg
import pytest
from fastapi.testclient import TestClient
from helpers import BRIEF_MSG, BRIEF_SAID, FORM, GAVE, make_form, say, stated, turn_record
from test_api import Scripted

from claims_agent import api
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps
from claims_agent.form.fields import Pick
from claims_agent.observability import BraintrustLog, NoLog, make_turn_log, trace_event
from claims_agent.reply.brief import ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.settings import load_settings, load_thresholds
from claims_agent.states import State


class FakeLog:
    """Records what the engine and the API hand to a turn log."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[dict] = []
        self.feedback_calls: list[tuple] = []
        self.spans: dict[int, str] = {}

    def start(self, conversation_id: UUID, turn_no: int) -> str:
        span = f"span-{len(self.started) + 1}"
        self.started.append(span)
        return span

    def finish(self, span_id, *, message, reply, record, tags: Sequence[str] = ()) -> None:
        self.finished.append({"span_id": span_id, "message": message, "reply": reply, "record": record, "tags": tags})

    def feedback(self, span_id, *, rating, comment) -> None:
        self.feedback_calls.append((span_id, rating, comment))

    def span_of(self, turn_id: int) -> str:
        return self.spans[turn_id]


class FailingLog:
    def start(self, *a, **k):
        raise RuntimeError("braintrust is down")

    finish = feedback = start


@pytest.fixture
def fake_log() -> FakeLog:
    return FakeLog()


@pytest.fixture
def scripted():
    return Scripted()


@pytest.fixture
async def api_db(load_dataset):
    return await load_dataset("starter")


@pytest.fixture
def logged_client(api_db, scripted, fake_log):
    deps = Deps(settings=load_settings(), thresholds=load_thresholds(), db=AppDb(api_db.app_url), guidance=None,
                filler=scripted.filler, extractor=scripted.extractor, wording=scripted.wording, turn_log=fake_log)
    client = TestClient(api.create_app(deps))
    client.scripted, client.database_url = scripted, api_db.app_url
    with client as c:
        yield c
        c.portal.call(deps.db.close)


def two_turns(client):
    cid = client.post("/api/conversations", json={}).json()["conversation_id"]
    ids = []
    for text in ("hello there", "hello again"):
        client.scripted.next(make_form())
        ids.append(client.post(f"/api/conversations/{cid}/messages", json={"text": text}).json()["turn_id"])
    log = client.app.state.deps.turn_log
    with psycopg.connect(client.database_url) as conn:
        for turn_id, span in conn.execute("SELECT id, braintrust_span_id FROM turns").fetchall():
            log.spans[turn_id] = span
    return cid, ids[0], ids[1]


def feedback_rows(client, cid):
    with psycopg.connect(client.database_url) as conn:
        return [(t, int(r)) for t, r in conn.execute(
            "SELECT f.turn_id, f.rating FROM feedback f JOIN turns t ON t.id = f.turn_id "
            "WHERE t.conversation_id = %s ORDER BY f.turn_id", [cid]).fetchall()]


# ---- with no key -------------------------------------------------------------------------------------------------

async def test_with_no_key_nothing_is_imported_and_everything_works(deps, conv):
    assert isinstance(make_turn_log(load_settings(), None), NoLog) and isinstance(make_turn_log(load_settings(), ""), NoLog)
    r = await say(deps, conv, BRIEF_MSG, FORM, BRIEF_SAID)
    assert r.state == State.PROCESS_CASE and "braintrust" not in sys.modules and (await turn_record(deps, r.turn_id))["reply_source"]
    assert r.span_id is None


def test_with_no_key_one_line_says_tracing_is_off(caplog):
    with caplog.at_level("INFO", logger="claims_agent.observability"):
        make_turn_log(load_settings(), None)
    assert [m for m in caplog.messages if "tracing is off" in m] == ["Braintrust tracing is off: BRAINTRUST_API_KEY is not set"]


# ---- ratings -----------------------------------------------------------------------------------------------------

def test_a_thumbs_rating_attaches_to_the_right_turn(logged_client, fake_log):
    cid, first, second = two_turns(logged_client)
    assert logged_client.post(f"/api/turns/{second}/feedback", json={"rating": -1, "comment": "wrong claim"}).json() == {"ok": True}
    assert fake_log.feedback_calls == [(fake_log.span_of(second), 0, "wrong claim")] and feedback_rows(logged_client, cid) == [(second, -1)]


def test_the_latest_rating_wins(logged_client, fake_log):
    cid, first, _ = two_turns(logged_client)
    for rating in (1, -1):
        logged_client.post(f"/api/turns/{first}/feedback", json={"rating": rating})
    assert feedback_rows(logged_client, cid) == [(first, -1)]
    assert [c[1] for c in fake_log.feedback_calls] == [1, 0]


def test_a_rating_must_be_up_or_down(logged_client):
    _, first, _ = two_turns(logged_client)
    r = logged_client.post(f"/api/turns/{first}/feedback", json={"rating": 5})
    assert r.status_code == 422 and set(r.json()) == {"error", "message"}


def test_a_rating_for_a_turn_that_does_not_exist_is_refused_with_fixed_text(logged_client, fake_log):
    two_turns(logged_client)
    r = logged_client.post("/api/turns/99999/feedback", json={"rating": 1, "comment": "secret words"})
    assert r.status_code == 404 and r.json() == {"error": "turn_not_found", "message": "There is no such turn."}
    assert "secret" not in r.text and fake_log.feedback_calls == []


@pytest.mark.parametrize("turn_id", ["abc", "0", "-3", "9223372036854775808"])
def test_a_turn_id_must_be_a_positive_integer_that_fits(logged_client, turn_id):
    r = logged_client.post(f"/api/turns/{turn_id}/feedback", json={"rating": 1})
    assert r.status_code == 422 and set(r.json()) == {"error", "message"}


def test_a_rating_body_has_a_size_cap_and_no_extra_fields(logged_client):
    _, first, _ = two_turns(logged_client)
    assert logged_client.post(f"/api/turns/{first}/feedback", json={"rating": 1, "comment": "x" * 70000}).status_code == 413
    assert logged_client.post(f"/api/turns/{first}/feedback", json={"rating": 1, "comment": "x" * 1001}).status_code == 422
    assert logged_client.post(f"/api/turns/{first}/feedback", json={"rating": 1, "extra": 1}).status_code == 422
    assert logged_client.post(f"/api/turns/{first}/feedback", json={"rating": True}).status_code == 422


def test_a_restored_conversation_gives_each_turns_id(logged_client):
    cid, first, second = two_turns(logged_client)
    got = logged_client.get(f"/api/conversations/{cid}").json()
    assert [t["turn_id"] for t in got["turns"]] == [first, second]


# ---- a failing Braintrust never reaches the caller -----------------------------------------------------------------

async def test_braintrust_being_down_does_not_fail_the_turn(deps, conv):
    deps.turn_log = FailingLog()
    r = await say(deps, conv, "hello there", make_form())
    assert r.reply and (await turn_record(deps, r.turn_id))["state_after"] == "VERIFY_ID"


def test_braintrust_being_down_does_not_fail_a_rating(logged_client):
    cid, first, _ = two_turns(logged_client)
    logged_client.app.state.deps.turn_log = FailingLog()
    assert logged_client.post(f"/api/turns/{first}/feedback", json={"rating": 1}).json() == {"ok": True}
    assert feedback_rows(logged_client, cid) == [(first, 1)]


# ---- what is sent ------------------------------------------------------------------------------------------------

async def test_the_record_sent_is_the_record_stored(deps, conv, fake_log):
    deps.turn_log = fake_log
    r = await say(deps, conv, "hello there", make_form())
    assert fake_log.finished[0]["record"] == await turn_record(deps, r.turn_id)
    assert fake_log.finished[0]["span_id"] == "span-1"


async def test_the_span_id_is_saved_with_the_turn(deps, conv, fake_log):
    deps.turn_log = fake_log
    r = await say(deps, conv, "hello there", make_form())
    async with deps.db.conversation(None) as c:
        row = await (await c.execute("SELECT braintrust_span_id FROM turns WHERE id = %s", [r.turn_id])).fetchone()
    assert row[0] == "span-1" == r.span_id


async def test_a_turn_with_an_unsure_field_is_tagged_for_review(deps, conv, fake_log):
    await say(deps, conv, BRIEF_MSG, FORM, BRIEF_SAID)
    deps.turn_log = fake_log
    await say(deps, conv, "what about the timing?", make_form(answer_wanted=Pick(value="submission_timing", p=0.45, options={"submission_timing": 0.45, "processing_time_after_submission": 0.4})))
    assert list(fake_log.finished[-1]["tags"]) == ["needs_review"]


async def test_a_sure_turn_carries_no_tag(deps, conv, fake_log):
    deps.turn_log = fake_log
    await say(deps, conv, "hello there", make_form())
    assert list(fake_log.finished[-1]["tags"]) == []


async def test_a_trace_holds_no_id_number_email_or_phone(deps, conv, fake_log):
    deps.turn_log = fake_log
    message = ("My SSN is 123-45-4472 and also 123454472, email margaret@email.com, phone +16505212836, "
               "name Margaret Chen, policy POL-9921.")
    r = await say(deps, conv, message, make_form(**GAVE),
                  stated(policyholder_name="Margaret Chen", policy_number="POL-9921", id_digits="123454472"))
    sent = fake_log.finished[-1]
    event = trace_event(message=message, reply=r.reply, record=sent["record"], tags=sent["tags"])
    text = json.dumps(event)
    for secret in ("123454472", "123-45-4472", "margaret@email.com", "16505212836", "6505212836"):
        assert secret not in text, secret
    assert "4472" in event["input"]  # the last four stay, as the inspector shows them
    brief = ReplyBrief(state=State.VERIFY_ID, parts=(ReplyPart("anything_else"),), mood="calm", caller_message="",
                       previous_reply=None, english=True, verified=True)
    checked = check_reply(text, brief, wary_status_words=load_settings().reply_check.wary_status_words, data_statuses=())
    assert checked.leak == ()


def test_a_comment_is_masked_before_it_is_sent():
    sent = []

    class Logger:
        def log_feedback(self, **kw): sent.append(kw)
        def flush(self): pass

    log = BraintrustLog.__new__(BraintrustLog)
    log._logger = Logger()
    log.feedback("span-1", rating=0, comment="it read out 123-45-4472 and a@b.com")
    assert sent[0]["comment"] == "it read out ***-**-4472 and [email]" and sent[0]["id"] == "span-1"


# ---- the real class, over a stand-in for the library ---------------------------------------------------------------

class _Span:
    def __init__(self, calls, kw): self.calls, self.kw = calls, kw
    def log(self, **kw): self.calls.append(("log", kw))
    def end(self): self.calls.append(("end",))


class _Logger:
    def __init__(self, calls): self.calls = calls
    def start_span(self, **kw):
        self.calls.append(("start_span", kw))
        return _Span(self.calls, kw)
    def log_feedback(self, **kw): self.calls.append(("log_feedback", kw))
    def flush(self): self.calls.append(("flush",))


def test_the_braintrust_log_uses_the_library_the_documented_way(monkeypatch):
    import types
    calls = []
    fake = types.ModuleType("braintrust")
    fake.set_masking_function = lambda f: calls.append(("mask", f))
    fake.init_logger = lambda **kw: (calls.append(("init_logger", kw)), _Logger(calls))[1]
    monkeypatch.setitem(sys.modules, "braintrust", fake)
    log = make_turn_log(load_settings(), "a-key")
    assert isinstance(log, BraintrustLog) and calls[1] == ("init_logger", {"project": "sop-claims-agent", "api_key": "a-key"})
    span = log.start(UUID(int=1), 3)
    log.finish(span, message="hi", reply="hello", record={"a": 1}, tags=["needs_review"])
    log.feedback(span, rating=1, comment=None)
    for t in __import__("threading").enumerate():
        if t.name == "braintrust-flush":
            t.join(5)
    names = [c[0] for c in calls]
    assert names[:2] == ["mask", "init_logger"] and names.count("start_span") == 1 and "end" in names
    start = next(c[1] for c in calls if c[0] == "start_span")
    assert start["id"] == span and start["name"] == "turn"
    logged = next(c[1] for c in calls if c[0] == "log")
    assert logged["input"] == "hi" and logged["output"] == "hello" and logged["tags"] == ["needs_review"]
    assert logged["metadata"]["turn_no"] == 3 and logged["metadata"]["record"] == {"a": 1}
    assert ("log_feedback", {"id": span, "scores": {"rating": 1}, "comment": None}) in calls


def test_a_library_that_cannot_start_leaves_logging_off(monkeypatch, caplog):
    import types
    fake = types.ModuleType("braintrust")
    fake.set_masking_function = lambda f: None
    fake.init_logger = lambda **kw: (_ for _ in ()).throw(RuntimeError("no network"))
    monkeypatch.setitem(sys.modules, "braintrust", fake)
    assert isinstance(make_turn_log(load_settings(), "a-key"), NoLog)
