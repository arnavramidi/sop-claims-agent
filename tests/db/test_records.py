"""Event records: what each model was asked and answered, what each rule decided, written in the turn's own
transaction, over real Postgres with scripted models."""

import json

import pytest
from helpers import (BRIEF_MSG, BRIEF_SAID, GAVE, assert_no_claim_data, focus, make_form, memory, new_conversation,
                     say, stated, turn_record)

from claims_agent.engine import mask_ids
from claims_agent.form.fields import Pick
from claims_agent.models.errors import ModelKeyRejected, ModelUnavailable
from claims_agent.records import EVENT_KINDS, Recorder, fit
from claims_agent.states import State

REP = stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472", caller_name="David Chen")
THIRD = dict(gave_identity_detail=True, caller_role="third_party", relationship="son")
RECHECK = make_form(consent_recheck=True, caller_role="third_party")


async def _all(deps, conv, query="SELECT kind, body FROM events WHERE conversation_id = %s ORDER BY id"):
    async with deps.db.conversation(conv) as c:
        return await (await c.execute(query, [conv])).fetchall()


async def events(deps, conv, kind):
    """The bodies of one kind for the conversation, in order."""
    return [b for k, b in await _all(deps, conv) if k == kind]


async def event_count(deps, conv) -> int:
    return len(await _all(deps, conv))


@pytest.fixture
async def margaret(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
                  stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert r.state == State.RESOLVE_INTENT
    return cid


@pytest.fixture
async def denial_case(deps, margaret):
    r = await say(deps, margaret, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                  stated(claim_number="CL-2048"))
    assert r.state == State.PROCESS_CASE and await focus(deps, margaret) == ["CL-2048"]
    return margaret


@pytest.fixture
async def waiting_conv(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, "I'm David, calling for my mother", make_form(**THIRD), REP)
    assert r.state == State.AWAIT_CONSENT
    return cid


async def test_every_escalation_has_a_reason_a_destination_and_the_transcript(deps, denial_case):
    deps.settler.queue(Pick(value=None, by="larger"))
    await say(deps, denial_case, "what about the timing?", make_form(answer_wanted=Pick(value=None)))
    await say(deps, denial_case, "a person please", make_form(wants_person=True))
    got = await events(deps, denial_case, "escalation")
    assert [e["went_to"] for e in got] == ["larger_model", "caller", "person"] and all(e["why"] and e["transcript"] for e in got)
    assert got[-1]["transcript"][-1]["message"] == "a person please"


async def test_every_field_below_its_threshold_has_a_record(deps, denial_case):
    await say(deps, denial_case, "what about the timing?", make_form(answer_wanted=Pick(
        value="submission_timing", p=0.45, options={"submission_timing": 0.45, "processing_time_after_submission": 0.4})))
    e = (await events(deps, denial_case, "below_threshold"))[0]
    assert e["field"] == "answer_wanted" and e["threshold"] == 0.6 and e["probabilities"]["submission_timing"] == 0.45 and e["went_to"]
    assert e["options"] == ["processing_time_after_submission", "submission_timing"] and e["transcript"]


async def test_extractions_are_recorded_with_the_three_flags(deps, conv):
    await say(deps, conv, "born 30 Feb 1985", make_form(**GAVE), stated(date_of_birth=["1985-02-30"]))
    await say(deps, conv, "4472", make_form(gave_identity_detail=False), stated(id_digits="4472"))
    await say(deps, conv, "born 16 March 1985", make_form(**GAVE), stated(date_of_birth=["1985-03-16"]))
    await say(deps, conv, "sorry, the 15th", make_form(**GAVE, correcting_detail=True), stated(date_of_birth=["1985-03-15"]))
    flags = [e["flags"] for e in await events(deps, conv, "extraction")]
    assert flags == [["format_failed"], ["cross_check_failed"], ["corrected_next_turn"], []]


async def test_every_reply_has_its_method_facts_text_and_pass_or_fail(deps, denial_case):
    r = await say(deps, denial_case, "status?", make_form(answer_wanted="status"))
    rec = await turn_record(deps, r.turn_id)
    assert rec["reply_source"] and rec["parts"] and rec["checks"][0]["passed"] in (True, False) and rec["fired"]


async def test_a_resolved_need_lists_the_messages_it_came_from(deps, conv):
    await say(deps, conv, "why was my healthcare claim denied?", make_form(
        need="denial_question", claim_types={"healthcare": 0.9}, claim_statuses={"denied": 0.9}, how_many="one"))
    await say(deps, conv, BRIEF_MSG, make_form(**GAVE, shaped_need=[0.9, 0.2]), BRIEF_SAID)
    e = (await events(deps, conv, "resolved_need"))[0]
    assert e["need"] == "denial_question" and e["set_on_turn"] == 1 and e["shaped_by"][0]["message"].startswith("why was my healthcare")
    assert e["shaped_by"][0]["turn"] == 1 and len(e["shaped_by"]) == 1


async def test_two_mood_changes_are_recorded_with_message_and_reply(deps, denial_case):
    for mood in ("calm", "upset", "calm"):
        await say(deps, denial_case, f"({mood}) status?", make_form(answer_wanted="status", mood=mood))
    got = await events(deps, denial_case, "mood_change")
    assert [(e["before"], e["after"]) for e in got] == [("calm", "upset"), ("upset", "calm")] and all(e["message"] and e["reply"] for e in got)


async def test_each_consent_check_is_recorded(deps, waiting_conv):
    await say(deps, waiting_conv, "check again", RECHECK)
    assert [(e["number"], e["script"], e["result"]) for e in await events(deps, waiting_conv, "consent_check")] == [
        (1, "default", "pending"), (2, "default", "approved")]


async def test_followup_choice_and_wrap_up_are_recorded(deps, denial_case):
    await say(deps, denial_case, "that's all", make_form(done=True))
    await say(deps, denial_case, "yes", make_form(email_offer_reply=Pick(value="send", p=0.95)))
    assert (await events(deps, denial_case, "followup_choice"))[0]["chosen"] == "missing_required_material_alternatives"
    w = (await events(deps, denial_case, "wrap_up"))[-1]
    assert w["action"] == "email_sent" and "CL-2048" in w["email_text"] and w["caller_reply"] == "yes"
    assert w["form_value"] == {"value": "send", "p": 0.95} and w["offer_as_worded"]


async def test_a_rejected_key_leaves_no_events(deps, conv):
    deps.filler.fail_with(ModelKeyRejected("scripted"))
    with pytest.raises(ModelKeyRejected):
        await say(deps, conv, "hello there")
    assert await event_count(deps, conv) == 0


async def test_a_turn_that_fails_leaves_only_the_hand_overs_own_events(deps, denial_case):
    before = await event_count(deps, denial_case)
    deps.settler.queue(Pick(value=None, by="larger"))
    deps.filler.fail_with(ModelUnavailable("scripted"))
    r = await say(deps, denial_case, "what about the timing?")
    assert r.state == State.ESCALATED
    new = (await _all(deps, denial_case))[before:]
    assert [(k, b["went_to"], b["why"]) for k, b in new] == [("escalation", "person", "model_failure")]


async def test_the_events_are_in_the_inspector_and_in_the_stored_record(deps, denial_case):
    r = await say(deps, denial_case, "a person please", make_form(wants_person=True))
    rec = await turn_record(deps, r.turn_id)
    assert rec["events"] == r.inspector["events"]
    assert [e["kind"] for e in rec["events"]] == ["escalation"] and rec["events"][0]["body"]["went_to"] == "person"


async def test_a_pre_verification_turn_has_events_with_no_claim_data(deps, conv):
    deps.settler.queue(Pick(value=None, by="larger"))
    await say(deps, conv, "why was my healthcare claim denied?", make_form(
        need=Pick(value="denial_question", p=0.4, options={"denial_question": 0.4, "status_inquiry": 0.35}),
        claim_types={"healthcare": 0.9}, claim_statuses={"denied": 0.9}))
    await say(deps, conv, "a person please", make_form(wants_person=True))
    got = await _all(deps, conv)
    assert got
    assert_no_claim_data(json.dumps(got))


async def test_a_typed_full_id_number_is_masked_in_every_event(deps, conv):
    await say(deps, conv, "my ID is 123-45-4472", make_form(**GAVE), stated(id_digits="123-45-4472"))
    await say(deps, conv, "a person please, ID 123456789", make_form(wants_person=True))
    got = await _all(deps, conv)
    text = json.dumps(got)
    assert got and "123-45-4472" not in text and "123456789" not in text
    assert any("***-**-4472" in json.dumps(b) for _, b in got)


async def test_an_email_address_is_never_in_an_event(deps, conv):
    await say(deps, conv, "my email is margaret@email.com", make_form(**GAVE), stated(email="margaret@email.com"))
    got = await _all(deps, conv)
    assert [b["value"] for k, b in got if k == "extraction"] == ["[email]"] and "margaret@email.com" not in json.dumps(got)


async def test_a_long_message_is_cut_and_the_event_count_is_capped(deps, conv):
    deps.settings.limits.event_body_chars = 600
    deps.settings.limits.events_per_turn = 3
    long = "word " * 700
    r = await say(deps, conv, long, make_form(**GAVE),
                  stated(policyholder_name="a", phone="1", email="x@y.zz", id_digits="4472", policy_number="POL-9921"))
    got = await _all(deps, conv)
    assert len(got) == 3 and all(len(json.dumps(b)) <= 600 for _, b in got)
    assert r.inspector["events"][-1] == {"kind": "dropped", "body": {"count": 2}}


def test_a_body_over_the_cap_loses_its_oldest_transcript_rows_last():
    body = {"why": "x", "transcript": [{"message": "m" * 50, "reply": "r" * 50} for _ in range(30)]}
    out = fit(body, 400)
    assert len(json.dumps(out)) <= 400 and out["truncated"] is True


def test_the_recorder_refuses_a_kind_it_does_not_know():
    import uuid
    rec = Recorder(uuid.uuid4(), 1)
    with pytest.raises(ValueError):
        rec.add("nonsense")
    assert "escalation" in EVENT_KINDS
