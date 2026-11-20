"""Whole turns through AWAIT_CONSENT: a listed representative waits for the policyholder's consent."""

import pytest
import json
from dataclasses import asdict

from helpers import (assert_no_claim_data, checks_used, claims_visible, escalation_reasons, facts, instructions, make_deps, make_form,
                     memory, new_conversation, say, stated, request_rows, verified_party)

from claims_agent.states import State

REP = stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472", caller_name="David Chen")
THIRD = dict(gave_identity_detail=True, caller_role="third_party", relationship="son")
RECHECK = make_form(consent_recheck=True, caller_role="third_party")


async def _waiting(deps, script):
    cid = await new_conversation(deps, consent_script=script)
    r = await say(deps, cid, "I'm David, calling for my mother", make_form(**THIRD), REP)
    assert r.state == State.AWAIT_CONSENT
    return cid


@pytest.fixture
async def waiting_conv(deps):
    return await _waiting(deps, "default")


@pytest.fixture
async def waiting_conv_timeout(deps):
    return await _waiting(deps, "timeout")


@pytest.fixture
async def extra_deps(load_dataset):
    return make_deps(await load_dataset("extra_consent"))


@pytest.fixture
async def waiting_conv_denied(extra_deps):
    return await _waiting(extra_deps, "denied")


@pytest.fixture
async def waiting_conv_unknown(extra_deps):
    return await _waiting(extra_deps, "unknown")


async def test_c1_a_listed_representative_waits_for_consent(deps, conv):
    r = await say(deps, conv, "I'm David, calling for my mother", make_form(**THIRD), REP)
    assert r.state == State.AWAIT_CONSENT and await claims_visible(deps, conv) == 0
    assert "consent_requested" in instructions(r)


async def test_c2_first_check_pending_second_approves(deps, waiting_conv):
    assert await checks_used(deps, waiting_conv) == 1 and await claims_visible(deps, waiting_conv) == 0
    r = await say(deps, waiting_conv, "can you check again?", RECHECK)
    assert r.state not in (State.AWAIT_CONSENT, State.ESCALATED, State.CLOSED, State.VERIFY_ID)
    assert await verified_party(deps, waiting_conv) == "P9"
    assert "verified" in instructions(r) and await checks_used(deps, waiting_conv) == 2


async def test_c3_five_pending_checks_then_handover(deps, waiting_conv_timeout):
    for _ in range(3):
        assert (await say(deps, waiting_conv_timeout, "check again", RECHECK)).state == State.AWAIT_CONSENT
    r = await say(deps, waiting_conv_timeout, "check again", RECHECK)
    assert r.state == State.ESCALATED and await escalation_reasons(deps, waiting_conv_timeout) == ["consent_timeout"]


async def test_c4_denied_closes_with_no_handover(extra_deps, waiting_conv_denied):
    r = await say(extra_deps, waiting_conv_denied, "check again", RECHECK)
    assert r.state == State.CLOSED and await escalation_reasons(extra_deps, waiting_conv_denied) == []
    assert "consent_denied" in instructions(r)
    assert (await say(extra_deps, waiting_conv_denied, "hello?")).inspector["reply_source"] == "fixed"


async def test_c5_waiting_uses_no_check_but_asking_does(deps, waiting_conv):
    r = await say(deps, waiting_conv, "hold on, I'm texting her", make_form(caller_role="third_party"))
    assert await checks_used(deps, waiting_conv) == 1 and "consent_pending" in instructions(r)
    for p in (False, 0.5):  # a present but not confident answer is no request
        await say(deps, waiting_conv, "not sure yet", make_form(consent_recheck=p, caller_role="third_party"))
        assert await checks_used(deps, waiting_conv) == 1
    await say(deps, waiting_conv, "she approved, check again", RECHECK)
    assert await checks_used(deps, waiting_conv) == 2


async def test_c6_an_unlisted_relative_is_blocked_with_no_request(deps, conv):
    r = await say(deps, conv, "I'm Tom, calling for my mother", make_form(**THIRD),
                  REP.model_copy(update={"caller_name": "Tom Chen"}))
    assert r.state == State.VERIFY_ID and "not_listed" in instructions(r) and await request_rows(deps, conv) == 0


async def test_c7_claiming_to_be_the_policyholder_now_hands_over(deps, waiting_conv):
    r = await say(deps, waiting_conv, "actually this is Margaret typing now", make_form(caller_role="policyholder"))
    assert r.state == State.ESCALATED and await escalation_reasons(deps, waiting_conv) == ["role_switch"]
    assert await checks_used(deps, waiting_conv) == 1


async def test_c8_an_unknown_consent_value_hands_over(extra_deps, waiting_conv_unknown):
    r = await say(extra_deps, waiting_conv_unknown, "check again", RECHECK)
    assert r.state == State.ESCALATED and await escalation_reasons(extra_deps, waiting_conv_unknown) == ["consent_unknown"]


async def test_c9_a_contradicting_relationship_is_not_listed(deps, conv):
    r = await say(deps, conv, "I'm her husband David", make_form(**{**THIRD, "relationship": "other"}), REP)
    assert r.state == State.VERIFY_ID


async def test_c10_a_claim_question_while_waiting_is_kept_for_later(deps, waiting_conv):
    r = await say(deps, waiting_conv, "why was her healthcare claim denied?",
                  make_form(caller_role="third_party", need="denial_question", claim_types={"healthcare": 0.95}))
    assert r.state == State.AWAIT_CONSENT and not any(f.claim for f in facts(r))
    assert instructions(r) == ["consent_pending"] and await checks_used(deps, waiting_conv) == 1
    assert await memory(deps, waiting_conv, "hints.need") == "denial_question"


async def test_c12_no_claim_rows_while_consent_is_pending(deps, waiting_conv):
    assert await claims_visible(deps, waiting_conv) == 0


async def test_the_status_is_a_consent_fact(deps, waiting_conv):
    r = await say(deps, waiting_conv, "hold on", make_form(caller_role="third_party"))
    assert any(f.kind == "consent" for f in facts(r))
    assert (await memory(deps, waiting_conv)).consent.last_status == "pending"


def _dump(x):
    return json.dumps(asdict(x), default=str)


async def test_f4_a_waiting_turn_sends_no_claim_data_anywhere(deps, waiting_conv):
    r = await say(deps, waiting_conv, "why was her healthcare claim denied?",
                  make_form(caller_role="third_party", need="denial_question", claim_types={"healthcare": 0.95}))
    assert_no_claim_data(r.reply)
    for b in deps.wording.briefs:
        assert_no_claim_data(_dump(b))
    for q in deps.filler.requests:
        assert_no_claim_data(_dump(q))


async def test_f4_the_approval_turns_requests_carry_no_claim_data(deps, waiting_conv):
    await say(deps, waiting_conv, "can you check again?", RECHECK)
    assert await verified_party(deps, waiting_conv) == "P9"
    for q in deps.filler.requests:
        assert_no_claim_data(_dump(q))


@pytest.mark.parametrize("value", ["approved", "no_request", "no_script"])
async def test_f6_an_approval_the_database_did_not_bind_hands_over(deps, waiting_conv, monkeypatch, value):
    async def fake(conn, cid):  # says approved, binds nobody
        return value
    monkeypatch.setattr("claims_agent.rules.await_consent.check_consent", fake)
    r = await say(deps, waiting_conv, "she approved, check again", RECHECK)
    assert r.state == State.ESCALATED and await escalation_reasons(deps, waiting_conv) == ["consent_unknown"]
    assert await verified_party(deps, waiting_conv) is None


async def test_f1_the_check_limit_survives_a_different_person_reset(deps, waiting_conv_timeout):
    for _ in range(3):
        assert (await say(deps, waiting_conv_timeout, "check again", RECHECK)).state == State.AWAIT_CONSENT
    assert await checks_used(deps, waiting_conv_timeout) == 4
    r = await say(deps, waiting_conv_timeout, "what about my dad's policy?", make_form(other_person_policy=True))
    assert r.state == State.VERIFY_ID
    r = await say(deps, waiting_conv_timeout, "here are the details again", make_form(**THIRD), REP)
    assert r.state == State.ESCALATED
    assert await escalation_reasons(deps, waiting_conv_timeout) == ["consent_timeout"]
