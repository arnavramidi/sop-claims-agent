"""Whole turns before verification: the models see what the agent just asked; a typed ID number or email is kept only
as its comparison key and a masked form; an upset caller who gives up is offered a person once; the ID is named so a
caller can act on it."""

import json
from dataclasses import asdict

import pytest
from helpers import (GAVE, assert_no_claim_data, escalation_reasons, instructions, make_form, memory, parts, say,
                     stated, verified_party)

from claims_agent.form.fields import Pick
from claims_agent.rules.verify_id import ID_LABEL
from claims_agent.states import State

pytestmark = pytest.mark.db

UPSET = Pick(value="upset", p=0.9, options={"calm": 0.0, "uneasy": 0.05, "upset": 0.9, "angry": 0.05})
REP = stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472", caller_name="David Chen")


# ---- the previous reply ------------------------------------------------------------------------------------------

async def test_during_verification_both_models_see_the_previous_reply(deps, conv):
    first = await say(deps, conv, "hello, I need help with my claim", make_form(caller_role="policyholder"))
    await say(deps, conv, "why do you need that?", make_form(asks_why_step=True))
    assert deps.filler.requests[-1].previous_reply == first.reply
    assert deps.wording.briefs[-1].previous_reply == first.reply


async def test_while_waiting_for_consent_both_models_see_the_previous_reply(deps, conv):
    first = await say(deps, conv, "I'm David, calling for my mother",
                      make_form(gave_identity_detail=True, caller_role="third_party", relationship="son"), REP)
    assert first.state == State.AWAIT_CONSENT
    await say(deps, conv, "what does that mean?", make_form(caller_role="third_party"))
    assert deps.filler.requests[-1].previous_reply == first.reply
    assert deps.wording.briefs[-1].previous_reply == first.reply


async def test_after_a_different_person_reset_no_earlier_claim_reply_reaches_a_model(deps, verified_conv):
    await say(deps, verified_conv, "what about CL-2048?", make_form(quoted_claim_number=True),
              stated(claim_number="CL-2048"))
    reset = await say(deps, verified_conv, "what about my husband's policy?", make_form(other_person_policy=True))
    assert reset.state == State.VERIFY_ID and deps.wording.briefs[-1].previous_reply is None
    await say(deps, verified_conv, "his name is Robert")
    for given in (deps.filler.requests[-1].previous_reply, deps.wording.briefs[-1].previous_reply):
        assert given == reset.reply  # the reset turn's own reply, made after the reset
        assert_no_claim_data(given)


# ---- what is kept of a typed ID number and email -----------------------------------------------------------------

async def test_a_typed_full_id_number_and_email_are_kept_masked_and_still_compared(deps, conv):
    await say(deps, conv, "my email is Margaret@Email.com and my SSN is 123-45-4472", make_form(**GAVE),
              stated(email="Margaret@Email.com", id_digits="123-45-4472"))
    held = await memory(deps, conv, "identity.held")
    assert (held["email"].as_stated, held["email"].key) == ("m***@email.com", "margaret@email.com")
    assert (held["id"].as_stated, held["id"].key) == ("***-**-4472", "4472")
    text = json.dumps((await memory(deps, conv)).model_dump(mode="json"))
    assert "123-45" not in text and "123454472" not in text and "Margaret@Email.com" not in text
    await say(deps, conv, "born 1985-03-15", make_form(**GAVE), stated(date_of_birth=["1985-03-15"]))
    assert await verified_party(deps, conv) == "P9"


async def test_a_hand_over_note_masks_emails_as_well_as_id_numbers(deps, conv):
    r = await say(deps, conv, "get me a person, I'm margaret@email.com, SSN 123454472", make_form(wants_person=True))
    asked = r.inspector["handover_note"]["asked"]
    assert "margaret@email.com" not in asked and "m***@email.com" in asked
    assert "123454472" not in asked and "*****4472" in asked


# ---- an upset caller who gives up before verification -----------------------------------------------------------

async def test_an_upset_unverified_caller_who_is_done_is_offered_a_person_once_then_closed(deps, conv):
    r = await say(deps, conv, "Forget it, I'm done with this.", make_form(done=True, mood=UPSET))
    assert r.state == State.VERIFY_ID and instructions(r) == ["offer_person_before_closing"]
    assert (await memory(deps, conv, "pending")).kind == "offer_person"
    r = await say(deps, conv, "No. Bye.", make_form(done=True, mood=UPSET, pending_reply="no"))
    assert r.state == State.CLOSED and await escalation_reasons(deps, conv) == []


async def test_the_offer_before_closing_hands_over_on_a_yes(deps, conv):
    await say(deps, conv, "I give up.", make_form(done=True, mood=UPSET))
    r = await say(deps, conv, "yes, fine", make_form(pending_reply="yes"))
    assert r.state == State.ESCALATED and await escalation_reasons(deps, conv) == ["asked_for_person"]


async def test_a_calm_unverified_caller_who_is_done_is_closed_at_once(deps, conv):
    r = await say(deps, conv, "never mind, bye", make_form(done=True))
    assert r.state == State.CLOSED


async def test_an_upset_representative_who_gives_up_while_waiting_is_offered_a_person(deps, conv):
    await say(deps, conv, "I'm David, calling for my mother",
              make_form(gave_identity_detail=True, caller_role="third_party", relationship="son"), REP)
    r = await say(deps, conv, "This is useless, I'm done.", make_form(done=True, mood=UPSET, caller_role="third_party"))
    assert r.state == State.AWAIT_CONSENT and instructions(r) == ["offer_person_before_closing"]


# ---- how the ID is named -----------------------------------------------------------------------------------------

async def test_the_id_is_named_as_the_data_holds_it_never_as_one_customers_type(deps, conv):
    r = await say(deps, conv, "hello", make_form(caller_role="policyholder"))
    ask = next(p for p in parts(r) if p.instruction == "ask_identity")
    assert ID_LABEL in ask.plain and all("ID on file" not in f.text for f in ask.facts)
    assert any(ID_LABEL in f.text for f in ask.facts)
    briefs = json.dumps([asdict(b) for b in deps.wording.briefs], default=str)
    assert "SSN or national ID" in briefs
