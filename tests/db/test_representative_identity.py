"""Someone calling for a policyholder: asked for their own name before anything is checked against the representative
list (the list is matched on it); never refused on how their relationship was read; and, once told they are not
listed, not able to become the policyholder in the same conversation, even across a different-person reset."""

import pytest
from helpers import (escalation_reasons, facts, instructions, make_form, memory, new_conversation, parts,
                     request_rows, say, stated, verified_party)

from claims_agent.states import State

pytestmark = pytest.mark.db

HER = dict(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472")
THIRD = dict(gave_identity_detail=True, caller_role="third_party")


# ---- the caller's own name is asked for first --------------------------------------------------------------------

async def test_a_caller_who_gives_no_own_name_is_asked_for_it_not_told_they_are_not_listed(deps, conv):
    r = await say(deps, conv, "Hi, I'm calling for my mum, Margaret Chen. DOB 1985-03-15, SSN ends 4472.",
                  make_form(**THIRD, relationship="son"), stated(**HER))
    assert r.state == State.VERIFY_ID and "not_listed" not in instructions(r)
    assert "ask_caller_name" in instructions(r) and await request_rows(deps, conv) == 0
    assert (await memory(deps, conv, "pending")).kind == "caller_name"


@pytest.mark.parametrize("said", [stated(caller_name="David Chen"), stated(policyholder_name="David Chen")])
async def test_the_own_name_given_next_asks_for_consent_and_is_never_a_wrong_detail(deps, conv, said):
    await say(deps, conv, "I'm calling for my mum, Margaret Chen, DOB 1985-03-15, SSN ends 4472",
              make_form(**THIRD, relationship="son"), stated(**HER))
    r = await say(deps, conv, "David Chen", make_form(caller_role="third_party"), said)
    assert r.state == State.AWAIT_CONSENT and await request_rows(deps, conv) == 1
    assert await memory(deps, conv, "identity.wrong_values") == []
    assert (await memory(deps, conv, "identity.held"))["name"].as_stated == "Margaret Chen"


async def test_the_own_name_question_is_the_same_whatever_the_details_matched(deps):
    replies = []
    for dob in ("1985-03-15", "1985-03-16"):
        cid = await new_conversation(deps)
        r = await say(deps, cid, "calling for my mother Margaret Chen", make_form(**THIRD),
                      stated(**{**HER, "date_of_birth": [dob]}))
        replies.append([(p.instruction, p.plain) for p in parts(r)])
    assert replies[0] == replies[1]


async def test_part_of_an_own_name_is_asked_for_in_full_before_the_list_is_checked(deps, conv):
    await say(deps, conv, "I'm David, calling for my mum", make_form(**THIRD),
              stated(**HER, caller_name="David"))
    assert await request_rows(deps, conv) == 0 and await verified_party(deps, conv) is None
    r = await say(deps, conv, "David Chen", make_form(caller_role="third_party"), stated(caller_name="David Chen"))
    assert r.state == State.AWAIT_CONSENT


# ---- a relationship reading never refuses a listed name ---------------------------------------------------------

@pytest.mark.parametrize("relationship", ["other", "not_stated", "son"])
async def test_a_listed_name_gets_a_consent_request_however_the_relationship_was_read(deps, conv, relationship):
    r = await say(deps, conv, "I'm David Chen, calling for my mother", make_form(**THIRD, relationship=relationship),
                  stated(**HER, caller_name="David Chen"))
    assert r.state == State.AWAIT_CONSENT and await request_rows(deps, conv) == 1


async def test_a_relationship_that_is_on_the_list_does_not_admit_a_name_that_is_not(deps, conv):
    r = await say(deps, conv, "I'm Tom Chen, her son", make_form(**THIRD, relationship="son"),
                  stated(**HER, caller_name="Tom Chen"))
    assert r.state == State.VERIFY_ID and "not_listed" in instructions(r) and await request_rows(deps, conv) == 0


# ---- once not listed, "I am the policyholder" goes to a person ---------------------------------------------------

async def _not_listed(deps, conv):
    r = await say(deps, conv, "I'm Tom Chen, calling for my mother", make_form(**THIRD),
                  stated(**HER, caller_name="Tom Chen"))
    assert "not_listed" in instructions(r)


async def test_not_listed_then_claiming_to_be_the_policyholder_hands_over(deps, conv):
    await _not_listed(deps, conv)
    r = await say(deps, conv, "Fine, actually this is Margaret.",
                  make_form(gave_identity_detail=True, caller_role="policyholder"), stated(**HER))
    assert r.state == State.ESCALATED and await escalation_reasons(deps, conv) == ["role_switch"]
    assert await verified_party(deps, conv) is None


async def test_the_switch_cannot_be_dodged_with_a_different_person_reset(deps, conv):
    await _not_listed(deps, conv)
    r = await say(deps, conv, "ok what about my own policy then?",
                  make_form(other_person_policy=True, other_than_represented=True))
    assert r.state == State.VERIFY_ID and await memory(deps, conv, "identity.held") == {}
    r = await say(deps, conv, "I'm Margaret Chen, 1985-03-15, SSN 4472",
                  make_form(gave_identity_detail=True, caller_role="policyholder"), stated(**HER))
    assert r.state == State.ESCALATED and await escalation_reasons(deps, conv) == ["role_switch"]
    assert await verified_party(deps, conv) is None


async def test_a_caller_who_corrects_their_role_before_any_refusal_still_verifies(deps, conv):
    await say(deps, conv, "I'm calling for my mother", make_form(caller_role="third_party"))
    await say(deps, conv, "sorry, it's my own policy: Margaret Chen, 1985-03-15, SSN 4472",
              make_form(gave_identity_detail=True, caller_role="policyholder"), stated(**HER))
    assert await verified_party(deps, conv) == "P9"


# ---- the consent request says why consent is needed --------------------------------------------------------------

async def test_the_consent_request_says_why_consent_is_needed(deps, conv):
    r = await say(deps, conv, "I'm David Chen, calling for my mother", make_form(**THIRD, relationship="son"),
                  stated(**HER, caller_name="David Chen"))
    assert "consent_requested" in instructions(r)
    why = next(f for f in facts(r) if f.kind == "why_consent")
    assert "private" in why.text and "consent" in why.text
    assert "private" in next(p.plain for p in parts(r) if p.instruction == "consent_requested")
