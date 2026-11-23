"""A caller who would rather not give a detail: thanked, never asked for that detail again in the conversation, told
which other details would do and that nothing about a claim can be shared without three that match, with a person
offered once three can no longer be reached. Refusals are kept in identity memory as kinds only."""

import pytest
from helpers import GAVE, instructions, make_form, memory, parts, say, stated, verified_party

from claims_agent.form.fields import YesNo
from claims_agent.rules.verify_id import LABEL
from claims_agent.states import State

pytestmark = pytest.mark.db


def _refuses(*kinds, declines=True, **more):
    return make_form(declines=declines, gave_identity_detail=False, caller_role="policyholder",
                     refuses={k: 1.0 for k in kinds}, **more)


async def test_a_refused_detail_is_acknowledged_and_not_offered_again(deps, conv):
    r = await say(deps, conv, "I'd rather not give my phone number", _refuses("phone"))
    assert r.state == State.VERIFY_ID and "acknowledge_refusal" in instructions(r)
    part = next(p for p in parts(r) if p.instruction == "acknowledge_refusal")
    kinds = {f.kind: f.text for f in part.facts}
    assert LABEL["phone"] in kinds["not_asked_again"]
    assert LABEL["phone"] not in kinds["other_details"]
    assert all(LABEL[k] in kinds["other_details"] for k in ("name", "dob", "email", "id"))
    assert "nothing about a claim can be shared" in kinds["without_three"]
    assert "person" in kinds["person_instead"]  # three can still be reached: a person is mentioned, not asked about
    assert (await memory(deps, conv, "pending")) is None
    assert await memory(deps, conv, "identity.refused") == ["phone"]


async def test_a_refused_detail_is_never_asked_for_again_in_the_conversation(deps, conv):
    await say(deps, conv, "I'm not giving you my social security number", _refuses("id"))
    r = await say(deps, conv, "Margaret Chen", make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    ask = next(p for p in parts(r) if p.instruction == "ask_another_detail")
    # the rule still names all five kinds that count; what is asked for leaves the refused one out
    assert LABEL["id"] not in ask.plain
    assert all(LABEL["id"] not in f.text for f in ask.facts if f.kind != "verification_rule")
    assert LABEL["phone"] in ask.plain


async def test_the_caller_can_still_verify_with_the_other_details(deps, conv):
    await say(deps, conv, "No SSN, sorry", _refuses("id"))
    await say(deps, conv, "Margaret Chen, born 1985-03-15, phone 650-521-2836", make_form(**GAVE),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], phone="650-521-2836"))
    assert await verified_party(deps, conv) == "P9"


async def test_a_person_is_offered_once_three_details_can_no_longer_be_reached(deps, conv):
    r = await say(deps, conv, "You're not getting my phone, email or SSN", _refuses("phone", "email", "id"))
    assert "offer_person" in instructions(r) and (await memory(deps, conv, "pending")).kind == "offer_person"
    assert sorted(await memory(deps, conv, "identity.refused")) == ["email", "id", "phone"]
    r = await say(deps, conv, "yes please", make_form(pending_reply="yes"))
    assert r.state == State.ESCALATED


async def test_a_refusal_naming_no_kind_still_says_what_would_do_and_why(deps, conv):
    r = await say(deps, conv, "I don't want to give you any of that", _refuses())
    part = next(p for p in parts(r) if p.instruction == "acknowledge_refusal")
    assert {f.kind for f in part.facts} >= {"other_details", "without_three", "verification_rule"}
    assert await memory(deps, conv, "identity.refused") == []


async def test_a_reading_below_the_yes_line_is_no_refusal(deps, conv):
    form = make_form(gave_identity_detail=False, caller_role="policyholder", refuses={"phone": YesNo(p=0.3)})
    r = await say(deps, conv, "hmm, phone?", form)
    assert "acknowledge_refusal" not in instructions(r) and await memory(deps, conv, "identity.refused") == []


async def test_refusals_hold_kinds_only(deps, conv):
    await say(deps, conv, "my email is none of your business", _refuses("email"))
    m = await memory(deps, conv)
    assert m.identity.refused == ["email"] and m.identity.held == {}


async def test_a_different_person_starts_with_no_refusals(deps, conv):
    await say(deps, conv, "no phone number from me", _refuses("phone"))
    await say(deps, conv, "what about my wife's policy?", make_form(other_person_policy=True))
    assert await memory(deps, conv, "identity.refused") == []


async def test_the_first_ask_after_a_refusal_leaves_the_refused_kind_out(deps, conv):
    await say(deps, conv, "I won't give my email", _refuses("email"))
    r = await say(deps, conv, "ok so what now?", make_form(caller_role="policyholder"))
    ask = next(p for p in parts(r) if p.instruction == "ask_identity")
    assert LABEL["email"] not in ask.plain and LABEL["phone"] in ask.plain


async def test_a_refusal_already_acknowledged_is_not_acknowledged_again_when_a_detail_follows(deps, conv):
    await say(deps, conv, "Margaret Chen", make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    await say(deps, conv, "I'd rather not give my phone number", _refuses("phone"))
    # the next message gives a detail; the refusal question still reads yes for the phone, carried over from before
    r = await say(deps, conv, "I was born on 1985-03-15", make_form(**GAVE, refuses={"phone": 1.0}),
                  stated(date_of_birth=["1985-03-15"]))
    assert "acknowledge_refusal" not in instructions(r)
    ask = next(p for p in parts(r) if p.instruction == "ask_another_detail")
    assert LABEL["phone"] not in ask.plain
    assert await memory(deps, conv, "identity.refused") == ["phone"]
