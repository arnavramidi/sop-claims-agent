"""Identity details given as bare values, details that could not be read, and replies that ask for details, over real
Postgres. The first conversation is the one a caller had on the running app: three correct details, the third a bare
phone number that came back from the extractor with every field empty, and no verification."""

import pytest
from helpers import GAVE, facts, instructions, make_form, memory, new_conversation, say, stated, verified_party

from claims_agent.db.identity import match_identity
from claims_agent.identity.normalize import normalise
from claims_agent.rules.verify_id import ID_LABEL, MENTION, THE_RULE

TURN_1 = "hi, my name is Margaret Chen, and I want information regarding my denied policy"
TURN_2 = "my date of birth is march 15, 1985"
TURN_3 = "16505212836"


def _part(deps, instruction):
    """The part as the wording model was handed it (the turn record keeps only its instruction, facts and text)."""
    return next(p for p in deps.wording.briefs[-1].parts if p.instruction == instruction)


async def _name_and_dob(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, TURN_1, make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    r = await say(deps, cid, TURN_2, make_form(gave_identity_detail=True),
                  stated(date_of_birth_as_stated="march 15, 1985", date_of_birth=["1985-03-15"]))
    assert set(await memory(deps, cid, "identity.held")) == {"name", "dob"}
    return cid, r


async def test_the_extractor_is_given_the_previous_reply_and_what_was_asked(deps):
    cid, before = await _name_and_dob(deps)
    r = await say(deps, cid, TURN_3, make_form(gave_identity_detail=True), stated(phone="16505212836"))
    asked = deps.extractor.requests[-1]
    assert asked.message == TURN_3 and asked.previous_reply == before.reply
    assert {"phone number", "email address", ID_LABEL} <= set(asked.asked_for)
    assert "full name" not in asked.asked_for and "date of birth" not in asked.asked_for
    assert await verified_party(deps, cid) == "P9" and "verified" in instructions(r)


async def test_a_detail_the_form_saw_but_the_extractor_did_not_is_said_to_be_unread_and_not_thanked_for(deps):
    # The app's turn 3: the form read a detail (the turn was flagged cross_check), the extractor returned nothing,
    # held stayed {name, dob}, and the reply thanked the caller and asked for the email address.
    cid, _ = await _name_and_dob(deps)
    r = await say(deps, cid, TURN_3, make_form(gave_identity_detail=True), stated())
    assert "cross_check" in r.inspector["flags"]
    assert set(await memory(deps, cid, "identity.held")) == {"name", "dob"}
    assert instructions(r)[0] == "unreadable_detail"
    ask = _part(deps, "ask_another_detail")
    assert ask.must_not_mention and "Thank" not in r.reply


async def test_a_value_the_extractor_found_but_the_form_did_not_is_asked_again_by_its_kind(deps):
    cid, _ = await _name_and_dob(deps)
    r = await say(deps, cid, "six five oh", make_form(gave_identity_detail=False), stated(phone="650"))
    assert instructions(r)[0] == "restate_value"
    assert [f.text for f in facts(r) if f.kind == "detail_to_restate"] == ["your phone number"]
    assert "650" not in r.reply and "Thank" not in r.reply


async def test_every_reply_that_asks_for_details_names_every_open_kind_and_the_rule(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, "Hi, I need help with a claim.", make_form())
    ask = _part(deps, "ask_identity")
    assert ask.must_mention == (THE_RULE, *(MENTION[k] for k in ("name", "dob", "phone", "email", "id")))
    r = await say(deps, cid, TURN_1, make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    ask = _part(deps, "ask_another_detail")
    assert ask.must_mention == (THE_RULE, *(MENTION[k] for k in ("dob", "phone", "email", "id")))
    for label in ("date of birth", "phone number", "email address", ID_LABEL):
        assert label in r.reply
    assert r.reply.startswith("Thank you.")


async def test_two_bare_values_in_one_message_are_both_kept(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "Hi, I'm Margaret Chen.", make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    r = await say(deps, cid, "03/15/1985, 650-521-2836", make_form(gave_identity_detail=True),
                  stated(date_of_birth_as_stated="03/15/1985", date_of_birth=["1985-03-15"], phone="650-521-2836"))
    assert await verified_party(deps, cid) == "P9" and "verified" in instructions(r)


@pytest.mark.parametrize("phone", ["16505212836", "6505212836", "650-521-2836", "(650) 521 2836", "1 650 521 2836",
                                   "+1 650-521-2836", "+16505212836"])
async def test_every_way_of_writing_the_phone_compares_equal_in_the_database(deps, phone):
    s, failures = normalise(stated(phone=phone))
    assert failures == []
    cid = await new_conversation(deps)  # the database compares only for a conversation, and counts what is wrong
    async with deps.db.conversation(cid) as c:
        matches = {m.party_id: m.kinds for m in await match_identity(c, cid, s)}
    assert "phone" in matches.get("P9", set())


async def test_a_worded_reply_that_asks_for_one_detail_is_replaced_by_one_that_lists_every_option(deps):
    # The app's turn-2 reply asked for the phone number alone.
    cid = await new_conversation(deps)
    await say(deps, cid, TURN_1, make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    one_at_a_time = "Thanks for that. Could you please provide your phone number to help verify your identity?"
    deps.wording.queue([one_at_a_time, one_at_a_time])
    r = await say(deps, cid, TURN_2, make_form(gave_identity_detail=True),
                  stated(date_of_birth_as_stated="march 15, 1985", date_of_birth=["1985-03-15"]))
    assert r.inspector["reply_source"] == "template"
    assert all("required_mention_missing" in c["failures"] for c in r.inspector["checks"][:2])
    for label in ("phone number", "email address", ID_LABEL):
        assert label in r.reply
    assert "3 details must match" in r.reply
