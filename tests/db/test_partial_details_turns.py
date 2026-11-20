"""Whole turns: a detail given only in part is asked for in full, is never held and never counts as a wrong value,
whatever the fast model thought of the message; a complete detail beside it is kept. A caller whose whole name on file
is one word can still verify with it."""

import pytest
from helpers import GAVE, instructions, make_deps, make_form, memory, new_conversation, say, stated, verified_party

from claims_agent.states import State

PARTS = stated(policyholder_name="Margaret", date_of_birth_as_stated="1985", phone="650")


@pytest.mark.parametrize("reads_as_partial", [0.95, 0.1])
async def test_a_first_name_a_birth_year_and_an_area_code_are_asked_for_in_full(deps, conv, reads_as_partial):
    r = await say(deps, conv, "My first name is Margaret, I was born in 1985, and my phone area code is 650.",
                  make_form(**GAVE, partial_detail=reads_as_partial), PARTS)
    assert await memory(deps, conv, "identity.held") == {}
    assert await memory(deps, conv, "identity.wrong_values") == []
    part = next(p for p in r.inspector["parts"] if p["instruction"] == "ask_full_value")
    assert {f["text"] for f in part["facts"]} == {"your full name", "your date of birth", "your phone number"}
    assert "restate_value" not in instructions(r)


async def test_a_complete_detail_given_with_parts_is_kept(deps, conv):
    await say(deps, conv, "First name Margaret, SSN last four is 4472",
              make_form(**GAVE, partial_detail=0.95), stated(policyholder_name="Margaret", id_digits="4472"))
    assert set(await memory(deps, conv, "identity.held")) == {"id"}


async def test_a_first_name_never_counts_against_the_caller_and_the_full_name_then_verifies(deps, conv):
    r = await say(deps, conv, "Margaret, born 1985-03-15, SSN ends 4472", make_form(**GAVE),
                  stated(policyholder_name="Margaret", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert r.state == State.VERIFY_ID and await memory(deps, conv, "identity.wrong_values") == []
    r = await say(deps, conv, "Margaret Chen", make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    assert await verified_party(deps, conv) == "P9"


async def test_a_wrong_first_name_is_not_a_wrong_value(deps, conv):
    await say(deps, conv, "it's Margarine", make_form(**GAVE), stated(policyholder_name="Margarine"))
    assert await memory(deps, conv, "identity.wrong_values") == []


async def test_a_first_name_after_the_full_name_is_not_asked_for_again(deps, conv):
    await say(deps, conv, "Margaret Chen", make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    r = await say(deps, conv, "yes, Margaret", make_form(**GAVE), stated(policyholder_name="Margaret"))
    assert "ask_full_value" not in instructions(r)
    assert (await memory(deps, conv, "identity.held"))["name"].as_stated == "Margaret Chen"


@pytest.fixture
async def one_word_deps(load_dataset):
    """starter data, except that P7's whole name on file is the one word "Oyelaran"."""
    return make_deps(await load_dataset("one_word_name"))


async def test_a_caller_whose_whole_name_on_file_is_one_word_verifies_with_it(one_word_deps):
    cid = await new_conversation(one_word_deps)
    await say(one_word_deps, cid, "I'm Oyelaran", make_form(**GAVE), stated(policyholder_name="Oyelaran"))
    await say(one_word_deps, cid, "born 1990-08-21, SSN ends 9180", make_form(**GAVE),
              stated(date_of_birth=["1990-08-21"], id_digits="9180"))
    assert await verified_party(one_word_deps, cid) is None  # the one word was asked for in full, not compared
    await say(one_word_deps, cid, "Oyelaran is my full name", make_form(**GAVE), stated(policyholder_name="Oyelaran"))
    assert await verified_party(one_word_deps, cid) == "P7"


async def test_the_reply_to_a_one_word_name_is_the_same_whether_or_not_it_is_a_whole_name_on_file(one_word_deps):
    replies = []
    for name in ("Oyelaran", "Okonkwo"):
        cid = await new_conversation(one_word_deps)
        r = await say(one_word_deps, cid, f"I'm {name}, born 1990-08-21", make_form(**GAVE),
                      stated(policyholder_name=name, date_of_birth=["1990-08-21"]))
        replies.append([(p["instruction"], [f["text"] for f in p["facts"]]) for p in r.inspector["parts"]])
    assert replies[0] == replies[1]
