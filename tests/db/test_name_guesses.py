"""A name is either asked for again and never sent to the database, or sent and counted when it does not match: no
value is compared without being counted. A one-word name is part of a name the first time; given again when the full
name was asked for, it is taken as the whole name (a customer whose name on file is one word), compared, and counted
when wrong."""

import pytest
from helpers import (BRIEF_MSG, BRIEF_SAID, GAVE, escalation_reasons, instructions, make_deps, make_form, memory,
                     new_conversation, say, stated, verified_party)

from claims_agent.rules import verify_id
from claims_agent.states import State

RIGHT_TWO = dict(date_of_birth=["1985-03-15"], id_digits="4472")


@pytest.fixture
def sent(monkeypatch):
    """Every identity the VERIFY_ID step sends to the database for comparison, in order."""
    seen = []
    real = verify_id.match_identity

    async def recording(conn, conversation_id, s):
        seen.append(s)
        return await real(conn, conversation_id, s)
    monkeypatch.setattr(verify_id, "match_identity", recording)
    return seen


@pytest.mark.parametrize("token", ["MargaretChen", "margaretchen"])
async def test_a_full_name_typed_as_one_token_does_not_verify_and_is_counted_once_compared(deps, conv, sent, token):
    r = await say(deps, conv, f"{token}, 1985-03-15, 4472", make_form(**GAVE), stated(policyholder_name=token, **RIGHT_TWO))
    assert await verified_party(deps, conv) is None and "ask_full_value" in instructions(r)
    assert sent[-1].name is None  # asked for in full, never compared
    r = await say(deps, conv, f"{token} is my full name", make_form(**GAVE), stated(policyholder_name=token))
    assert await verified_party(deps, conv) is None and sent[-1].name == token
    assert len(await memory(deps, conv, "identity.wrong_values")) == 1  # compared, so counted


@pytest.mark.parametrize("token", ["Margaret-Chen", "Margaret.Chen"])
async def test_a_hyphen_or_a_dot_between_the_words_is_the_whole_name(deps, conv, token):
    await say(deps, conv, f"{token}, 1985-03-15, 4472", make_form(**GAVE), stated(policyholder_name=token, **RIGHT_TWO))
    assert await verified_party(deps, conv) == "P9"


async def test_a_wrong_hyphenated_name_is_counted(deps, conv):
    await say(deps, conv, "Mary-Smith, 1985-03-15", make_form(**GAVE),
              stated(policyholder_name="Mary-Smith", date_of_birth=["1985-03-15"]))
    assert len(await memory(deps, conv, "identity.wrong_values")) == 1


async def test_three_wrong_one_token_names_hand_over_like_any_other_wrong_values(deps, conv):
    await say(deps, conv, "1985-03-15, 4472", make_form(**GAVE), stated(**RIGHT_TWO))
    r = None
    for token in ("Smithson", "Jonesy", "Brownlow"):
        await say(deps, conv, token, make_form(**GAVE), stated(policyholder_name=token))  # asked for in full
        r = await say(deps, conv, f"{token}, that is my full name", make_form(**GAVE), stated(policyholder_name=token))
    assert r.state == State.ESCALATED and await escalation_reasons(deps, conv) == ["wrong_values"]


async def test_a_first_name_alone_is_asked_for_in_full_and_never_compared(deps, conv, sent):
    r = await say(deps, conv, "My first name is Margaret", make_form(**GAVE), stated(policyholder_name="Margaret"))
    assert "ask_full_value" in instructions(r) and all(s.name is None for s in sent)
    assert await memory(deps, conv, "identity.wrong_values") == []


@pytest.fixture
async def one_word_deps(load_dataset):
    """starter data, except that P7's whole name on file is the one word "Oyelaran"."""
    return make_deps(await load_dataset("one_word_name"))


async def test_a_one_word_name_on_file_verifies_when_given_again_as_the_whole_name(one_word_deps):
    cid = await new_conversation(one_word_deps)
    await say(one_word_deps, cid, "Oyelaran, born 1990-08-21, SSN ends 9180", make_form(**GAVE),
              stated(policyholder_name="Oyelaran", date_of_birth=["1990-08-21"], id_digits="9180"))
    assert await verified_party(one_word_deps, cid) is None
    await say(one_word_deps, cid, "Oyelaran is my whole name", make_form(**GAVE), stated(policyholder_name="Oyelaran"))
    assert await verified_party(one_word_deps, cid) == "P7"


async def test_a_wrong_one_word_guess_against_a_one_word_name_on_file_is_counted(one_word_deps):
    cid = await new_conversation(one_word_deps)
    await say(one_word_deps, cid, "Okonkwo, born 1990-08-21", make_form(**GAVE),
              stated(policyholder_name="Okonkwo", date_of_birth=["1990-08-21"]))
    await say(one_word_deps, cid, "Okonkwo, that's all of it", make_form(**GAVE), stated(policyholder_name="Okonkwo"))
    assert len(await memory(one_word_deps, cid, "identity.wrong_values")) == 1


async def test_the_assignments_message_still_verifies_in_one_turn(deps, conv):
    r = await say(deps, conv, BRIEF_MSG, make_form(**GAVE), BRIEF_SAID)
    assert r.state != State.VERIFY_ID and await verified_party(deps, conv) == "P9"


# ---- the invariant, kind by kind: asked again and never sent, or sent and counted when wrong ---------------------

PARTS = {"name": dict(policyholder_name="Margaret"), "dob": dict(date_of_birth_as_stated="1985"),
         "phone": dict(phone="650"), "email": dict(email="margaret at email"), "id": dict(id_digits="44")}
WRONG = {"name": dict(policyholder_name="Mary Smith"), "dob": dict(date_of_birth=["1970-01-01"]),
         "phone": dict(phone="+1 212 555 0100"), "email": dict(email="nobody@example.com"), "id": dict(id_digits="0000")}
FIELD = {"name": "name", "dob": "dobs", "phone": "phone", "email": "email", "id": "id_last4"}


@pytest.mark.parametrize("kind", sorted(PARTS))
async def test_a_part_or_malformed_value_is_asked_again_and_never_sent(deps, conv, sent, kind):
    await say(deps, conv, "here you go", make_form(**GAVE), stated(**PARTS[kind]))
    assert all(getattr(s, FIELD[kind]) in ((), None) for s in sent)
    assert kind not in await memory(deps, conv, "identity.held")
    assert await memory(deps, conv, "identity.wrong_values") == []


@pytest.mark.parametrize("kind", sorted(WRONG))
async def test_a_whole_value_is_sent_and_counted_when_wrong(deps, conv, sent, kind):
    other = {"name": dict(policyholder_name="Margaret Chen")} if kind != "name" else {"dob": dict(date_of_birth=["1985-03-15"])}
    await say(deps, conv, "here you go", make_form(**GAVE), stated(**WRONG[kind], **next(iter(other.values()))))
    assert getattr(sent[-1], FIELD[kind]) not in ((), None)
    assert len(await memory(deps, conv, "identity.wrong_values")) == 1
