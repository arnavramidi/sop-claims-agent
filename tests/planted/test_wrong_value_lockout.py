"""Planted fault: the server forgets the wrong values it counted. The database counts them itself, so the caller is
still handed over at the limit and can never be verified afterwards, even with the right details."""

import pytest
from helpers import (GAVE, claims_visible, escalation_reasons, make_deps, make_form, memory, say, stated,
                     verified_party)

from claims_agent.rules import verify_id
from claims_agent.states import State

pytestmark = pytest.mark.planted

WRONG = [dict(email="a@nowhere.example"), dict(phone="2025550101"), dict(id_digits="0000")]
MARGARET = dict(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472")


@pytest.fixture
async def deps(load_dataset):
    return make_deps(await load_dataset("starter"))


async def forget(deps, conversation_id) -> None:
    """The fault: the conversation's memory loses every wrong value counted so far (the server's account may write
    its own conversation's memory, so a bug or a compromised server can do this between any two turns)."""
    async with deps.db.conversation(conversation_id) as c:
        await c.execute("UPDATE conversations SET memory = jsonb_set(memory, '{identity,wrong_values}', '[]') "
                        "WHERE id = %s", [conversation_id])


async def test_memory_cleared_between_turns_still_hands_over_at_the_third_wrong_value(deps, conv):
    for v in WRONG:
        r = await say(deps, conv, "here are my details", make_form(**GAVE), stated(**v))
        if r.state == State.VERIFY_ID:
            await forget(deps, conv)
    assert r.state == State.ESCALATED and await escalation_reasons(deps, conv) == ["wrong_values"]
    assert len(await memory(deps, conv, "identity.wrong_values")) == 3   # the database's count, read back


async def test_a_server_that_never_reads_the_count_still_cannot_verify_after_three_wrong_values(deps, conv, monkeypatch):
    async def nothing_counted(conn, conversation_id):
        return []
    monkeypatch.setattr(verify_id, "wrong_values", nothing_counted)
    for v in WRONG:
        r = await say(deps, conv, "here are my details", make_form(**GAVE), stated(**v))
        await forget(deps, conv)
    assert r.state == State.VERIFY_ID   # the server believes nothing is wrong yet
    r = await say(deps, conv, "Margaret Chen, 1985-03-15, 4472", make_form(**GAVE), stated(**MARGARET))
    assert r.state == State.ESCALATED and await escalation_reasons(deps, conv) == ["wrong_values"]
    assert await verified_party(deps, conv) is None and await claims_visible(deps, conv) == 0
