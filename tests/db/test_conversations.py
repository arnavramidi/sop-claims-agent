import asyncio
import pytest
from datetime import date

from helpers import verified_party

from claims_agent.db.conversations import (
    add_escalation, append_turn, create, load_for_turn, save, set_controls, transcript,
)
from claims_agent.memory import Memory


async def new_conversation(db, **kw):
    async with db.app.conversation(None) as c:
        return await create(c, **kw)


async def load(db, cid):
    async with db.app.conversation(cid) as c:
        return await load_for_turn(c, cid)


async def save_with(db, cid, *, need):
    async with db.app.conversation(cid) as c:
        conv, mem = await load_for_turn(c, cid)
        mem.hints.need = need
        await save(c, conv, mem)


async def delayed(seconds, coro):
    await asyncio.sleep(seconds)
    return await coro


async def test_a_new_conversation_starts_in_verify_id_with_empty_memory(test_db):
    cid = await new_conversation(test_db)
    conv, mem = await load(test_db, cid)
    assert conv.state.value == "VERIFY_ID" and conv.consent_script == "default" and conv.today is None
    assert conv.verified_party_id is None and conv.pending_party_id is None
    assert mem == Memory.fresh()


async def test_memory_round_trips(test_db):
    cid = await new_conversation(test_db)
    await save_with(test_db, cid, need="status_inquiry")
    assert (await load(test_db, cid))[1].hints.need == "status_inquiry"


async def test_save_cannot_change_who_is_verified(test_db):
    cid = await new_conversation(test_db)
    async with test_db.app.conversation(cid) as c:
        conv, mem = await load_for_turn(c, cid)
        conv.verified_party_id = "P9"
        conv.pending_party_id = "P9"
        await save(c, conv, mem)
    assert await verified_party(test_db, cid) is None
    assert (await load(test_db, cid))[0].pending_party_id is None


async def test_save_writes_the_state(test_db):
    from claims_agent.states import State
    cid = await new_conversation(test_db)
    async with test_db.app.conversation(cid) as c:
        conv, mem = await load_for_turn(c, cid)
        conv.state = State.AWAIT_CONSENT
        await save(c, conv, mem)
    assert (await load(test_db, cid))[0].state == State.AWAIT_CONSENT


async def test_two_messages_at_once_are_handled_one_after_the_other(test_db):
    cid = await new_conversation(test_db)
    order = []

    async def turn(tag, hold):
        async with test_db.app.conversation(cid) as c:
            conv, mem = await load_for_turn(c, cid)
            order.append(f"{tag}:saw {mem.counters.turn_no}")
            await asyncio.sleep(hold)
            mem.counters.turn_no += 1
            await save(c, conv, mem)

    await asyncio.gather(turn("first", 0.3), delayed(0.05, turn("second", 0)))
    assert order == ["first:saw 0", "second:saw 1"]


async def test_transcript_returns_turns_in_order(test_db):
    cid = await new_conversation(test_db)
    async with test_db.app.conversation(cid) as c:
        await append_turn(c, cid, 2, "second q", "second a", "PROCESS_CASE", "POST_PROCESS", {"n": 2})
        await append_turn(c, cid, 1, "first q", "first a", "VERIFY_ID", "VERIFY_ID", {"n": 1})
        rows = await transcript(c, cid)
    assert [r["turn_no"] for r in rows] == [1, 2]
    assert [r["reply"] for r in rows] == ["first a", "second a"]
    assert rows[0]["message"] == "first q" and rows[0]["state_after"] == "VERIFY_ID" and rows[1]["record"] == {"n": 2}


async def test_controls_and_escalations(test_db):
    cid = await new_conversation(test_db, consent_script="strict", today=date(2026, 4, 1))
    assert (await load(test_db, cid))[0].today == date(2026, 4, 1)
    async with test_db.app.conversation(cid) as c:
        await set_controls(c, cid, today=date(2026, 5, 1))
        await add_escalation(c, cid, 3, "handover", "three wrong values", {"count": 3})
    conv, _ = await load(test_db, cid)
    assert conv.consent_script == "strict" and conv.today == date(2026, 5, 1)
    async with test_db.app.conversation(cid) as c:
        row = await (await c.execute("SELECT turn_no, kind, reason, note FROM escalations WHERE conversation_id = %s", [cid])).fetchone()
    assert row == (3, "handover", "three wrong values", {"count": 3})


async def test_the_store_only_serves_the_conversation_the_transaction_is_bound_to(test_db):
    a, b = await new_conversation(test_db), await new_conversation(test_db)
    async with test_db.app.conversation(a) as c:
        with pytest.raises(LookupError):
            await load_for_turn(c, b)
    async with test_db.app.conversation(a) as c:
        conv, mem = await load_for_turn(c, a)
        mem.hints.need = "leak"
        conv.id = b
        with pytest.raises(LookupError):
            await save(c, conv, mem)
    assert (await load(test_db, b))[1].hints.need is None


async def test_every_turn_function_refuses_a_mismatched_or_missing_binding(test_db):
    a, b = await new_conversation(test_db), await new_conversation(test_db)
    calls = [
        lambda c: load_for_turn(c, b),
        lambda c: append_turn(c, b, 1, "m", "r", "VERIFY_ID", "VERIFY_ID", {}),
        lambda c: add_escalation(c, b, 1, "k", "r", {}),
        lambda c: transcript(c, b),
        lambda c: set_controls(c, b, today=date(2026, 1, 1)),
    ]
    for call in calls:
        async with test_db.app.conversation(a) as c:
            with pytest.raises(LookupError):
                await call(c)
        async with test_db.app.conversation(None) as c:
            with pytest.raises(LookupError):
                await call(c)
    async with test_db.app.conversation(b) as c:
        assert await transcript(c, b) == []
    assert (await load(test_db, b))[0].today is None
