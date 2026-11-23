"""The database counts wrong identity values itself and refuses to compare any more once a conversation reaches its
limit, whatever the server does: the server's own account cannot write, change or delete the count, cannot see
another conversation's, and cannot get a comparison (or a verification) out of a conversation at its limit, even
with the right values and calling the functions directly."""

import itertools
import random
import uuid
from datetime import date

import psycopg
import pytest

from claims_agent.db.identity import (IdentityLocked, bind_policyholder, clear_verified, match_identity,
                                      request_consent, wrong_values)
from claims_agent.identity.types import KINDS, StatedIdentity
from claims_agent.identity.verify import judge
from claims_agent.settings import load_settings
from helpers import verified_party

pytestmark = pytest.mark.db

LIMIT = load_settings().limits.wrong_values
LOCKED = "IL001"

MARGARET = StatedIdentity("Margaret Chen", (date(1985, 3, 15),), None, None, "4472", None)   # three right details
WRONG = [StatedIdentity(None, (), None, "a@nowhere.example", None, None),
         StatedIdentity(None, (), "2025550101", None, None, None),
         StatedIdentity(None, (), None, None, "0000", None),
         StatedIdentity(None, (), None, None, None, "POL0000")]

_STATED = "%s::text, %s::date[], %s::text, %s::text, %s::text, %s::text"
DIRECT = {   # each function as the server's account could call it, bypassing the server's code
    "check_identity": f"SELECT * FROM check_identity(%s::uuid, {_STATED})",
    "bind_policyholder": f"SELECT bind_policyholder(%s::uuid, 'P9', {_STATED})",
    "request_consent": f"SELECT request_consent(%s::uuid, 'P9', {_STATED}, 'David Chen')",
}


def _args(s: StatedIdentity) -> list:
    return [s.name, list(s.dobs), s.phone, s.email, s.id_last4, s.policy]


@pytest.fixture
async def db(load_dataset):
    return await load_dataset("starter")


async def new_conversation(db) -> uuid.UUID:
    cid = uuid.uuid4()
    async with db.app.conversation(cid) as c:
        await c.execute("INSERT INTO conversations (id, state, today) VALUES (%s, 'VERIFY_ID', '2026-03-05')", [cid])
    return cid


@pytest.fixture
async def conv(db) -> uuid.UUID:
    return await new_conversation(db)


async def compare(db, cid, s: StatedIdentity):
    """One comparison in a transaction of its own, as one turn makes it."""
    async with db.app.conversation(cid) as c:
        return await match_identity(c, cid, s)


async def counted(db, cid) -> list[str]:
    async with db.app.conversation(cid) as c:
        return await wrong_values(c, cid)


async def at_the_limit(db, cid) -> None:
    for s in WRONG[:LIMIT]:
        await compare(db, cid, s)
    assert len(await counted(db, cid)) == LIMIT


async def test_the_limit_is_the_setting():
    assert LIMIT == 3


# ---- the count: only the database writes it ------------------------------------------------------------------

@pytest.mark.parametrize("statement", [
    "INSERT INTO identity_wrong_values (conversation_id, fingerprint, counted_with, counted_in) "
    "VALUES (%(c)s, 'dob:0', 'x', pg_current_xact_id())",
    "UPDATE identity_wrong_values SET fingerprint = 'dob:0' WHERE conversation_id = %(c)s",
    "UPDATE identity_wrong_values SET counted_in = '0' WHERE conversation_id = %(c)s",
    "DELETE FROM identity_wrong_values WHERE conversation_id = %(c)s",
    "TRUNCATE identity_wrong_values",
    "SELECT counted_with FROM identity_wrong_values WHERE conversation_id = %(c)s",
])
async def test_the_servers_account_cannot_change_the_count(db, conv, statement):
    await compare(db, conv, WRONG[0])
    before = await counted(db, conv)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with db.app.conversation(conv) as c:
            await c.execute(statement, {"c": conv})
    assert await counted(db, conv) == before and len(before) == 1


async def test_clearing_the_conversations_memory_does_not_clear_the_count(db, conv):
    for s in WRONG[:2]:
        await compare(db, conv, s)
    async with db.app.conversation(conv) as c:   # what a server that forgets can do: its memory is its own to write
        await c.execute("UPDATE conversations SET memory = '{}' WHERE id = %s", [conv])
    assert len(await counted(db, conv)) == 2


async def test_another_conversations_count_is_neither_visible_nor_affected(db, conv):
    await at_the_limit(db, conv)
    other = await new_conversation(db)
    async with db.app.conversation(other) as c:
        assert (await (await c.execute("SELECT count(*) FROM identity_wrong_values")).fetchone())[0] == 0
        assert await wrong_values(c, conv) == []
    await compare(db, other, WRONG[0])             # the other conversation compares and counts on its own
    async with db.app.conversation(other) as c:
        assert await bind_policyholder(c, other, "P9", MARGARET) is True
    assert len(await counted(db, other)) == 1 and len(await counted(db, conv)) == LIMIT
    async with db.app.conversation(None) as c:
        assert (await (await c.execute("SELECT count(*) FROM identity_wrong_values")).fetchone())[0] == 0


# ---- what counts once ---------------------------------------------------------------------------------------------

async def test_the_same_wrong_value_twice_counts_once(db, conv):
    await compare(db, conv, WRONG[0])
    await compare(db, conv, WRONG[0])
    await compare(db, conv, StatedIdentity(None, (), None, "  A@Nowhere.Example ", None, None))  # the same, written so
    assert len(await counted(db, conv)) == 1


async def test_a_corrected_value_is_a_different_value(db, conv):
    await compare(db, conv, StatedIdentity("Margaret Chen", (date(1985, 3, 16),), None, None, "4472", None))
    await compare(db, conv, StatedIdentity("Margaret Chen", (date(1985, 3, 17),), None, None, "4472", None))
    marks = await counted(db, conv)
    assert len(marks) == 2 and all(m.startswith("dob:") for m in marks)


async def test_the_two_readings_of_one_date_are_one_value_in_any_order(db, conv):
    one, other = date(2001, 2, 3), date(2001, 3, 2)
    await compare(db, conv, StatedIdentity(None, (one, other), None, None, None, None))
    await compare(db, conv, StatedIdentity(None, (other, one), None, None, None, None))
    assert len(await counted(db, conv)) == 1


async def test_a_fingerprint_holds_the_kind_and_never_the_value(db, conv):
    await compare(db, conv, StatedIdentity(None, (), "2025550101", None, "0000", None))
    marks = await counted(db, conv)
    assert sorted(m.split(":")[0] for m in marks) == ["id", "phone"]
    assert all(len(m.split(":")[1]) == 16 and "2025550101" not in m for m in marks)
    other = await new_conversation(db)
    await compare(db, other, StatedIdentity(None, (), "2025550101", None, "0000", None))
    assert set(await counted(db, other)).isdisjoint(marks)   # salted with the conversation


@pytest.mark.parametrize("s", [
    StatedIdentity(None, (), "2830", None, None, None),            # a short phone is never compared
    StatedIdentity(None, (), None, "not-an-email", None, None),
    StatedIdentity(None, (), None, None, "44A2", None),
    StatedIdentity("M. Chen", (), None, None, None, None),          # only part of a name
    StatedIdentity(None, (), None, None, None, " - "),
])
async def test_a_value_the_database_does_not_compare_is_never_counted(db, conv, s):
    rows = await compare(db, conv, s)
    assert rows == [] and await counted(db, conv) == []


async def test_with_no_single_customer_a_value_that_matches_someone_is_not_counted(load_dataset):
    """The recorded rule kept as it was: two Margaret Chens tie, nobody can be singled out, and a value that matches
    some customer is compared but not counted (one that matches nobody is)."""
    db = await load_dataset("shared_name")
    cid = await new_conversation(db)
    await compare(db, cid, StatedIdentity("Margaret Chen", (), None, None, "0000", None))
    assert [m.split(":")[0] for m in await counted(db, cid)] == ["id"]


# ---- at the limit: every comparison refused ---------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DIRECT))
async def test_at_the_limit_each_function_refuses_even_the_right_values_called_directly(db, conv, name):
    await at_the_limit(db, conv)
    with pytest.raises(psycopg.DatabaseError) as e:
        async with db.app.conversation(conv) as c:
            await c.execute(DIRECT[name], [conv, *_args(MARGARET)])
    assert e.value.sqlstate == LOCKED and "4472" not in str(e.value)
    assert await verified_party(db, conv) is None and len(await counted(db, conv)) == LIMIT


async def test_at_the_limit_the_servers_calls_raise_and_are_never_an_empty_answer(db, conv):
    await at_the_limit(db, conv)
    for call in (lambda c: match_identity(c, conv, MARGARET),
                 lambda c: bind_policyholder(c, conv, "P9", MARGARET),
                 lambda c: request_consent(c, conv, "P9", MARGARET, "David Chen")):
        async with db.app.conversation(conv) as c:
            with pytest.raises(IdentityLocked):
                await call(c)
            # the refusal happened in a savepoint: the turn's own transaction goes on
            assert (await (await c.execute("SELECT 1")).fetchone())[0] == 1
    assert await verified_party(db, conv) is None


async def test_the_count_survives_the_identity_reset(db, conv):
    for s in WRONG[:LIMIT - 1]:
        await compare(db, conv, s)
    async with db.app.conversation(conv) as c:
        assert await bind_policyholder(c, conv, "P9", MARGARET) is True
        await clear_verified(c, conv)          # the database's part of the different-person reset
    assert await verified_party(db, conv) is None and len(await counted(db, conv)) == LIMIT - 1
    await compare(db, conv, WRONG[LIMIT - 1])
    with pytest.raises(IdentityLocked):
        await compare(db, conv, MARGARET)


async def test_the_values_that_reach_the_limit_may_still_verify_in_that_same_transaction(db, conv):
    """The turn that gives the last wrong value along with three right ones still verifies: the same values compared
    again reveal nothing new."""
    for s in WRONG[:LIMIT - 1]:
        await compare(db, conv, s)
    reaching = StatedIdentity("Margaret Chen", MARGARET.dobs, None, "b@nowhere.example", "4472", None)
    async with db.app.conversation(conv) as c:
        assert {m.party_id for m in await match_identity(c, conv, reaching)} == {"P9"}
        assert len(await wrong_values(c, conv)) == LIMIT
        assert await bind_policyholder(c, conv, "P9", reaching) is True
    assert await verified_party(db, conv) == "P9"


async def test_after_the_limit_other_values_in_the_same_transaction_are_refused(db, conv):
    for s in WRONG[:LIMIT - 1]:
        await compare(db, conv, s)
    with pytest.raises(IdentityLocked):
        async with db.app.conversation(conv) as c:
            await match_identity(c, conv, WRONG[LIMIT - 1])
            await bind_policyholder(c, conv, "P9", MARGARET)    # not the values that reached the limit
    assert await verified_party(db, conv) is None


async def test_the_values_that_reached_the_limit_are_refused_in_a_later_transaction(db, conv):
    for s in WRONG[:LIMIT - 1]:
        await compare(db, conv, s)
    reaching = StatedIdentity("Margaret Chen", MARGARET.dobs, None, "b@nowhere.example", "4472", None)
    await compare(db, conv, reaching)
    async with db.app.conversation(conv) as c:
        with pytest.raises(IdentityLocked):
            await bind_policyholder(c, conv, "P9", reaching)
    assert await verified_party(db, conv) is None


# ---- what is wrong: the database's judgement is the server's ---------------------------------------------------

POOLS = {   # for the two-Margarets data: values of one customer, of two, of another, and of nobody
    "name": [None, "Margaret Chen", "Ava Lopez", "Nobody Atall"],
    "dob": [None, "1985-03-15", "1971-06-02", "2000-01-01"],
    "phone": [None, "6503882920", "6505212836", "2025550101"],
    "email": [None, "margaret@email.com", "rosa.diaz@example.com", "none@nowhere.example"],
    "id": [None, "4472", "8301", "0000"],
    "policy": [None, "POL9921", "POL6021", "POL0000"],
}


async def test_the_database_judges_wrong_values_exactly_as_the_server_rule_does(load_dataset):
    """Every value the database counts is one the server's rule (identity/verify.py) calls wrong, and the reverse,
    over a sample of the ways five details and a policy number can match one customer, two, another or nobody."""
    db = await load_dataset("shared_name")
    combos = list(itertools.product(*POOLS.values()))
    sample = random.Random(20261122).sample(combos, 600)
    mismatches, seen = [], set()
    async with db.app.conversation(None) as c:   # one transaction, bound afresh to a new conversation per sample
        for values in sample:
            v = dict(zip(POOLS, values))
            s = StatedIdentity(v["name"], (date.fromisoformat(v["dob"]),) if v["dob"] else (), v["phone"],
                               v["email"], v["id"], v["policy"])
            cid = uuid.uuid4()
            await c.execute("SELECT set_config('app.conversation_id', %s, true)", [str(cid)])
            matches = await match_identity(c, cid, s)
            by_database = {m.split(":")[0] for m in await wrong_values(c, cid)}
            decision = judge(frozenset(k for k in v if v[k] is not None), matches)
            by_rule = set(decision.wrong_now)
            seen.add(("candidate" if decision.candidate else "no candidate", "wrong" if by_rule else "none wrong",
                      "verified" if decision.verified_party else "not verified"))
            if by_database != by_rule:
                mismatches.append((v, by_database, by_rule))
    assert mismatches == []
    assert set(KINDS) | {"policy"} == set(POOLS)
    # the sample reached every kind of outcome the rule has, so agreement means something
    assert {("candidate", "wrong", "verified"), ("candidate", "wrong", "not verified"),
            ("candidate", "none wrong", "verified"), ("no candidate", "wrong", "not verified"),
            ("no candidate", "none wrong", "not verified")} <= seen
