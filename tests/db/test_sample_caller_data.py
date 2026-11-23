"""The added sample callers, through the real loader and the database's own identity functions: every row loads with
nothing set aside, each caller verifies with three of their own details and sees only their own claims, the two
people who share a name are told apart by other details, and the claims the new callers have are worded correctly."""

import uuid
from datetime import date

import pytest
from helpers import stated

from claims_agent.answers.facts import deadline_fact
from claims_agent.db.claims import ClaimFilter, fetch
from claims_agent.db.identity import bind_policyholder, match_identity
from claims_agent.identity.normalize import normalise
from claims_agent.loader.read import read_folder
from claims_agent.settings import ROOT

pytestmark = pytest.mark.db

TODAY = date(2026, 3, 5)
NEW = {
    "P14": dict(policyholder_name="Daniel Okafor", date_of_birth=["1978-06-02"], id_digits="2051"),
    "P15": dict(policyholder_name="Priya Nair", date_of_birth=["1992-11-09"], id_digits="7733"),
    "P16": dict(policyholder_name="José García", date_of_birth=["1983-04-27"], id_digits="3946"),
    "P17": dict(policyholder_name="Sam Rivera", date_of_birth=["1987-02-14"], id_digits="1208"),
    "P18": dict(policyholder_name="Sam Rivera", date_of_birth=["1995-09-30"], id_digits="8365"),
}
CLAIMS = {"P14": {"CL-4101", "CL-4102"}, "P15": {"CL-4201", "CL-4202"}, "P16": {"CL-4301", "CL-4302"},
          "P17": {"CL-4401"}, "P18": {"CL-4501"}}


@pytest.fixture
async def db(load_dataset):
    return await load_dataset("starter")


async def kinds(db, **v) -> dict[str, frozenset[str]]:
    s, _ = normalise(stated(**v))
    cid = uuid.uuid4()
    async with db.app.conversation(cid) as c:
        return {m.party_id: m.kinds for m in await match_identity(c, cid, s)}


async def verified(db, party_id, **v) -> tuple[bool, uuid.UUID]:
    s, _ = normalise(stated(**v))
    cid = uuid.uuid4()
    async with db.app.conversation(cid) as c:
        await c.execute("INSERT INTO conversations (id, state, consent_script, today) "
                        "VALUES (%s, 'VERIFY_ID', 'default', '2026-03-05')", [cid])
        return await bind_policyholder(c, cid, party_id, s), cid


def test_every_new_row_loads_and_nothing_is_set_aside():
    r = read_folder(ROOT / "data" / "starter")
    assert r.report == [] and (len(r.customers), len(r.claims), len(r.representatives)) == (9, 13, 2)


@pytest.mark.parametrize("party", sorted(NEW))
async def test_each_new_caller_verifies_with_three_details_and_sees_only_their_claims(db, party):
    ok, cid = await verified(db, party, **NEW[party])
    assert ok is True
    async with db.app.conversation(cid) as c:
        assert {x.case_id for x in await fetch(c, ClaimFilter(), TODAY, 20)} == CLAIMS[party]


async def test_two_details_do_not_verify_a_new_caller(db):
    two = {k: v for k, v in NEW["P14"].items() if k != "id_digits"}
    assert (await verified(db, "P14", **two))[0] is False


async def test_the_name_and_one_detail_of_a_same_name_caller_do_not_verify(db):
    k = await kinds(db, policyholder_name="Sam Rivera", phone="650-555-0104")
    assert k["P17"] == {"name", "phone"} and k["P18"] == {"name"}
    assert (await verified(db, "P17", policyholder_name="Sam Rivera", phone="650-555-0104"))[0] is False


async def test_three_details_of_one_same_name_caller_verify_that_one_only(db):
    ok, cid = await verified(db, "P18", **NEW["P18"])
    assert ok is True
    async with db.app.conversation(cid) as c:
        assert {x.case_id for x in await fetch(c, ClaimFilter(), TODAY, 20)} == {"CL-4501"}
    k = await kinds(db, **NEW["P18"])
    assert k["P18"] == {"name", "dob", "id"} and k["P17"] == {"name"}
    assert (await verified(db, "P17", **NEW["P18"]))[0] is False  # the other Sam's details do not open P17


async def test_a_name_without_accents_matches_the_accented_name(db):
    k = await kinds(db, policyholder_name="Jose Garcia", date_of_birth=["1983-04-27"], id_digits="3946")
    assert k["P16"] == {"name", "dob", "id"}
    assert (await verified(db, "P16", policyholder_name="Jose Garcia", date_of_birth=["1983-04-27"],
                           id_digits="3946"))[0] is True


async def test_no_new_caller_is_confused_with_an_existing_one(db):
    for party, v in NEW.items():
        others = {p for p, kk in (await kinds(db, **v)).items() if len(kk) >= 2}
        assert others <= {party}, party


async def test_a_deadline_that_has_passed_never_says_days_are_left(db):
    ok, cid = await verified(db, "P15", **NEW["P15"])
    async with db.app.conversation(cid) as c:
        claim = next(x for x in await fetch(c, ClaimFilter(), TODAY, 20) if x.case_id == "CL-4201")
    fact, passed = deadline_fact(claim, TODAY)
    assert passed is True and "left" not in fact.text and "has passed (13 days ago)" in fact.text


async def test_deadlines_today_and_tomorrow_are_not_passed(db):
    for party, case, words in (("P17", "CL-4401", "which is today"), ("P14", "CL-4102", "1 day left, tomorrow")):
        ok, cid = await verified(db, party, **NEW[party])
        async with db.app.conversation(cid) as c:
            claim = next(x for x in await fetch(c, ClaimFilter(), TODAY, 20) if x.case_id == case)
        fact, passed = deadline_fact(claim, TODAY)
        assert passed is False and words in fact.text


async def test_a_customer_with_two_denied_claims_of_one_type_has_both_visible(db):
    ok, cid = await verified(db, "P14", **NEW["P14"])
    async with db.app.conversation(cid) as c:
        claims = await fetch(c, ClaimFilter(), TODAY, 20)
    assert sorted((x.case_type, x.status) for x in claims) == [("healthcare", "denied")] * 2


async def test_an_open_claim_with_a_document_outstanding_has_no_denial(db):
    ok, cid = await verified(db, "P16", **NEW["P16"])
    async with db.app.conversation(cid) as c:
        claim = next(x for x in await fetch(c, ClaimFilter(), TODAY, 20) if x.case_id == "CL-4301")
    assert claim.status == "open" and claim.documents == ("treating provider office note",)
    assert claim.denial_reason is None and claim.appeal_deadline is None
