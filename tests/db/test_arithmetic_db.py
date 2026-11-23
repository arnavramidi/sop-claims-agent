"""Whole turns for arithmetic: the database adds up the caller's own claims, code words the result."""

import pytest
from helpers import GAVE, facts, instructions, make_deps, make_form, new_conversation, say, stated

from claims_agent.db.claims import ClaimFilter, aggregate

PAID, ALL = {"net_pay": 0.95}, dict(how_many="all")


@pytest.fixture
async def margaret(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    return cid


def _t(r):
    return " ".join(f.text for f in facts(r))


async def test_total_paid_is_summed_by_code(deps, margaret):
    r = await say(deps, margaret, "how much have you paid me in total?", make_form(calculation="total", figures=PAID, **ALL))
    t = _t(r)
    assert "1205.00" in t and "CL-2102" in t and "not paid yet" in t


async def test_which_claim_paid_the_most(deps, margaret):
    r = await say(deps, margaret, "which claim paid the most?", make_form(calculation="highest", figures=PAID, **ALL))
    assert "CL-2011" in facts(r)[0].text and "780.00" in facts(r)[0].text


async def test_how_many_claims_are_open(deps, margaret):
    r = await say(deps, margaret, "how many claims do I have open?",
                  make_form(calculation="count", claim_statuses={"open": 0.95}, **ALL))
    assert "1" in facts(r)[0].text.split()


async def test_an_expected_total_is_labelled_expected(deps, margaret):
    r = await say(deps, margaret, "how much will I get if everything pays out?",
                  make_form(calculation="total", figures={"expected_reimbursement_amount": 0.9}, **ALL))
    t = _t(r)
    assert "4405.00" in t and "expected" in t and "will receive" not in t


async def test_a_total_names_the_claim_whose_figure_is_missing(deps, margaret, loader_conn):
    await loader_conn.execute("UPDATE claims SET net_pay = NULL WHERE case_id = 'CL-1899'")
    await loader_conn.commit()
    r = await say(deps, margaret, "total paid?", make_form(calculation="total", figures=PAID, **ALL))
    t = _t(r)
    assert "780.00" in t and "CL-1899" in t and "unavailable" in t


async def test_a_total_straight_after_verification_needs_no_claim_in_focus(deps, margaret):
    r = await say(deps, margaret, "how much have you paid me in total?",
                  make_form(calculation="total", figures=PAID, how_many="none"))
    assert "which_claim" not in instructions(r) and "1205.00" in _t(r)


async def test_no_figure_named_asks_which_of_the_four(deps, margaret):
    r = await say(deps, margaret, "which claim paid the most?",
                  make_form(calculation="highest", no_figure_said=True, **ALL))
    assert "ask_which_figure" in instructions(r)


async def test_a_money_field_that_is_not_one_never_reaches_sql(deps, margaret):
    async with deps.db.conversation(None) as conn:
        with pytest.raises(ValueError):
            await aggregate(conn, ClaimFilter(), __import__("datetime").date(2026, 3, 5), "net_pay; DROP TABLE claims")


async def test_a_total_covers_only_the_callers_own_claims(deps, margaret):
    async with deps.db.conversation(None) as conn:  # no conversation bound: the row rules show nothing
        agg = await aggregate(conn, ClaimFilter(), __import__("datetime").date(2026, 3, 5), "net_pay")
    assert agg.count == 0 and agg.total == 0


async def test_two_claims_in_focus_can_be_compared(deps, margaret):
    r = await say(deps, margaret, "compare my two closed claims",
                  make_form(mentions_claim=True, claim_statuses={"closed": 0.95}, how_many="several",
                            answer_wanted="compare_claims"))
    t = _t(r)
    assert "1205.00" in t and "355.00" in t and "related" not in t
