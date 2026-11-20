"""Arithmetic as facts: totals, differences, rankings, counts and day counts, and the cross-claim comparison."""

from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal

import pytest

from claims_agent.answers.arithmetic import calculation_facts
from claims_agent.answers.facts import available, build
from claims_agent.db.claims import Aggregate, Claim
from claims_agent.db.guidance import guidance_from
from claims_agent.form.questions import FormRequest, build_questions
from claims_agent.loader.read import MONEY_FIELDS, read_folder
from claims_agent.settings import ROOT, load_settings
from claims_agent.states import State

STARTER = ROOT / "data" / "starter"


def _claims():
    r = read_folder(STARTER)
    return {c.case_id: Claim(c.case_id, c.case_type, c.created_at, c.status, c.summary, c.denial_reason, (),
                             c.appeal_deadline, c.expected_reimbursement_amount, c.allowed_max_amount, c.net_pay,
                             c.net_fee, datetime(2026, 3, 5)) for c in r.claims}


_C = _claims()
CL2048, CL2011, CL1899, CL2102 = _C["CL-2048"], _C["CL-2011"], _C["CL-1899"], _C["CL-2102"]
G, S, TODAY = guidance_from(read_folder(STARTER)), load_settings(), date(2026, 3, 5)


def text(r):
    return " ".join(f.text for f in r.facts)


def test_a3_a_difference_gives_both_figures_and_no_reason():
    r = calculation_facts("difference", ["allowed_max_amount", "net_pay"], [CL2011], None, G, S, TODAY)
    t = text(r)
    assert "800.00" in t and "780.00" in t and "20.00" in t and "because" not in t
    assert any(f.kind == "not_on_file" and "reason" in f.text for f in r.facts)


def test_a7_days_open_are_counted_against_today():
    assert "5 days" in text(calculation_facts("days_open", [], [CL2102], None, G, S, TODAY))


def test_a5_with_no_figure_named_it_asks_which_of_the_four():
    r = calculation_facts("highest", [], [CL2048, CL2011], None, G, S, TODAY)
    assert r.grounded is False and all(name.replace("_", " ") in text(r) for name in G.glossary)


def test_compare_claims_gives_figures_totals_and_differences_and_links_nothing():
    assert "compare_claims" in available(CL2011, G, 2, S) and "compare_claims" not in available(CL2011, G, 1, S)
    t = text(build("compare_claims", [CL2011, CL1899], G, S, TODAY))
    assert "780.00" in t and "425.00" in t and "1205.00" in t and "355.00" in t
    assert "related" not in t and "linked" not in t


def test_an_unknown_money_field_is_refused():
    with pytest.raises(ValueError):
        calculation_facts("total", ["party_id"], [CL2011], None, G, S, TODAY)


def _request(verified):
    return FormRequest(message="how much in total?", state=State.RESOLVE_INTENT, verified=verified)


def test_calculation_questions_are_not_asked_before_verification():
    before = {q.name for q in build_questions(_request(False))}
    after = {q.name for q in build_questions(_request(True))}
    wanted = {"calculation", "no_figure_said", *(f"figure:{m}" for m in MONEY_FIELDS)}
    assert not wanted & before and wanted <= after


def test_a_total_says_how_many_claims_it_covers_and_names_the_one_without_a_figure():
    agg = Aggregate(Decimal("780.00"), 3, ("CL-1899",), (("CL-2011", Decimal("780.00")),), ("CL-2102",))
    t = text(calculation_facts("total", ["net_pay"], [], agg, G, S, TODAY))
    assert "across 2 claims: 780.00" in t and "CL-1899" in t and "unavailable" in t and "not paid yet" in t


def test_a_total_of_nothing_on_file_is_not_a_zero():
    r = calculation_facts("total", ["net_pay"], [], Aggregate(Decimal(0), 2, ("A", "B"), ()), G, S, TODAY)
    assert r.facts[0].kind == "not_on_file" and "0.00" not in text(r)


def test_an_expected_total_is_labelled_expected():
    agg = Aggregate(Decimal("4405.00"), 4, (), ())
    t = text(calculation_facts("total", ["expected_reimbursement_amount"], [], agg, G, S, TODAY))
    assert "4405.00" in t and "expected" in t and "will receive" not in t


def test_tied_claims_are_both_named():
    agg = Aggregate(Decimal(0), 2, (), (("CL-1", Decimal("5.00")), ("CL-2", Decimal("5.00"))))
    assert "CL-1 and CL-2" in text(calculation_facts("highest", ["net_pay"], [], agg, G, S, TODAY))


def test_days_left_is_the_deadline_answer():
    r = calculation_facts("days_left", [], [CL2048], None, G, S, TODAY)
    assert "13 days left" in text(r)
