"""The claim filter built from the form: pure code, no database."""

from helpers import make_form

from claims_agent.db.claims import ClaimFilter, filter_from_form, merge, relax, with_need
from claims_agent.settings import load_thresholds

T = load_thresholds()


# ---- the brief's tests ----------------------------------------------------------------------------------------

def test_an_unsure_answer_does_not_include_the_value():
    f = filter_from_form(make_form(claim_types={"healthcare": 0.4, "dental": 0.2}), T); assert f.types == frozenset()


def test_no_year_said_means_all_years():
    assert filter_from_form(make_form(months={1: 0.9}, no_year_said=True), T).years == frozenset()


def test_r6_both_types_and_both_months():
    f = filter_from_form(make_form(claim_types={"healthcare": 0.9, "dental": 0.8}, months={1: 0.9, 2: 0.9}), T)
    assert f.types == {"healthcare", "dental"} and f.months == {1, 2}


def test_the_need_implies_a_condition():
    assert with_need(ClaimFilter(), "denial_question").has_denial_reason and with_need(ClaimFilter(), "document_submission").has_documents


def test_ruling_10_relax_drops_the_least_sure_condition_first():
    f = ClaimFilter(types=frozenset({"healthcare"}), months=frozenset({3}), sureness={"types": 0.9, "months": 0.6})
    assert relax(f).months == frozenset() and relax(f).types == {"healthcare"}


def test_ruling_10_ties_drop_month_before_type():
    f = ClaimFilter(types=frozenset({"auto"}), months=frozenset({3}), sureness={"types": 0.8, "months": 0.8}); assert relax(f).types == {"auto"}


def test_r19_a_correction_replaces_only_that_value():
    old = ClaimFilter(types=frozenset({"healthcare"}), months=frozenset({1}))
    assert merge(old, ClaimFilter(months=frozenset({2})), "correcting") == ClaimFilter(types=frozenset({"healthcare"}), months=frozenset({2}))


def test_m4_a_new_search_does_not_combine_with_the_old():
    assert merge(ClaimFilter(types=frozenset({"auto"})), ClaimFilter(types=frozenset({"healthcare"})), "new_search").types == {"healthcare"}


# ---- added ------------------------------------------------------------------------------------------------------

def test_sureness_is_the_lowest_p_among_the_values_included():
    f = filter_from_form(make_form(claim_types={"healthcare": 0.9, "dental": 0.7, "auto": 0.1}, years={2025: 0.7}), T)
    assert f.sureness == {"types": 0.7, "years": 0.7}


def test_a_year_said_is_a_condition_and_a_relative_time_is_kept():
    f = filter_from_form(make_form(years={2025: 0.9}, relative_time="most_recent"), T)
    assert f.years == {2025} and f.relative == "most_recent" and f.sureness["relative"] == 1.0


def test_an_empty_form_is_an_empty_filter():
    assert filter_from_form(make_form(), T) == ClaimFilter()


def test_narrowing_adds_the_new_conditions_to_the_old():
    old = ClaimFilter(statuses=frozenset({"denied"}), sureness={"statuses": 0.9})
    f = merge(old, ClaimFilter(types=frozenset({"healthcare"}), sureness={"types": 0.7}), "narrowing")
    assert f.statuses == {"denied"} and f.types == {"healthcare"} and f.sureness == {"statuses": 0.9, "types": 0.7}


def test_narrowing_a_condition_already_set_keeps_the_values_both_allow():
    old = ClaimFilter(types=frozenset({"healthcare", "dental"}))
    assert merge(old, ClaimFilter(types=frozenset({"dental"})), "narrowing").types == {"dental"}


def test_with_no_list_shown_the_new_filter_stands_alone():
    assert merge(ClaimFilter(types=frozenset({"auto"})), ClaimFilter(months=frozenset({2})), "no_list_shown") == \
        ClaimFilter(months=frozenset({2}))


def test_merge_with_nothing_stored_is_the_new_filter():
    new = ClaimFilter(types=frozenset({"auto"}))
    assert merge(None, new, "narrowing") == new


def test_ruling_10_the_full_tie_order():
    f = ClaimFilter(types=frozenset({"auto"}), statuses=frozenset({"open"}), months=frozenset({3}),
                    years=frozenset({2026}), relative="this_year")
    dropped = []
    while f is not None:
        nxt = relax(f)
        if nxt is None:
            break
        dropped.append(next(k for k in ("months", "years", "relative", "statuses", "types")
                            if getattr(f, k) not in (frozenset(), "none") and getattr(nxt, k) in (frozenset(), "none")))
        f = nxt
    assert dropped == ["months", "years", "relative", "statuses", "types"]


def test_relax_returns_none_when_no_condition_is_left():
    assert relax(ClaimFilter()) is None


def test_relax_keeps_what_the_need_implies():
    f = ClaimFilter(types=frozenset({"auto"}), has_denial_reason=True)
    assert relax(f) == ClaimFilter(has_denial_reason=True)


# ---- fix round 1 ---------------------------------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.mark.parametrize("relation", ["narrowing", "correcting", "new_search", "no_list_shown"])
def test_an_empty_new_filter_never_replaces_a_stored_one(relation):
    old = ClaimFilter(types=frozenset({"dental"}), sureness={"types": 0.9})
    assert merge(old, ClaimFilter(), relation) == old
