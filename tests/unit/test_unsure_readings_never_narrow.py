"""A claim description narrows the search only with values the fast model is sure the caller said. A reading in the
unsure band is not the caller naming a value: it never hides claims the caller asked about."""

from helpers import make_form

from claims_agent.db.claims import ClaimFilter, filter_from_form
from claims_agent.settings import load_thresholds

T = load_thresholds()
JUST_BELOW = T.unsure_high - 0.01


def test_a_type_read_in_the_unsure_band_does_not_narrow_all_my_claims():
    # "Can you give me the status of all my claims?" as a real form read it: no type named, two weak readings.
    f = filter_from_form(make_form(how_many="all", claim_types={"healthcare": 0.39, "auto": 0.32}), T)
    assert f == ClaimFilter()


def test_a_weak_type_reading_does_not_turn_most_recent_into_most_recent_of_that_type():
    f = filter_from_form(make_form(relative_time="most_recent", claim_types={"healthcare": 0.48}), T)
    assert f.types == frozenset() and f.relative == "most_recent"


def test_every_kind_of_value_needs_a_sure_reading():
    f = filter_from_form(make_form(claim_types={"dental": JUST_BELOW}, claim_statuses={"denied": JUST_BELOW},
                                   months={3: JUST_BELOW}, years={2025: JUST_BELOW}), T)
    assert f == ClaimFilter()


def test_a_named_value_still_narrows_and_an_unsure_one_beside_it_does_not_join():
    f = filter_from_form(make_form(claim_types={"dental": 0.97, "healthcare": 0.44}), T)
    assert f.types == {"dental"} and f.sureness == {"types": 0.97}


def test_a_value_at_the_top_of_the_unsure_band_counts():
    assert filter_from_form(make_form(claim_types={"dental": T.unsure_high}), T).types == {"dental"}
