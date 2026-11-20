"""Fitting the thresholds from recorded, labelled readings: the selection rule, and where each reading counts."""

import tomllib

from calibrate_thresholds import (Labelled, candidates, fitted_toml, labelled_from, readings_of, request_for, sweep)

from claims_agent.form.fields import Pick, YesNo
from claims_agent.settings import Thresholds, load_thresholds
from claims_agent.states import State


def cases(name, yes_ps, no_ps):
    return [Labelled(name, p, True) for p in yes_ps] + [Labelled(name, p, False) for p in no_ps]


# ---- the selection rule ---------------------------------------------------------------------------------------------

def test_the_threshold_with_the_fewest_wrong_reads_wins():
    value, why = sweep(cases("yes", [0.7] * 12, [0.3] * 12), current=0.5, forbid_false_yes=False)
    assert 0.35 <= value <= 0.7 and "fewest" in why


def test_a_tie_goes_to_the_more_cautious_value():
    assert sweep(cases("yes", [0.9] * 12, [0.1] * 12), current=0.5, forbid_false_yes=False)[0] == 0.9


def test_where_no_labelled_no_may_read_as_yes_none_does():
    value, _ = sweep(cases("send_email_min", [0.95] * 15 + [0.6] * 5, [0.7] * 3), current=0.8, forbid_false_yes=True)
    assert value > 0.7


def test_too_few_cases_keeps_the_current_value_and_says_so():
    value, why = sweep(cases("pick_confident", [0.9] * 5, [0.2] * 5), current=0.6, forbid_false_yes=False)
    assert value == 0.6 and "too few" in why


def test_when_every_candidate_lets_a_labelled_no_through_the_current_value_is_kept():
    value, why = sweep(cases("send_email_min", [0.99] * 20, [0.97]), current=0.8, forbid_false_yes=True)
    assert value == 0.8 and "kept" in why


def test_the_candidates_are_exact_twentieths():
    assert candidates() == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                            0.85, 0.9, 0.95]


def test_a_reading_equal_to_the_threshold_reads_as_yes():
    # a yes at exactly 0.9 must count as a yes at 0.9, as the app reads it (p >= threshold)
    assert sweep(cases("yes", [0.9] * 12, [0.85] * 12), current=0.5, forbid_false_yes=False)[0] == 0.9


# ---- which reading feeds which threshold ------------------------------------------------------------------------------

def test_a_yes_no_field_feeds_yes():
    assert labelled_from({"field": "wants_person", "expected": True, "p": 0.8, "value": None}) == [Labelled("yes", 0.8, True)]


def test_a_refused_yes_no_reads_as_no():
    assert labelled_from({"field": "english", "expected": True, "p": None, "value": None}) == [Labelled("yes", 0.0, True)]


def test_claim_filter_fields_feed_the_include_threshold():
    for field in ("claim_type:dental", "claim_status:denied", "month:1", "year:2025"):
        assert labelled_from({"field": field, "expected": False, "p": 0.4, "value": None}) == \
            [Labelled("filter_include_min", 0.4, False)]


def test_a_pick_counts_as_right_only_when_it_picked_the_labelled_option():
    assert labelled_from({"field": "mood", "expected": "calm", "p": 0.7, "value": "calm"}) == [Labelled("pick_confident", 0.7, True)]
    assert labelled_from({"field": "mood", "expected": "calm", "p": 0.7, "value": "upset"}) == [Labelled("pick_confident", 0.7, False)]


def test_the_email_offer_reply_feeds_send_only_through_a_send_reading():
    send = labelled_from({"field": "email_offer_reply", "expected": "send", "p": 0.9, "value": "send"})
    other = labelled_from({"field": "email_offer_reply", "expected": "dont_send", "p": 0.9, "value": "dont_send"})
    assert Labelled("send_email_min", 0.9, True) in send and Labelled("send_email_min", 0.0, False) in other


def test_answer_wanted_feeds_the_prepared_reply_threshold_only_in_calm_single_question_cases():
    calm = labelled_from({"field": "answer_wanted", "expected": "status", "p": 0.8, "value": "status", "calm_single_question": True})
    busy = labelled_from({"field": "answer_wanted", "expected": "status", "p": 0.8, "value": "status", "calm_single_question": False})
    assert Labelled("prepared_reply_min", 0.8, True) in calm and all(c.threshold != "prepared_reply_min" for c in busy)
    assert Labelled("pick_confident", 0.8, True) in calm and Labelled("pick_confident", 0.8, True) in busy


def test_readings_are_taken_from_the_form_by_question_name():
    from helpers import make_form
    form = make_form(claim_types={"dental": 0.7}, months={3: 0.1}, mood=Pick(value="upset", p=0.6), wants_person=0.2)
    got = readings_of(form, ["claim_type:dental", "month:3", "mood", "wants_person", "caller_role"])
    assert got == {"claim_type:dental": (0.7, None), "month:3": (0.1, None), "mood": (0.6, "upset"),
                   "wants_person": (0.2, None)}  # a question not asked has no reading


def test_a_case_request_is_built_like_the_apps_for_its_state():
    vocab = {"case_type": {"dental"}, "status": {"denied"}, "relationship": {"son"}, "need": {"next_steps"}}
    req = request_for({"message": "hi", "context": {"state": "POST_PROCESS", "verified": True, "pending": "send it?"}},
                      vocab, ("status",), {"answer_wanted:status": ("where is it",)}, 2026)
    assert req.state == State.POST_PROCESS and req.verified and req.pending == "send it?" and req.claim_types == ("dental",)
    assert req.years == (2022, 2023, 2024, 2025, 2026) and req.answers == (("status", "status"),)
    assert req.examples == {"answer_wanted:status": ("where is it",)}


# ---- the fitted file ------------------------------------------------------------------------------------------------

def test_the_fitted_file_is_valid_thresholds_with_a_comment_per_threshold():
    current = load_thresholds()
    fitted = {"yes": (0.45, "0.45: fewest wrong reads"), "send_email_min": (current.send_email_min, "kept: too few")}
    text = fitted_toml(current, fitted, source="evals/probabilities.jsonl")
    values = Thresholds.model_validate(tomllib.loads(text))
    assert values.yes == 0.45 and values.send_email_min == current.send_email_min
    lines = text.splitlines()
    for name in Thresholds.model_fields:
        at = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = "))
        assert lines[at - 1].startswith("# "), name
