"""Turning back to a claim discussed earlier while another is in focus, with the readings the fast model returned for
"Back to the denied one: when is the appeal deadline?" after the caller had turned to the auto claim."""

import pytest
from helpers import GAVE, facts, focus, instructions, make_form, new_conversation, say, stated

from claims_agent.form.fields import Pick
from claims_agent.models.errors import ModelRequestRejected, ModelUnavailable
from claims_agent.states import State


async def _on_the_auto_claim(deps):
    """P9, told about the denied claim CL-2048, then turned to the auto claim CL-2102 ("And what about my auto
    claim?"): CL-2102 in focus."""
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    r = await say(deps, cid, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                  stated(claim_number="CL-2048"))
    assert r.state == State.PROCESS_CASE and await focus(deps, cid) == ["CL-2048"]
    await say(deps, cid, "And what about my auto claim?",
              make_form(mentions_claim=True, how_many=Pick(value="one", p=0.49), claim_types={"auto": 1.0},
                        search_relation=Pick(value="new_search", p=0.97), need=Pick(value="general_claim_question",
                                                                                    p=0.93),
                        answer_wanted=Pick(value="something_else", p=0.63)))
    assert await focus(deps, cid) == ["CL-2102"]
    return cid


async def test_a_description_none_of_the_claims_in_focus_fits_is_a_new_search_even_when_read_as_narrowing(deps):
    # The fast model read "back to the denied one" as narrowing (0.97) the claims being discussed. Added to the auto
    # claim's own filter, "denied" found nothing, and the caller was told no claim matched and shown the auto claim.
    cid = await _on_the_auto_claim(deps)
    r = await say(deps, cid, "Back to the denied one: when is the appeal deadline?",
                  make_form(mentions_claim=True, how_many=Pick(value="one", p=0.98), claim_statuses={"denied": 1.0},
                            no_year_said=0.78, search_relation=Pick(value="narrowing", p=0.97),
                            need=Pick(value="denial_question", p=0.58),
                            answer_wanted=Pick(value="next_steps", p=0.44,
                                               options={"next_steps": 0.44, "something_else": 0.4}),
                            calculation=Pick(value="days_left", p=0.51)))
    assert await focus(deps, cid) == ["CL-2048"]
    assert "nearest_matches" not in instructions(r)
    assert {"rule": "narrowing_fits_none_shown"} in r.inspector["fired"]


async def _with_the_january_list(deps):
    """P9 shown the two healthcare claims from January (CL-2048, CL-2011) and asked which one."""
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    r = await say(deps, cid, "I have a question about my healthcare claim from January.",
                  make_form(mentions_claim=True, how_many="one", claim_types={"healthcare": 1.0}, months={1: 1.0},
                            no_year_said=True, need="general_claim_question"))
    assert "which_claim" in instructions(r)
    return cid


async def test_a_description_one_of_the_listed_claims_fits_still_narrows_the_list(deps):
    cid = await _with_the_january_list(deps)
    r = await say(deps, cid, "the denied one", make_form(claim_statuses={"denied": 1.0}, search_relation="narrowing",
                                                         how_many="one"))
    assert await focus(deps, cid) == ["CL-2048"]
    assert {"rule": "narrowing_fits_none_shown"} not in r.inspector["fired"]


async def test_a_time_counted_from_today_is_not_checked_against_the_claims_shown(deps):
    # "The newer one" picks among the listed claims; whether it fits them is the narrowing's own business.
    cid = await _with_the_january_list(deps)
    r = await say(deps, cid, "The newer one.", make_form(relative_time="most_recent", search_relation="narrowing",
                                                         how_many="one"))
    assert await focus(deps, cid) == ["CL-2048"]
    assert {"rule": "narrowing_fits_none_shown"} not in r.inspector["fired"]


async def test_a_question_asked_on_turning_to_a_claim_is_read_again_over_that_claims_answers(deps):
    # The form was filled with the auto claim's answers (no appeal deadline among them), so "when is the appeal
    # deadline?" read as something else (0.47). Over CL-2048's answers the fast model read the appeal deadline (0.89).
    cid = await _on_the_auto_claim(deps)
    deps.filler.queue_one(Pick(value="appeal_deadline", p=0.89, options={"appeal_deadline": 0.9}))
    r = await say(deps, cid, "Back to the denied one: when is the appeal deadline?",
                  make_form(mentions_claim=True, how_many=Pick(value="one", p=0.85), claim_statuses={"denied": 1.0},
                            search_relation=Pick(value="narrowing", p=0.95),
                            need=Pick(value="denial_question", p=0.79),
                            answer_wanted=Pick(value="something_else", p=0.47,
                                               options={"something_else": 0.5, "next_steps": 0.3}),
                            calculation=Pick(value="days_left", p=0.57)))
    assert await focus(deps, cid) == ["CL-2048"]
    asked = deps.filler.one_questions[-1]
    assert asked.name == "answer_wanted" and "appeal_deadline" in dict(asked.choices)
    assert [f.kind for f in facts(r) if f.claim == "CL-2048" and f.kind not in ("found_claim", "followup")] == ["deadline"]
    assert "2026-03-18" in r.reply


async def test_a_question_read_again_that_settles_nothing_leaves_the_first_reading(deps):
    cid = await _on_the_auto_claim(deps)
    r = await say(deps, cid, "Back to the denied one: and its status?",
                  make_form(mentions_claim=True, how_many="one", claim_statuses={"denied": 1.0},
                            search_relation="new_search", need="denial_question", answer_wanted="status"))
    assert await focus(deps, cid) == ["CL-2048"]
    assert deps.filler.one_questions and deps.filler.one_questions[-1].name == "answer_wanted"
    assert [f.kind for f in facts(r) if f.claim == "CL-2048" and f.kind not in ("found_claim", "followup")] == ["status"]


@pytest.mark.parametrize("error", [ModelUnavailable("m"), ModelRequestRejected("m")])
async def test_a_question_read_again_whose_call_fails_leaves_the_first_reading(deps, error):
    cid = await _on_the_auto_claim(deps)
    deps.filler.queue_one(error)
    r = await say(deps, cid, "Back to the denied one: and its status?",
                  make_form(mentions_claim=True, how_many="one", claim_statuses={"denied": 1.0},
                            search_relation="new_search", need="denial_question", answer_wanted="status"))
    assert r.state == State.PROCESS_CASE and await focus(deps, cid) == ["CL-2048"]
    assert [f.kind for f in facts(r) if f.claim == "CL-2048" and f.kind not in ("found_claim", "followup")] == ["status"]
