"""What the caller asked about one claim does not colour the answers about another; coming back to a claim already
discussed answers what was asked; a question about one money figure gets that figure."""

import pytest
from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, facts, focus, make_form, memory, new_conversation, say, stated

from claims_agent.states import State

BRIEF_WITH_NEED = make_form(**GAVE, mentions_claim=True, claim_types={"healthcare": 0.95},
                            claim_statuses={"denied": 0.95}, months={1: 0.95}, no_year_said=True, how_many="one",
                            need="denial_question")
TO_AUTO = make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95},
                    search_relation="new_search", how_many="one")


@pytest.fixture
async def brief_case(deps, conv):
    """The assignment's message, read as a denial question: verified, CL-2048 in focus with the denial opening."""
    r = await say(deps, conv, BRIEF_MSG, BRIEF_WITH_NEED, BRIEF_SAID)
    assert r.state == State.PROCESS_CASE and await focus(deps, conv) == ["CL-2048"]
    assert await memory(deps, conv, "focus.need") == "denial_question"
    return conv


def _claim_kinds(r, claim):
    return [f.kind for f in facts(r) if f.claim == claim]


async def test_the_first_need_does_not_decide_the_need_on_another_claim(deps, brief_case):
    await say(deps, brief_case, "what if I can't get the pathology report?",
              make_form(cant_get_document=True, which_document="pathology report"))
    r = await say(deps, brief_case, "what about my auto claim?", TO_AUTO)
    f = await memory(deps, brief_case, "focus")
    assert f.claim_ids == ["CL-2102"] and f.need != "denial_question" and f.need_note is None
    assert "not_denied" not in _claim_kinds(r, "CL-2102") and "not denied" not in r.reply


async def test_a_need_asked_before_a_list_still_applies_to_the_claim_picked_from_it(deps, conv):
    await say(deps, conv, BRIEF_MSG, make_form(**GAVE, claim_types={"healthcare": 0.95}, months={1: 0.95},
                                              no_year_said=True, how_many="one", need="next_steps"), BRIEF_SAID)
    assert await memory(deps, conv, "focus.last_list") == ["CL-2048", "CL-2011"]
    await say(deps, conv, "the first one", make_form(listed_claim="CL-2048"))
    assert await memory(deps, conv, "focus.need") == "next_steps"


async def test_a_need_said_again_later_is_used_for_the_claim_it_was_said_about(deps, brief_case):
    await say(deps, brief_case, "why was my dental claim denied?",
              make_form(mentions_claim=True, claim_types={"dental": 0.95}, search_relation="new_search", how_many="one",
                        need="denial_question", answer_wanted="something_else"))
    f = await memory(deps, brief_case, "focus")
    assert f.claim_ids == ["CL-1899"] and f.need_note == "not_denied"


# ---- coming back to a claim already discussed -----------------------------------------------------------------

BACK_TO_DENIED = dict(claim_statuses={"denied": 0.95}, how_many="one", mentions_claim=True,
                      search_relation="new_search", need="denial_question")


async def test_coming_back_to_a_claim_already_discussed_answers_what_was_asked(deps, brief_case):
    await say(deps, brief_case, "what about my auto claim?", TO_AUTO)
    r = await say(deps, brief_case, "back to the denied one, what was the deadline?",
                  make_form(**BACK_TO_DENIED, answer_wanted="appeal_deadline"))
    kinds = _claim_kinds(r, "CL-2048")
    assert await focus(deps, brief_case) == ["CL-2048"] and "deadline" in kinds
    assert not {"status", "denial_reason", "documents"} & set(kinds)


async def test_coming_back_to_a_claim_already_discussed_does_not_replay_its_opening(deps, brief_case):
    await say(deps, brief_case, "what about my auto claim?", TO_AUTO)
    r = await say(deps, brief_case, "back to the denied one", make_form(**BACK_TO_DENIED))
    kinds = _claim_kinds(r, "CL-2048")
    assert await focus(deps, brief_case) == ["CL-2048"]
    assert not {"denial_reason", "documents", "deadline"} & set(kinds)


async def test_a_claim_seen_for_the_first_time_still_gets_its_whole_opening(deps, conv):
    r = await say(deps, conv, BRIEF_MSG, BRIEF_WITH_NEED, BRIEF_SAID)
    assert {"status", "denial_reason", "documents", "deadline"} <= set(_claim_kinds(r, "CL-2048"))


# ---- one money figure ------------------------------------------------------------------------------------------

def _amounts(r):
    return [f.text for f in facts(r) if f.kind == "amount"]


async def test_asking_about_one_money_figure_gives_that_figure_and_its_meaning(deps, brief_case):
    r = await say(deps, brief_case, "what's the net pay on it?",
                  make_form(answer_wanted="amounts", figures={"net_pay": 0.95}))
    got = _amounts(r)
    assert len(got) == 1 and "net pay: 0.00" in got[0] and "Meaning of net pay" in got[0]


async def test_a_figure_named_without_an_answer_pick_gives_that_figure(deps, brief_case):
    r = await say(deps, brief_case, "what does allowed amount mean?",
                  make_form(answer_wanted="something_else", figures={"allowed_max_amount": 0.95}))
    got = _amounts(r)
    assert len(got) == 1 and "allowed max amount: 1450.00" in got[0]


async def test_asking_about_the_amounts_with_no_figure_named_gives_all_four(deps, brief_case):
    r = await say(deps, brief_case, "what are the amounts?", make_form(answer_wanted="amounts"))
    assert len(_amounts(r)) == 4


async def test_two_named_figures_give_those_two(deps, brief_case):
    r = await say(deps, brief_case, "what are the net pay and the net fee?",
                  make_form(answer_wanted="amounts", figures={"net_pay": 0.95, "net_fee": 0.95}))
    got = _amounts(r)
    assert len(got) == 2 and any("net pay" in t for t in got) and any("net fee" in t for t in got)


async def test_the_assignment_message_without_a_need_still_opens_with_status(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, BRIEF_MSG, make_form(**GAVE, claim_types={"healthcare": 0.95},
                                                 claim_statuses={"denied": 0.95}, months={1: 0.95},
                                                 no_year_said=True, how_many="one"),
                  stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert r.state == State.PROCESS_CASE and "status" in _claim_kinds(r, "CL-2048")
