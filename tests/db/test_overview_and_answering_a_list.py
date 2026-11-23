"""A question about everything on the policy, "all of them" in answer to a list, and the same question asked again,
with the readings the fast model returned in a conversation on the running app (Margaret Chen, four claims)."""

from helpers import GAVE, facts, focus, instructions, make_form, memory, new_conversation, say, stated

from claims_agent.form.fields import Pick
from claims_agent.states import State

ALL_FOUR = {"CL-2102", "CL-2048", "CL-1899", "CL-2011"}


async def _verified(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    return cid


async def _with_the_january_list(deps):
    cid = await _verified(deps)
    await say(deps, cid, "my healthcare claim from January",
              make_form(how_many="one", claim_types={"healthcare": 0.95}, months={1: 0.95}, no_year_said=True))
    assert await memory(deps, cid, "focus.last_list") == ["CL-2048", "CL-2011"]
    return cid


POLICY_QUESTION = make_form(how_many=Pick(value="one", p=0.89), mentions_claim=0.24,
                            need=Pick(value="general_claim_question", p=0.45,
                                      options={"general_claim_question": 0.45, "account_question": 0.36}),
                            answer_wanted=Pick(value="something_else", p=0.43,
                                               options={"something_else": 0.43, "status": 0.37}))


async def test_a_question_about_the_whole_policy_gets_every_claim_told_and_one_question(deps):
    # The app's turn 6: "whats the deal with Policy POL-9921" was answered "your description does not distinguish
    # between these 4 claims" with a list, though nothing had been described.
    cid = await _verified(deps)
    r = await say(deps, cid, "whats the deal with Policy POL-9921", POLICY_QUESTION)
    assert set(await focus(deps, cid)) == ALL_FOUR and r.state == State.PROCESS_CASE
    assert {"rule": "nothing_described"} in r.inspector["fired"]
    assert "which_claim" not in instructions(r) and instructions(r)[-1] == "ask_which_of_these"
    assert not any(f.kind == "cannot_tell" for f in facts(r))
    # each claim is told with its summary on file
    summaries = [f for f in facts(r) if f.kind == "found_claim" and "Summary on file" in f.text]
    assert {f.claim for f in summaries} == ALL_FOUR


async def test_all_of_them_in_answer_to_a_list_puts_every_listed_claim_in_focus(deps):
    # The app's turns 7 to 9: how_many all (1.0) with a list pending was answered with the same list three times.
    cid = await _with_the_january_list(deps)
    r = await say(deps, cid, "give me the information regarding all of them",
                  make_form(how_many=Pick(value="all", p=1.0), need=Pick(value="general_claim_question", p=0.82),
                            listed_claim=Pick(value="none", p=0.94)))
    assert await focus(deps, cid) == ["CL-2048", "CL-2011"]
    assert "which_claim" not in instructions(r) and instructions(r)[-1] == "ask_which_of_these"


async def test_both_of_them_in_answer_to_a_list_of_two_puts_both_in_focus(deps):
    cid = await _with_the_january_list(deps)
    await say(deps, cid, "both of them", make_form(how_many="several"))
    assert await focus(deps, cid) == ["CL-2048", "CL-2011"]


async def test_the_healthcare_ones_from_a_list_of_every_claim_narrows_it(deps):
    cid = await _verified(deps)
    await say(deps, cid, "about CL-9999", make_form(quoted_claim_number=True), stated(claim_number="CL-9999"))
    assert set(await memory(deps, cid, "focus.last_list")) == ALL_FOUR
    await say(deps, cid, "the healthcare ones", make_form(how_many="several", claim_types={"healthcare": 1.0},
                                                          search_relation="narrowing"))
    assert set(await focus(deps, cid)) == {"CL-2048", "CL-2011"}


async def test_the_first_one_after_several_claims_were_told_picks_the_first(deps):
    cid = await _verified(deps)
    await say(deps, cid, "whats the deal with Policy POL-9921", POLICY_QUESTION)
    told = await focus(deps, cid)
    assert deps.filler.requests and len(told) == 4
    r = await say(deps, cid, "the first one", make_form(listed_claim=told[0], how_many="one"))
    assert [c.case_id for c in deps.filler.requests[-1].listed_claims] == told
    assert await focus(deps, cid) == told[:1] and r.state == State.PROCESS_CASE


async def test_a_list_shown_with_summaries(deps):
    cid = await _verified(deps)
    r = await say(deps, cid, "my healthcare claim from January",
                  make_form(how_many="one", claim_types={"healthcare": 0.95}, months={1: 0.95}, no_year_said=True))
    lines = [f.text for f in facts(r) if f.kind == "status"]
    assert len(lines) == 2 and all("summary:" in line for line in lines)


async def test_the_same_question_asked_three_times_with_only_hints_changing_offers_a_person(deps):
    # Re-stated needs were counted as progress, so the same list went out again and again with no person offered.
    cid = await _with_the_january_list(deps)
    for need in ("general_claim_question", "status_inquiry", "next_steps"):
        r = await say(deps, cid, "the one I told you about", make_form(how_many="one", need=need))
        assert "which_claim" in instructions(r)
    assert "offer_person" in instructions(r)
    assert {"rule": "stall_limit"} in r.inspector["fired"]


async def test_all_of_them_after_an_account_answer_tells_the_claims_not_the_policy_number_again(deps):
    # Live: "whats the deal with Policy POL-9921" read as an account question (0.78) and was answered with the policy
    # number; the hint stayed, and "give me the information regarding all of them" (all 0.8, need unsure) got the
    # policy number again.
    cid = await _verified(deps)
    await say(deps, cid, "whats the deal with Policy POL-9921",
              make_form(how_many=Pick(value="one", p=0.59), need=Pick(value="account_question", p=0.78)))
    r = await say(deps, cid, "give me the information regarding all of them",
                  make_form(how_many=Pick(value="all", p=0.8), need=Pick(value="general_claim_question", p=0.32),
                            answer_wanted=Pick(value="something_else", p=0.62)))
    assert {"rule": "account_question"} not in r.inspector["fired"]
    assert set(await focus(deps, cid)) == ALL_FOUR


async def test_a_request_to_change_account_details_is_not_answered_with_the_policy_number(deps):
    # Live: "I want to change the phone number on my account" read unsupported (1.0) and an account question (0.63).
    cid = await _verified(deps)
    r = await say(deps, cid, "No. I want to change the phone number on my account.",
                  make_form(about="unsupported_insurance", need=Pick(value="account_question", p=0.63)))
    assert "unsupported_offer_person" in instructions(r)
    assert {"rule": "account_question"} not in r.inspector["fired"]
    assert not any("POL-9921" in f.text for f in facts(r))
