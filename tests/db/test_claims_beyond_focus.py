"""With claims in focus, a message that points at other claims goes back to choosing them, decided by what the message
says rather than by how sure the fast model is that it starts a new search: a claim number that is none of the claims
in focus, a description that a claim in focus does not fit, or a wish for more claims than are in focus."""

import pytest
from helpers import facts, focus, instructions, make_form, say, stated

from claims_agent.form.fields import Pick
from claims_agent.states import State


def unsure(value: str, p: float = 0.4, **others: float) -> Pick:
    return Pick(value=value, p=p, options={value: p, **others})


@pytest.fixture
async def margaret(deps):
    from helpers import new_conversation, verified_party
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, 1985-03-15, SSN 4472", make_form(gave_identity_detail=True, caller_role="policyholder"),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert await verified_party(deps, cid) == "P9"
    return cid


async def _on_2048(deps, cid):
    r = await say(deps, cid, "What's happening with cl 2048?",
                  make_form(mentions_claim=True, quoted_claim_number=True, how_many="one", need="status_inquiry",
                            answer_wanted="status"), stated(claim_number="cl 2048"))
    assert r.state == State.PROCESS_CASE and await focus(deps, cid) == ["CL-2048"]


async def test_a_claim_number_that_is_none_of_the_claims_in_focus_is_looked_up(deps, margaret):
    await _on_2048(deps, margaret)
    r = await say(deps, margaret, "Wait, I actually meant claim CL-2084.",
                  make_form(mentions_claim=True, quoted_claim_number=True, how_many="one",
                            search_relation=unsure("correcting", 0.45, narrowing=0.3),
                            answer_wanted=unsure("status", 0.3, appeal_deadline=0.25)), stated(claim_number="CL-2084"))
    assert "claim_not_on_policy" in instructions(r)


async def test_the_number_of_the_claim_in_focus_typed_another_way_stays_on_it(deps, margaret):
    await _on_2048(deps, margaret)
    r = await say(deps, margaret, "And for claim 2048, when is the deadline?",
                  make_form(mentions_claim=True, quoted_claim_number=True, how_many="one", answer_wanted="appeal_deadline",
                            search_relation="narrowing"), stated(claim_number="claim 2048"))
    assert await focus(deps, margaret) == ["CL-2048"] and "claim_not_on_policy" not in instructions(r)
    assert any("2026-03-18" in f.text for f in facts(r))


async def test_all_my_claims_then_all_my_denied_claims(deps, margaret):
    await say(deps, margaret, "Can you give me the status of all my claims?",
              make_form(how_many="all", claim_types={"healthcare": 0.39, "auto": 0.32}, need="status_inquiry",
                        answer_wanted="status"))
    assert sorted(await focus(deps, margaret)) == ["CL-1899", "CL-2011", "CL-2048", "CL-2102"]
    r = await say(deps, margaret, "And what about all my denied claims?",
                  make_form(mentions_claim=0.54, how_many=unsure("all", 0.55), claim_statuses={"denied": 0.87},
                            search_relation="narrowing", need="denial_question", answer_wanted=unsure("status", 0.27)))
    assert await focus(deps, margaret) == ["CL-2048"] and "which_claim" not in instructions(r)


async def test_my_oldest_claim_after_my_most_recent_one(deps, margaret):
    await say(deps, margaret, "What's happening with my most recent claim?",
              make_form(mentions_claim=True, how_many="one", relative_time="most_recent", claim_types={"healthcare": 0.48},
                        need="status_inquiry"))
    assert await focus(deps, margaret) == ["CL-2102"]
    await say(deps, margaret, "And what about my oldest claim?",
              make_form(mentions_claim=True, how_many="one", relative_time="oldest",
                        search_relation=unsure("new_search", 0.36, narrowing=0.3), answer_wanted=unsure("status", 0.4)))
    assert await focus(deps, margaret) == ["CL-2011"]


async def test_my_other_claims_after_the_opening_about_one(deps, margaret):
    await _on_2048(deps, margaret)
    r = await say(deps, margaret, "What about my other claims?",
                  make_form(mentions_claim=0.6, how_many="all", search_relation=unsure("new_search", 0.5, narrowing=0.3),
                            answer_wanted=unsure("status", 0.4, next_steps=0.3)))
    assert "ask_which_answer" not in instructions(r)
    told = " ".join(f.text for f in facts(r))
    assert all(c in told for c in ("CL-2011", "CL-1899", "CL-2102"))


async def test_a_question_that_describes_the_claim_in_focus_stays_on_it(deps, margaret):
    await _on_2048(deps, margaret)
    r = await say(deps, margaret, "Why was my healthcare claim denied?",
                  make_form(mentions_claim=True, how_many="one", claim_types={"healthcare": 0.95},
                            claim_statuses={"denied": 0.95}, search_relation=unsure("narrowing", 0.5),
                            answer_wanted="denial_reason"))
    assert await focus(deps, margaret) == ["CL-2048"] and r.state == State.PROCESS_CASE
    assert "which_claim" not in instructions(r)



async def test_another_claim_with_the_same_digits_is_not_the_claim_in_focus(load_dataset):
    """many_claims: Omar Reyes (P30) has CL-4006 and AU-4006."""
    from helpers import make_deps, new_conversation, verified_party
    deps = make_deps(await load_dataset("many_claims"))
    cid = await new_conversation(deps)
    await say(deps, cid, "Omar Reyes, 1978-04-09, SSN 2210", make_form(gave_identity_detail=True, caller_role="policyholder"),
              stated(policyholder_name="Omar Reyes", date_of_birth=["1978-04-09"], id_digits="2210"))
    assert await verified_party(deps, cid) == "P30"
    await say(deps, cid, "CL-4006 please", make_form(quoted_claim_number=True, how_many="one", need="status_inquiry"),
              stated(claim_number="CL-4006"))
    assert await focus(deps, cid) == ["CL-4006"]
    await say(deps, cid, "and AU-4006?", make_form(quoted_claim_number=True, how_many="one",
                                                   search_relation=unsure("new_search", 0.4)), stated(claim_number="AU-4006"))
    assert await focus(deps, cid) == ["AU-4006"]


async def test_all_my_denied_claims_read_with_an_unsure_status_but_a_sure_denial_need(deps, margaret):
    await say(deps, margaret, "Can you give me the status of all my claims?",
              make_form(how_many="all", need="status_inquiry", answer_wanted="status"))
    r = await say(deps, margaret, "And what about all my denied claims?",
                  make_form(mentions_claim=0.96, how_many="all", claim_statuses={"denied": 0.51},
                            search_relation="narrowing", need="denial_question", answer_wanted=unsure("status", 0.21)))
    assert await focus(deps, margaret) == ["CL-2048"] and "which_claim" not in instructions(r)


async def test_my_other_claims_read_as_several_while_one_claim_is_in_focus(deps, margaret):
    await _on_2048(deps, margaret)
    r = await say(deps, margaret, "What about my other claims?",
                  make_form(how_many="several", search_relation="narrowing", need="general_claim_question",
                            answer_wanted=unsure("something_else", 0.5)))
    told = " ".join(f.text for f in facts(r))
    assert all(c in told for c in ("CL-2011", "CL-1899", "CL-2102")) and "not_on_file" not in instructions(r)


async def test_why_was_it_denied_about_a_claim_that_was_not_keeps_the_claim_and_says_so(deps, margaret):
    await say(deps, margaret, "what about CL-2011?", make_form(quoted_claim_number=True, how_many="one",
                                                              need="status_inquiry"), stated(claim_number="CL-2011"))
    r = await say(deps, margaret, "why was it denied?", make_form(how_many="one", need="denial_question",
                                                                 answer_wanted="denial_reason"))
    assert await focus(deps, margaret) == ["CL-2011"]
    assert any(f.kind == "not_denied" for f in facts(r))


async def test_a_sum_asked_about_another_claim_is_worked_out_once_that_claim_is_chosen(deps, margaret):
    await _on_2048(deps, margaret)
    r = await say(deps, margaret, "On my older healthcare claim, CL-2011: why was I paid 780 and not 800?",
                  make_form(mentions_claim=True, quoted_claim_number=True, how_many="one", claim_types={"healthcare": 1.0},
                            search_relation="new_search", calculation="difference",
                            answer_wanted=Pick(value="something_else", p=0.71, options={"something_else": 0.71})),
                  stated(claim_number="CL-2011"))
    assert await focus(deps, margaret) == ["CL-2011"]
    assert any(f.get("rule") == "calculation" for f in r.inspector["fired"])
