"""Whole turns where a pick is unsure: the cheaper steps run before the caller is asked, the larger model settles at
most one pick and rewrites at most one reply per turn, and only in a state allowed to."""

import pytest
from helpers import GAVE, facts, focus, instructions, make_form, memory, new_conversation, say, stated

from claims_agent.form.fields import Pick
from claims_agent.models.errors import ModelKeyRejected, ModelRequestRejected, ModelUnavailable
from claims_agent.states import State

BAD = "Your claim CL-9999 was approved."


def unsure_pick(a, pa, b, pb):
    return Pick(value=a, p=pa, options={a: pa, b: pb})


@pytest.fixture
async def margaret(deps):
    """Verified as P9 and has said nothing about a claim, so it sits in RESOLVE_INTENT."""
    cid = await new_conversation(deps)
    r = await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
                  stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert r.state == State.RESOLVE_INTENT
    return cid


@pytest.fixture
async def denial_case(deps, margaret):
    """P9 asking why CL-2048 was denied: in PROCESS_CASE, the opening given."""
    r = await say(deps, margaret, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                  stated(claim_number="CL-2048"))
    assert r.state == State.PROCESS_CASE
    return margaret


@pytest.fixture
async def margaret_one_closed_claim_in_focus(deps, margaret):
    """P9 quoted the closed claim CL-2011 with an unsure need nothing settled: the claim is in focus and the caller was
    asked which of the two needs, so the conversation is still in RESOLVE_INTENT."""
    r = await say(deps, margaret, "about CL-2011", make_form(quoted_claim_number=True, need=unsure_pick("next_steps", 0.4, "status_inquiry", 0.38)),
                  stated(claim_number="CL-2011"))
    assert r.state == State.RESOLVE_INTENT and await focus(deps, margaret) == ["CL-2011"]
    assert (await memory(deps, margaret, "pending")).kind == "which_need"
    return margaret


# ---- the brief's tests -------------------------------------------------------------------------------------------

async def test_an_unsure_need_in_resolve_intent_uses_the_loop(deps, margaret_one_closed_claim_in_focus):
    deps.filler.queue_one(Pick(value=None)); deps.settler.queue(Pick(value="status_inquiry", p=1.0, by="larger"))     # CL-2011 rules out two needs, so the narrowed re-ask runs first
    r = await say(deps, margaret_one_closed_claim_in_focus, "so what about it?", make_form(need=unsure_pick("next_steps", 0.4, "status_inquiry", 0.38)))
    assert r.state == State.PROCESS_CASE and {"unsure_steps": ["narrowed", "larger"]} in r.inspector["fired"]


async def test_a_strict_state_never_calls_the_larger_model(deps, conv):
    await say(deps, conv, "hmm", make_form(need=Pick(value=None))); assert deps.settler.calls == []


async def test_the_rewrite_budget_is_one_per_turn(deps, denial_case):
    rewrites = len(deps.wording.rewrites)
    deps.wording.queue([BAD, BAD])     # the first reply and its rewrite both fail the check
    r = await say(deps, denial_case, "anything else?", make_form())
    assert len(deps.wording.rewrites) - rewrites == 1 and r.inspector["reply_source"] == "template"


# ---- beyond the brief --------------------------------------------------------------------------------------------

async def test_the_narrowed_question_leaves_out_the_needs_the_claim_rules_out(deps, margaret_one_closed_claim_in_focus):
    deps.filler.queue_one(Pick(value="status_inquiry", p=0.9))
    r = await say(deps, margaret_one_closed_claim_in_focus, "so what about it?", make_form(need=unsure_pick("next_steps", 0.4, "status_inquiry", 0.38)))
    offered = dict(deps.filler.one_questions[-1].choices)
    assert "denial_question" not in offered and "document_submission" not in offered
    assert r.state == State.PROCESS_CASE and {"unsure_steps": ["narrowed"]} in r.inspector["fired"]
    assert deps.settler.calls[-1][1].name == "need" and len(deps.settler.calls) == 1   # only the fixture's own turn asked it


async def test_nothing_settled_asks_the_caller_between_the_two_readings(deps, margaret_one_closed_claim_in_focus):
    r = await say(deps, margaret_one_closed_claim_in_focus, "so what about it?", make_form(need=unsure_pick("next_steps", 0.4, "status_inquiry", 0.38)))
    assert r.state == State.RESOLVE_INTENT and {"unsure_steps": ["narrowed", "larger", "ask_caller"]} in r.inspector["fired"]
    assert {"rule": "unsure_need", "readings": ["next_steps", "status_inquiry"]} in r.inspector["fired"]
    assert {"rule": "larger_model", "outcome": "cannot_tell"} in r.inspector["fired"]


async def test_the_larger_model_unavailable_asks_the_caller_and_is_recorded(deps, margaret_one_closed_claim_in_focus):
    deps.settler.queue(ModelUnavailable("scripted"))
    r = await say(deps, margaret_one_closed_claim_in_focus, "so what about it?", make_form(need=unsure_pick("next_steps", 0.4, "status_inquiry", 0.38)))
    assert r.state == State.RESOLVE_INTENT and "ask_need" in instructions(r)
    assert {"rule": "larger_model", "outcome": "unavailable"} in r.inspector["fired"]


async def test_a_malformed_request_to_the_larger_model_asks_the_caller(deps, margaret_one_closed_claim_in_focus):
    deps.settler.queue(ModelRequestRejected("scripted"))
    r = await say(deps, margaret_one_closed_claim_in_focus, "so what about it?", make_form(need=unsure_pick("next_steps", 0.4, "status_inquiry", 0.38)))
    assert r.state == State.RESOLVE_INTENT and {"rule": "larger_model", "outcome": "request_rejected"} in r.inspector["fired"]


async def test_a_rejected_key_on_the_larger_model_rolls_the_turn_back_and_is_raised(deps, margaret_one_closed_claim_in_focus):
    turns_before = await memory(deps, margaret_one_closed_claim_in_focus, "counters.turn_no")
    deps.settler.queue(ModelKeyRejected("scripted"))
    with pytest.raises(ModelKeyRejected):
        await say(deps, margaret_one_closed_claim_in_focus, "so what about it?", make_form(need=unsure_pick("next_steps", 0.4, "status_inquiry", 0.38)))
    assert await memory(deps, margaret_one_closed_claim_in_focus, "counters.turn_no") == turns_before


async def test_an_unsure_answer_in_process_case_is_settled_by_the_larger_model(deps, denial_case):
    deps.settler.queue(Pick(value="appeal_deadline", p=1.0, by="larger"))
    r = await say(deps, denial_case, "and the timing on that?", make_form(answer_wanted=unsure_pick("appeal_deadline", 0.4, "submission_timing", 0.35)))
    assert "deadline" in [f.kind for f in facts(r)] and {"unsure_steps": ["larger"]} in r.inspector["fired"]
    assert deps.settler.calls[-1][1].name == "answer_wanted"


async def test_an_unsure_answer_nothing_settles_asks_which_answer(deps, denial_case):
    r = await say(deps, denial_case, "and the timing on that?", make_form(answer_wanted=unsure_pick("appeal_deadline", 0.4, "submission_timing", 0.35)))
    assert "ask_which_answer" in instructions(r) and {"unsure_steps": ["larger", "ask_caller"]} in r.inspector["fired"]


async def test_a_state_whose_freedom_does_not_settle_never_asks_the_larger_model(deps, denial_case):
    deps.settings.freedom["free"].settle_unsure = False
    try:
        r = await say(deps, denial_case, "and the timing on that?", make_form(answer_wanted=unsure_pick("appeal_deadline", 0.4, "submission_timing", 0.35)))
    finally:
        deps.settings.freedom["free"].settle_unsure = True
    assert deps.settler.calls == [] and {"unsure_steps": ["ask_caller"]} in r.inspector["fired"]


async def test_a_reading_the_claim_rules_out_is_never_put_to_the_caller(deps, margaret):
    # CL-1899 is closed with no denial reason: the denial question is gone before anyone is asked, and with one real
    # need left the claim opens as a general question instead of asking the caller.
    unsure = Pick(value="denial_question", p=0.45, options={"denial_question": 0.45, "next_steps": 0.4, "none": 0.15})
    r = await say(deps, margaret, "about my dental claim, what now?", make_form(how_many="one", claim_types={"dental": 0.95}, need=unsure))
    assert await focus(deps, margaret) == ["CL-1899"] and r.state == State.PROCESS_CASE
    assert not any("denial" in f.text for f in facts(r) if f.kind == "possible_need")
    assert "denial_question" not in dict(deps.settler.calls[-1][1].choices)
