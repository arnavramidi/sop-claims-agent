"""Engine rules that every state relies on, exercised through whole turns (scripted models, real Postgres)."""

import pytest
from helpers import instructions, make_deps, make_form, memory, new_conversation, say, stated

from claims_agent import engine
from claims_agent.engine import StepOutcome
from claims_agent.memory import Pending
from claims_agent.reply.brief import ReplyPart
from claims_agent.states import State

REP = stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472", caller_name="David Chen")
THIRD = dict(gave_identity_detail=True, caller_role="third_party", relationship="son")


@pytest.fixture
async def consent_deps(load_dataset):
    """extra_consent data: its `denied` script answers pending, then denied."""
    return make_deps(await load_dataset("extra_consent"))


async def test_a_step_that_ends_the_conversation_owns_the_whole_reply(consent_deps):
    cid = await new_conversation(consent_deps, consent_script="denied")
    assert (await say(consent_deps, cid, "I'm David, her son", make_form(**THIRD), REP)).state == State.AWAIT_CONSENT
    # the refusal comes back in a turn whose form also holds an unsure request for a person
    r = await say(consent_deps, cid, "check again, and maybe a person?",
                  make_form(consent_recheck=True, caller_role="third_party", wants_person=0.5))
    assert r.state == State.CLOSED and instructions(r) == ["consent_denied"]
    assert await memory(consent_deps, cid, "pending") is None


WHICH = Pending(kind="which_claim", label="which claim?", yes_no=False)
WHICH_AGAIN = Pending(kind="which_claim_again", label="which of these claims?", yes_no=False)


@pytest.fixture
def asking_step(monkeypatch):
    """RESOLVE_INTENT replaced by a test step: "list" asks which claim (a question that is not yes/no), "relist" asks
    a new one, anything else asks nothing."""
    async def step(ctx):
        if ctx.message in ("list", "relist"):
            pending = WHICH if ctx.message == "list" else WHICH_AGAIN
            return StepOutcome(State.RESOLVE_INTENT, [ReplyPart("which_claim", plain="Which one?", pending=pending)])
        return StepOutcome(State.RESOLVE_INTENT, [ReplyPart("anything_else", plain="Anything else?")])
    monkeypatch.setitem(engine.STEPS, State.RESOLVE_INTENT, step)


async def test_a_question_that_is_not_yes_no_lives_for_one_reply(deps, verified_conv, asking_step):
    await say(deps, verified_conv, "list")
    assert await memory(deps, verified_conv, "pending") == WHICH
    asked = len(deps.filler.requests)
    r = await say(deps, verified_conv, "2")  # one character: readable because a question is pending
    assert len(deps.filler.requests) == asked + 1 and deps.filler.requests[-1].pending == WHICH.label
    assert "ask_again" not in instructions(r) and await memory(deps, verified_conv, "pending") is None
    r = await say(deps, verified_conv, "3")  # nothing pending any more: unreadable (and the third stalled turn)
    assert instructions(r)[0] == "ask_again" and len(deps.filler.requests) == asked + 1


async def test_a_new_question_replaces_the_pending_one(deps, verified_conv, asking_step):
    await say(deps, verified_conv, "list")
    await say(deps, verified_conv, "relist")
    assert await memory(deps, verified_conv, "pending") == WHICH_AGAIN


async def test_an_unreadable_reply_keeps_the_question_that_is_not_yes_no(deps, verified_conv, asking_step):
    await say(deps, verified_conv, "list")
    r = await say(deps, verified_conv, "?")
    assert instructions(r) == ["ask_again"] and await memory(deps, verified_conv, "pending") == WHICH
    assert "ask_again" not in instructions(await say(deps, verified_conv, "1"))


async def test_an_unanswered_yes_no_offer_still_stays(deps, verified_conv, asking_step):
    await say(deps, verified_conv, "maybe a person?", make_form(wants_person=0.5))
    await say(deps, verified_conv, "hmm")
    assert (await memory(deps, verified_conv, "pending")).kind == "offer_person"


async def _show_healthcare_list(deps, cid):
    r = await say(deps, cid, "my healthcare claim", make_form(mentions_claim=True, how_many="one",
                                                              claim_types={"healthcare": 0.95}))
    assert "which_claim" in instructions(r)
    shown = await memory(deps, cid, "focus.last_list")
    assert len(shown) == 2
    return shown


async def test_the_form_is_told_which_claims_were_listed(deps, verified_conv):
    shown = await _show_healthcare_list(deps, verified_conv)
    await say(deps, verified_conv, "the newer one")
    listed = deps.filler.requests[-1].listed_claims
    assert [(c.case_id, c.case_type) for c in listed] == [(case_id, "healthcare") for case_id in shown]


async def test_no_listed_claims_before_verification_or_after_a_reset(deps, conv, verified_conv):
    await say(deps, conv, "hello there")
    assert deps.filler.requests[-1].listed_claims == ()
    await _show_healthcare_list(deps, verified_conv)
    await say(deps, verified_conv, "what about my husband's policy?", make_form(other_person_policy=True))
    await say(deps, verified_conv, "hello again")
    assert deps.filler.requests[-1].listed_claims == () and not deps.filler.requests[-1].verified
