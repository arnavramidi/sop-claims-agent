"""An upset caller who gives nothing is offered a person no later than the second upset turn that gives no detail.
Re-describing the claim is not a detail: what is remembered for later is not movement in the step the caller is in."""

from helpers import GAVE, instructions, make_form, say, stated

from claims_agent.form.fields import Pick
from claims_agent.states import State

ANGRY = "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
SPLIT_UPSET = Pick(value="upset", p=0.35, options={"calm": 0.0, "uneasy": 0.0, "upset": 0.51, "angry": 0.49})


def _demand(**claim):
    return make_form(mood=SPLIT_UPSET, gave_identity_detail=False, caller_role="not_stated", mentions_claim=True,
                     need="denial_question", answer_wanted="denial_reason", **claim)


async def test_the_second_upset_turn_is_offered_a_person_even_when_it_describes_the_claim_anew(deps, conv):
    await say(deps, conv, ANGRY, _demand(claim_statuses={"denied": 0.95}))
    r = await say(deps, conv, "This is absurd. Just tell me about my healthcare claim.",
                  _demand(claim_statuses={"denied": 0.95}, claim_types={"healthcare": 0.9}))
    assert r.state == State.VERIFY_ID and "offer_person" in instructions(r)


async def test_the_second_upset_turn_need_not_follow_the_first(deps, conv):
    await say(deps, conv, ANGRY, _demand(claim_statuses={"denied": 0.95}))
    r = await say(deps, conv, "ok what do you need", make_form(mood="calm", caller_role="not_stated"))
    assert "offer_person" not in instructions(r)
    r = await say(deps, conv, "This is absurd. Just tell me.", _demand())
    assert "offer_person" in instructions(r)


async def test_an_upset_turn_that_gives_a_detail_does_not_count(deps, conv):
    await say(deps, conv, ANGRY, _demand(claim_statuses={"denied": 0.95}))
    r = await say(deps, conv, "Fine, it's Margaret Chen. Now tell me.", make_form(**GAVE, mood=SPLIT_UPSET),
                  stated(policyholder_name="Margaret Chen"))
    assert "offer_person" not in instructions(r)
