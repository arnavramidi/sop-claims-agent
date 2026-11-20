"""When it is unclear whether a representative has moved to someone other than the policyholder they act for, the
agent asks ("Is this still about the policyholder you're calling for?"), naming no one and telling no claim, instead
of carrying on with the same policyholder; a clear "no" starts verification again."""

from helpers import facts, instructions, make_form, memory, new_conversation, say, stated, verified_party

from claims_agent.states import State

REP = stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472", caller_name="David Chen")
NO_POLICYHOLDER_NAME = stated(date_of_birth=["1985-03-15"], id_digits="4472", phone="+16505212836",
                              caller_name="David Chen")
THIRD = dict(gave_identity_detail=True, caller_role="third_party", relationship="son")


async def _approved(deps, said=REP):
    cid = await new_conversation(deps)
    await say(deps, cid, "I'm David Chen, calling for my mother", make_form(**THIRD), said)
    await say(deps, cid, "check again", make_form(consent_recheck=True, caller_role="third_party"))
    assert await verified_party(deps, cid) == "P9"
    return cid


QUESTION = make_form(mentions_claim=True, how_many="one", claim_statuses={"denied": 0.95}, need="denial_question",
                     other_person_policy=1.0, other_than_represented=0.5)


async def test_an_unsure_reading_asks_whether_it_is_still_the_same_policyholder_and_tells_nothing(deps):
    cid = await _approved(deps)
    r = await say(deps, cid, "And what about the denied claim?", QUESTION)
    assert "ask_same_policyholder" in instructions(r) and not any(f.claim for f in facts(r))
    assert (await memory(deps, cid, "pending")).kind == "same_policyholder"
    r = await say(deps, cid, "No, it's my father's", make_form(pending_reply="no", other_person_policy=1.0))
    assert r.state == State.VERIFY_ID and await verified_party(deps, cid) is None


async def test_a_yes_carries_on_with_the_same_policyholder(deps):
    cid = await _approved(deps)
    await say(deps, cid, "And what about the denied claim?", QUESTION)
    r = await say(deps, cid, "Yes, still my mother", make_form(pending_reply="yes", other_person_policy=1.0))
    assert r.state != State.VERIFY_ID and await verified_party(deps, cid) == "P9"
    assert await memory(deps, cid, "pending") is None or (await memory(deps, cid, "pending")).kind != "same_policyholder"


async def test_a_name_said_when_no_policyholder_name_is_held_is_asked_about(deps):
    """The representative verified with the date of birth, ID and phone: no policyholder name to compare with."""
    cid = await _approved(deps, NO_POLICYHOLDER_NAME)
    r = await say(deps, cid, "What about Robert Chen's claim?",
                  make_form(mentions_claim=True, other_person_policy=1.0, other_than_represented=0.1),
                  stated(policyholder_name="Robert Chen"))
    assert "ask_same_policyholder" in instructions(r) and not any(f.claim for f in facts(r))
