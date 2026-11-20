"""A representative asks about someone else's policy by definition: that is the policyholder they act for, never a
different person. A different person is someone other than the customer the conversation is verified for, or is
being verified against; only then are the conversation's identity and claims forgotten."""

from helpers import (BRIEF_MSG, BRIEF_SAID, GAVE, focus, make_form, memory, new_conversation, say, stated,
                     verified_party)

from claims_agent.states import State

REP = stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472", caller_name="David Chen")
THIRD = dict(gave_identity_detail=True, caller_role="third_party", relationship="son")
SOMEONE_ELSES = dict(other_person_policy=1.0)  # how a real form reads every message of a representative


async def test_a_representative_read_as_asking_about_someone_else_on_every_message_gets_through(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, "I'm David Chen, calling for my mother Margaret Chen", make_form(**THIRD, **SOMEONE_ELSES), REP)
    assert r.state == State.AWAIT_CONSENT
    r = await say(deps, cid, "Hold on, I'm texting her now.", make_form(caller_role="third_party", **SOMEONE_ELSES))
    assert r.state == State.AWAIT_CONSENT and (await memory(deps, cid, "identity.held"))
    r = await say(deps, cid, "She says she approved it. Can you check again?",
                  make_form(consent_recheck=True, caller_role="third_party", **SOMEONE_ELSES))
    assert await verified_party(deps, cid) == "P9" and r.state != State.VERIFY_ID
    r = await say(deps, cid, "Why was her healthcare claim denied?",
                  make_form(mentions_claim=True, claim_types={"healthcare": 0.95}, claim_statuses={"denied": 0.95},
                            need="denial_question", how_many="one", **SOMEONE_ELSES))
    assert await verified_party(deps, cid) == "P9" and await focus(deps, cid) == ["CL-2048"]
    assert not any(f.get("outcome") == "reset_identity" for f in r.inspector["fired"])


async def test_a_representative_naming_her_by_first_name_is_still_asking_about_her(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "I'm David Chen, calling for my mother Margaret Chen", make_form(**THIRD, **SOMEONE_ELSES), REP)
    r = await say(deps, cid, "It's Margaret's policy, yes", make_form(caller_role="third_party", **SOMEONE_ELSES),
                  stated(policyholder_name="Margaret"))
    assert r.state == State.AWAIT_CONSENT


async def test_a_representative_who_names_a_different_policyholder_starts_again(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "I'm David Chen, calling for my mother Margaret Chen", make_form(**THIRD, **SOMEONE_ELSES), REP)
    r = await say(deps, cid, "Actually it's about my father Robert Chen's policy",
                  make_form(caller_role="third_party", **SOMEONE_ELSES), stated(policyholder_name="Robert Chen"))
    assert r.state == State.VERIFY_ID and await memory(deps, cid, "identity.held") == {}


async def test_a_verified_representative_who_names_a_different_policyholder_starts_again(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "I'm David Chen, calling for my mother Margaret Chen", make_form(**THIRD), REP)
    await say(deps, cid, "check again", make_form(consent_recheck=True, caller_role="third_party"))
    assert await verified_party(deps, cid) == "P9"
    r = await say(deps, cid, "And what about Robert Chen's policy?", make_form(**SOMEONE_ELSES),
                  stated(policyholder_name="Robert Chen"))
    assert r.state == State.VERIFY_ID and await verified_party(deps, cid) is None


async def test_a_policyholder_asking_about_someone_elses_policy_still_starts_again(deps, conv):
    await say(deps, conv, BRIEF_MSG, make_form(**GAVE), BRIEF_SAID)
    assert await verified_party(deps, conv) == "P9"
    r = await say(deps, conv, "What about my husband's policy?", make_form(**SOMEONE_ELSES))
    assert r.state == State.VERIFY_ID and await verified_party(deps, conv) is None


async def test_a_representative_read_as_moving_to_another_relative_starts_again(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "I'm David Chen, calling for my mother Margaret Chen", make_form(**THIRD), REP)
    r = await say(deps, cid, "What about my dad's policy?",
                  make_form(caller_role="third_party", other_than_represented=0.9, **SOMEONE_ELSES))
    assert r.state == State.VERIFY_ID and await memory(deps, cid, "identity.held") == {}


async def test_the_question_about_another_person_than_the_one_represented_is_asked_only_of_a_representative(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, BRIEF_MSG, make_form(**GAVE), BRIEF_SAID)
    rep = await new_conversation(deps)
    await say(deps, rep, "I'm David Chen, calling for my mother Margaret Chen", make_form(**THIRD), REP)
    await say(deps, rep, "Hold on", make_form(caller_role="third_party"))
    flags = [r.acting_for_policyholder for r in deps.filler.requests]
    assert flags == [False, False, True]
