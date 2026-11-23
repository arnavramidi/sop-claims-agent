"""After verification, a message that leaves the script: an off-topic question is declined once and the caller is
brought back to where the conversation was; a request for the summary email is acknowledged and kept; skipping a step
is not answered as if identity were at stake."""

import pytest
from helpers import GAVE, instructions, make_form, memory, new_conversation, parts, say, stated

from claims_agent.form.fields import Pick
from claims_agent.states import State

ALTERNATIVES_OFFER = "offer_followup:CL-2048:missing_required_material_alternatives"
OFF_TOPIC = make_form(about="off_topic", answer_wanted="something_else")


async def outbox_count(conn, cid):
    cur = await conn.execute("SELECT count(*) FROM outbox WHERE conversation_id = %s", [cid])
    return (await cur.fetchone())[0]


@pytest.fixture
async def margaret(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
                  stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert r.state == State.RESOLVE_INTENT
    return cid


@pytest.fixture
async def denial_case(deps, margaret):
    """CL-2048's denial opening given; the alternatives follow-up is offered and pending."""
    r = await say(deps, margaret, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                  stated(claim_number="CL-2048"))
    assert r.state == State.PROCESS_CASE and (await memory(deps, margaret, "pending")).kind == ALTERNATIVES_OFFER
    return margaret


# ---- off-topic ------------------------------------------------------------------------------------------------

async def test_an_off_topic_question_is_declined_and_the_pending_offer_asked_again(deps, denial_case):
    r = await say(deps, denial_case, "What is RL?", OFF_TOPIC)
    assert instructions(r) == ["decline_off_topic", "offer_followup"]
    assert r.state == State.PROCESS_CASE and (await memory(deps, denial_case, "pending")).kind == ALTERNATIVES_OFFER
    assert "not on file" not in r.reply and "?" not in parts(r)[0].plain


async def test_an_off_topic_question_with_no_answer_pick_asks_one_question(deps, denial_case):
    r = await say(deps, denial_case, "What's the capital of France?", make_form(about="off_topic"))
    asking = [p for p in parts(r) if p.pending is not None]
    assert len(asking) == 1 and asking[0].pending.kind == ALTERNATIVES_OFFER


async def test_a_person_is_offered_only_when_off_topic_questions_repeat(deps, denial_case):
    for _ in range(deps.settings.limits.stalled_turns - 1):
        r = await say(deps, denial_case, "Tell me a joke.", OFF_TOPIC)
        assert "offer_person" not in instructions(r) and r.state == State.PROCESS_CASE
    r = await say(deps, denial_case, "Tell me a joke.", OFF_TOPIC)
    assert "offer_person" in instructions(r) and r.state == State.PROCESS_CASE


async def test_an_off_topic_question_while_choosing_a_claim_asks_the_same_choice_again(deps, margaret):
    first = await say(deps, margaret, "my healthcare claim from January",
                      make_form(how_many="one", claim_types={"healthcare": 0.95}, months={1: 0.95}, no_year_said=True))
    asked = next(p for p in parts(first) if p.instruction == "which_claim")
    r = await say(deps, margaret, "What is RL?", OFF_TOPIC)
    assert instructions(r) == ["decline_off_topic", "which_claim"]
    assert next(p for p in parts(r) if p.instruction == "which_claim") == asked
    assert await memory(deps, margaret, "focus.last_list") == ["CL-2048", "CL-2011"]


async def test_an_off_topic_question_during_the_email_offer_asks_the_offer_again(deps, denial_case, owner_conn):
    await say(deps, denial_case, "that's everything", make_form(done=True))
    r = await say(deps, denial_case, "What is RL?", make_form(about="off_topic", email_offer_reply="something_else"))
    assert instructions(r) == ["decline_off_topic", "offer_email"] and r.state == State.POST_PROCESS
    assert await outbox_count(owner_conn, denial_case) == 0


async def test_a_question_about_the_claim_is_still_answered_when_the_message_also_strays(deps, denial_case):
    r = await say(deps, denial_case, "When is the deadline, and what is RL?",
                  make_form(about="off_topic", answer_wanted="appeal_deadline"))
    assert "decline_off_topic" in instructions(r) and "answer" in instructions(r)


# ---- skipping a step after verification ----------------------------------------------------------------------

async def test_after_verification_skipping_a_step_is_not_told_it_protects_identity(deps, denial_case):
    r = await say(deps, denial_case, "can we skip this and just give me the deadline?",
                  make_form(skips_step=True, answer_wanted="appeal_deadline"))
    assert "explain_step" not in instructions(r) and "answer" in instructions(r)


async def test_before_verification_skipping_a_step_is_still_explained(deps, conv):
    r = await say(deps, conv, "can we skip this?", make_form(skips_step=True))
    assert "explain_step" in instructions(r)


# ---- "email me a summary" mid-conversation -------------------------------------------------------------------

async def test_asking_for_the_summary_mid_conversation_is_acknowledged_and_sent_at_wrap_up_without_asking(
        deps, denial_case, owner_conn):
    r = await say(deps, denial_case, "Can you email me a summary of this?",
                  make_form(wants_email_summary=True, answer_wanted="something_else"))
    assert instructions(r) == ["email_at_wrap_up", "offer_followup"]
    assert "not on file" not in r.reply and (await memory(deps, denial_case, "pending")).kind == ALTERNATIVES_OFFER
    assert await outbox_count(owner_conn, denial_case) == 0
    r = await say(deps, denial_case, "that's all, thanks", make_form(done=True))
    assert r.state == State.DONE and "email_sent" in instructions(r) and "offer_email" not in instructions(r)
    assert await outbox_count(owner_conn, denial_case) == 1


async def test_a_summary_of_this_claim_is_an_email_request_not_a_question_about_the_claim(deps, denial_case,
                                                                                        owner_conn):
    # "of this" names the claim being discussed: the form says the message mentions a claim, yet it asks nothing of it
    r = await say(deps, denial_case, "Can you email me a summary of this?",
                  make_form(wants_email_summary=True, mentions_claim=0.98, answer_wanted="something_else"))
    assert "email_at_wrap_up" in instructions(r)
    assert "not_on_file" not in instructions(r) and "offer_person" not in instructions(r)
    assert await outbox_count(owner_conn, denial_case) == 0


async def test_an_off_topic_question_the_form_says_mentions_a_claim_is_still_declined(deps, denial_case):
    r = await say(deps, denial_case, "OK, what's the capital of France then?",
                  make_form(about="off_topic", mentions_claim=0.83, answer_wanted="something_else"))
    assert instructions(r) == ["decline_off_topic", "offer_followup"]
    assert "not on file" not in r.reply


async def test_an_off_topic_question_read_with_a_vague_claim_need_is_still_declined(deps, denial_case):
    # in a conversation about a claim, the form also reads a general claim question into "What is RL?"
    r = await say(deps, denial_case, "What is RL?",
                  make_form(about="off_topic", need=Pick(value="general_claim_question", p=0.64),
                            answer_wanted="something_else"))
    assert instructions(r) == ["decline_off_topic", "offer_followup"]
    assert "not on file" not in r.reply


async def test_a_summary_of_this_read_as_meaning_one_claim_is_still_only_an_email_request(deps, denial_case):
    r = await say(deps, denial_case, "Can you email me a summary of this?",
                  make_form(wants_email_summary=True, how_many=Pick(value="one", p=0.71),
                            answer_wanted=Pick(value="something_else", p=0.94)))
    assert "email_at_wrap_up" in instructions(r)
    assert "not_on_file" not in instructions(r) and "offer_person" not in instructions(r)


async def test_off_topic_questions_read_with_a_need_still_offer_a_person_when_they_repeat(deps, denial_case):
    vague = make_form(about="off_topic", need=Pick(value="general_claim_question", p=0.69),
                      answer_wanted="something_else")
    for _ in range(deps.settings.limits.stalled_turns - 1):
        r = await say(deps, denial_case, "What is RL?", vague)
        assert "offer_person" not in instructions(r)
    assert await memory(deps, denial_case, "hints.need") is None
    r = await say(deps, denial_case, "Tell me a joke.", vague)
    assert "offer_person" in instructions(r) and r.state == State.PROCESS_CASE


async def test_a_summary_asked_for_with_a_question_gets_both(deps, denial_case):
    r = await say(deps, denial_case, "When is the deadline? And email me a summary.",
                  make_form(wants_email_summary=True, answer_wanted="appeal_deadline"))
    got = instructions(r)
    assert "answer" in got and "email_at_wrap_up" in got and got.index("email_at_wrap_up") < len(got) - 1


async def test_an_unsure_request_for_the_summary_is_confirmed_at_wrap_up(deps, denial_case, owner_conn):
    r = await say(deps, denial_case, "maybe email me?", make_form(wants_email_summary=0.6))
    assert "email_at_wrap_up" in instructions(r) and "not_on_file" not in instructions(r)
    r = await say(deps, denial_case, "that's all", make_form(done=True))
    assert "offer_email" in instructions(r) and await outbox_count(owner_conn, denial_case) == 0
