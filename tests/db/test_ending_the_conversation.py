"""Ending a conversation after the email step: a finished conversation stays finished for pleasantries, the email is
offered once per wrap-up and sent once unless asked for again after new answers, and a real new question reopens."""

from dataclasses import replace

import pytest
from helpers import GAVE, facts, instructions, make_form, memory, new_conversation, parts, say, stated

from claims_agent.form.fields import Pick
from claims_agent.states import State

SEND = make_form(email_offer_reply=Pick(value="send", p=0.95))
DONT_SEND = make_form(email_offer_reply="dont_send")
AUTO_STATUS = make_form(claim_types={"auto": 0.95}, how_many="one", need="status_inquiry", mentions_claim=True,
                        search_relation="new_search")


async def _verified(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
                  stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert r.state == State.RESOLVE_INTENT
    return cid


async def outbox_count(conn, cid):
    cur = await conn.execute("SELECT count(*) FROM outbox WHERE conversation_id = %s", [cid])
    return (await cur.fetchone())[0]


@pytest.fixture
async def wrapping(deps):
    """P9, CL-2048 discussed, the caller done: the email offer is pending."""
    cid = await _verified(deps)
    r = await say(deps, cid, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                  stated(claim_number="CL-2048"))
    assert r.state == State.PROCESS_CASE
    r = await say(deps, cid, "that's everything", make_form(done=True))
    assert "offer_email" in instructions(r) and r.state == State.POST_PROCESS
    return cid


@pytest.fixture
async def sent(deps, wrapping):
    r = await say(deps, wrapping, "yes please", SEND)
    assert r.state == State.DONE and "email_sent" in instructions(r)
    return wrapping


@pytest.fixture
async def declined(deps, wrapping):
    r = await say(deps, wrapping, "no thanks", DONT_SEND)
    assert r.state == State.DONE and "email_skipped" in instructions(r)
    return wrapping


# ---- pleasantries after the end -------------------------------------------------------------------------------

async def test_thanks_after_the_summary_was_sent_is_a_goodbye_and_nothing_is_offered(deps, sent, owner_conn):
    r = await say(deps, sent, "Thanks!", make_form(done=True))
    assert r.state == State.DONE and instructions(r) == ["goodbye"]
    assert await outbox_count(owner_conn, sent) == 1 and await memory(deps, sent, "pending") is None


async def test_a_thanks_not_read_as_done_is_a_goodbye_too(deps, sent):
    r = await say(deps, sent, "cheers", make_form(about="conversation"))
    assert r.state == State.DONE and instructions(r) == ["goodbye"]


async def test_the_skip_reply_says_goodbye_and_asks_nothing(deps, declined):
    assert await memory(deps, declined, "pending") is None
    r = await say(deps, declined, "ok", make_form())
    assert r.state == State.DONE and instructions(r) == ["goodbye"]


async def test_no_thats_all_after_declining_never_offers_the_email_again(deps, declined, owner_conn):
    for text in ("No, that's all.", "no", "bye then"):
        r = await say(deps, declined, text, make_form(done=True, pending_reply="no"))
        assert r.state == State.DONE and instructions(r) == ["goodbye"]
    assert await outbox_count(owner_conn, declined) == 0


async def test_pleasantries_after_the_end_never_offer_a_person(deps, sent):
    for _ in range(deps.settings.limits.stalled_turns + 1):
        r = await say(deps, sent, "thank you so much", make_form(about="conversation"))
        assert "offer_person" not in instructions(r)
    assert await memory(deps, sent, "counters.stalled_in_a_row") == 0


async def test_an_unreadable_message_after_the_end_is_a_goodbye(deps, sent):
    r = await say(deps, sent, "👍", make_form())
    assert r.state == State.DONE and instructions(r) == ["goodbye"]


async def test_a_person_can_still_be_asked_for_after_the_end(deps, sent):
    assert (await say(deps, sent, "let me talk to a person", make_form(wants_person=True))).state == State.ESCALATED


# ---- the email once ------------------------------------------------------------------------------------------

async def test_asking_for_the_summary_again_with_nothing_new_does_not_send_it_twice(deps, sent, owner_conn):
    r = await say(deps, sent, "can you send that summary again?", make_form(wants_email_summary=True))
    assert r.state == State.DONE and "email_already_sent" in instructions(r) and "offer_email" not in instructions(r)
    assert await outbox_count(owner_conn, sent) == 1


async def test_a_summary_asked_for_after_declining_it_is_sent(deps, declined, owner_conn):
    r = await say(deps, declined, "actually, yes, email me the summary", make_form(wants_email_summary=True))
    assert r.state == State.DONE and "email_sent" in instructions(r)
    assert await outbox_count(owner_conn, declined) == 1


async def test_a_new_question_after_the_goodbye_reopens_without_offering_the_email_again(deps, declined, owner_conn):
    r = await say(deps, declined, "actually, what's the status of my auto claim?", AUTO_STATUS)
    assert r.state == State.PROCESS_CASE and any(f.claim == "CL-2102" for f in facts(r))
    assert "offer_email" not in instructions(r)
    r = await say(deps, declined, "that's all, thanks", make_form(done=True))
    assert r.state == State.DONE and "offer_email" not in instructions(r) and "goodbye" in instructions(r)
    assert await outbox_count(owner_conn, declined) == 0


async def test_after_new_answers_an_explicit_request_sends_the_summary_again(deps, sent, owner_conn):
    await say(deps, sent, "one more thing, my auto claim?", AUTO_STATUS)
    r = await say(deps, sent, "please email me an updated summary", make_form(wants_email_summary=True))
    assert "email_at_wrap_up" in instructions(r) and await outbox_count(owner_conn, sent) == 1
    r = await say(deps, sent, "that's all", make_form(done=True))
    assert "email_sent" in instructions(r) and "offer_email" not in instructions(r)
    assert await outbox_count(owner_conn, sent) == 2


# ---- the send reading below the bar ---------------------------------------------------------------------------

UNSURE_SEND = make_form(email_offer_reply=Pick(value="send", p=0.62))


async def test_a_send_read_below_the_bar_is_asked_again_only_as_often_as_the_settings_allow(deps, wrapping, owner_conn):
    for _ in range(deps.settings.limits.unclear_email_replies - 1):
        r = await say(deps, wrapping, "sure i guess?", UNSURE_SEND)
        assert r.state == State.POST_PROCESS and "offer_email" in instructions(r)
    r = await say(deps, wrapping, "sure i guess?", UNSURE_SEND)
    assert r.state == State.DONE and "?" not in parts(r)[-1].plain
    assert await outbox_count(owner_conn, wrapping) == 0


async def test_the_limit_on_unclear_replies_is_read_from_the_settings(deps, wrapping, owner_conn):
    three = replace(deps, settings=deps.settings.model_copy(update={
        "limits": deps.settings.limits.model_copy(update={"unclear_email_replies": 3})}))
    for _ in range(2):
        assert (await say(three, wrapping, "hmm", make_form(email_offer_reply="unclear"))).state == State.POST_PROCESS
    assert (await say(three, wrapping, "eh", make_form(email_offer_reply="unclear"))).state == State.DONE
    assert await outbox_count(owner_conn, wrapping) == 0


# ---- the offer answered by a message that also says the caller is done ------------------------------------------

async def test_a_no_to_the_offer_that_also_says_done_ends_without_offering_again(deps, wrapping, owner_conn):
    r = await say(deps, wrapping, "No thanks, don't send anything.", make_form(done=True, email_offer_reply="dont_send"))
    assert r.state == State.DONE and "email_skipped" in instructions(r) and "offer_email" not in instructions(r)
    assert await outbox_count(owner_conn, wrapping) == 0


async def test_a_yes_to_the_offer_that_also_says_done_sends_the_summary(deps, wrapping, owner_conn):
    r = await say(deps, wrapping, "Yes send it, that's all.",
                  make_form(done=True, email_offer_reply=Pick(value="send", p=0.95)))
    assert r.state == State.DONE and "email_sent" in instructions(r)
    assert await outbox_count(owner_conn, wrapping) == 1
