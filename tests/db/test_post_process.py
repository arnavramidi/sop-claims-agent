"""Whole turns through POST_PROCESS: the email offer, the send, and the way back to a question."""

import pytest
from helpers import (GAVE, facts, instructions, make_deps, make_form, memory, new_conversation, say, stated)

from claims_agent.form.fields import Pick
from claims_agent.states import State

REP = stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472", caller_name="David Chen")
THIRD = dict(gave_identity_detail=True, caller_role="third_party", relationship="son")
SEND = make_form(email_offer_reply=Pick(value="send", p=0.95))


async def _verified(deps, name, dob, last4):
    cid = await new_conversation(deps)
    r = await say(deps, cid, f"{name}, born {dob}, ID ends {last4}", make_form(**GAVE),
                  stated(policyholder_name=name, date_of_birth=[dob], id_digits=last4))
    assert r.state == State.RESOLVE_INTENT
    return cid


async def _discuss(deps, cid, claim_id="CL-2048", need="denial_question", **form):
    r = await say(deps, cid, f"about {claim_id}", make_form(quoted_claim_number=True, need=need, **form),
                  stated(claim_number=claim_id))
    assert r.state == State.PROCESS_CASE
    return r


async def outbox_rows(conn, cid):
    cur = await conn.execute("SELECT party_id, recipient, subject, body FROM outbox WHERE conversation_id = %s "
                             "ORDER BY id", [cid])
    return await cur.fetchall()


async def outbox_count(conn, cid):
    return len(await outbox_rows(conn, cid))


async def outbox_row(conn, cid):
    from types import SimpleNamespace
    rows = await outbox_rows(conn, cid)
    assert len(rows) == 1
    return SimpleNamespace(**dict(zip(("party_id", "recipient", "subject", "body"), rows[0])))


@pytest.fixture
async def margaret(deps):
    return await _verified(deps, "Margaret Chen", "1985-03-15", "4472")


@pytest.fixture
async def wrapping(deps, margaret):
    """P9, CL-2048 discussed, the caller done: the email offer is pending."""
    await _discuss(deps, margaret)
    r = await say(deps, margaret, "that's everything", make_form(done=True))
    assert "offer_email" in instructions(r) and r.state == State.POST_PROCESS
    assert (await memory(deps, margaret, "pending")).kind == "offer_email"
    return margaret


@pytest.fixture
async def verified_nothing_discussed(margaret):
    return margaret


@pytest.fixture
async def done_conv(deps, wrapping):
    r = await say(deps, wrapping, "no thanks", make_form(email_offer_reply="dont_send"))
    assert r.state == State.DONE
    return wrapping


@pytest.fixture
async def case_with_email_hint(deps, margaret):
    await say(deps, margaret, "email me a summary of why CL-2048 was denied",
              make_form(quoted_claim_number=True, need="denial_question", wants_email_summary=True),
              stated(claim_number="CL-2048"))
    assert await memory(deps, margaret, "hints.wants_email") is True
    return margaret


@pytest.fixture
async def rep_wrapping(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, "I'm David, calling for my mother", make_form(**THIRD), REP)
    assert r.state == State.AWAIT_CONSENT
    r = await say(deps, cid, "check again", make_form(consent_recheck=True, caller_role="third_party"))
    assert r.state == State.RESOLVE_INTENT
    await _discuss(deps, cid, caller_role="third_party")
    r = await say(deps, cid, "that's everything", make_form(done=True, caller_role="third_party"))
    assert "offer_email" in instructions(r)
    return cid


@pytest.fixture
async def many_deps(load_dataset):
    return make_deps(await load_dataset("many_claims"))


@pytest.fixture
async def omar(many_deps):
    return await _verified(many_deps, "Omar Reyes", "1978-04-09", "2210")


@pytest.fixture
async def denial_case(deps, margaret):
    await _discuss(deps, margaret)
    return margaret


# ---- core behaviour ----------------------------------------------------------------------------------------

async def test_a_clear_yes_stores_the_summary_for_the_policyholder(deps, wrapping, owner_conn):
    r = await say(deps, wrapping, "yes, send it", SEND)
    row = await outbox_row(owner_conn, wrapping)
    assert r.state == State.DONE and row.party_id == "P9" and row.recipient == "margaret@email.com"
    assert "email_sent" in instructions(r) and "CL-2048" in row.body


async def test_a_clear_no_sends_nothing(deps, wrapping, owner_conn):
    r = await say(deps, wrapping, "no thanks", make_form(email_offer_reply="dont_send"))
    assert r.state == State.DONE and await outbox_count(owner_conn, wrapping) == 0
    assert "email_skipped" in instructions(r)


async def test_unclear_twice_sends_nothing(deps, wrapping, owner_conn):
    assert (await say(deps, wrapping, "hmm", make_form(email_offer_reply="unclear"))).state == State.POST_PROCESS
    r = await say(deps, wrapping, "eh", make_form(email_offer_reply="unclear"))
    assert r.state == State.DONE and await outbox_count(owner_conn, wrapping) == 0


async def test_a_low_probability_send_is_asked_again(deps, wrapping, owner_conn):
    r = await say(deps, wrapping, "sure i guess?", make_form(email_offer_reply=Pick(value="send", p=0.62)))
    assert r.state == State.POST_PROCESS and await outbox_count(owner_conn, wrapping) == 0
    assert "offer_email" in instructions(r)


async def test_a_different_address_is_refused_and_a_person_offered(deps, wrapping):
    r = await say(deps, wrapping, "send it to my other address", make_form(email_offer_reply="different_address"))
    assert "different_address_not_possible" in instructions(r)
    assert (await memory(deps, wrapping, "pending")).kind == "offer_person"


async def test_the_address_is_never_read_out(deps, wrapping):
    r = await say(deps, wrapping, "which address is that?", make_form(email_offer_reply="which_address"))
    assert "email_address_not_read_out" in instructions(r) and "@" not in r.reply and r.state == State.POST_PROCESS


async def test_a_question_is_answered_and_the_offer_repeated(deps, wrapping):
    r = await say(deps, wrapping, "wait, what was that deadline again?",
                  make_form(email_offer_reply="something_else", answer_wanted="appeal_deadline"))
    assert [p for p in instructions(r) if p in ("answer", "offer_email")] == ["answer", "offer_email"]
    assert r.state == State.POST_PROCESS


async def test_nothing_discussed_means_no_offer(deps, verified_nothing_discussed):
    r = await say(deps, verified_nothing_discussed, "that's all, bye", make_form(done=True))
    assert "offer_email" not in instructions(r) and r.state == State.DONE and "goodbye" in instructions(r)


async def test_a_message_after_done_reopens_while_verified(deps, done_conv):
    r = await say(deps, done_conv, "one more thing, my auto claim?",
                  make_form(claim_types={"auto": 0.95}, how_many="one", need="status_inquiry"))
    assert r.state == State.PROCESS_CASE


async def test_an_earlier_request_for_an_email_is_sent_at_wrap_up_without_asking_again(deps, case_with_email_hint,
                                                                                          owner_conn):
    r = await say(deps, case_with_email_hint, "that's everything", make_form(done=True))
    assert "email_sent" in instructions(r) and "offer_email" not in instructions(r)
    assert await outbox_count(owner_conn, case_with_email_hint) == 1


async def test_a_representatives_summary_goes_to_the_policyholder(deps, rep_wrapping, owner_conn):
    assert "the policyholder's email" in deps.wording.briefs[-1].plain_text()
    await say(deps, rep_wrapping, "yes", make_form(email_offer_reply=Pick(value="send", p=0.95),
                                                   caller_role="third_party"))
    row = await outbox_row(owner_conn, rep_wrapping)
    assert row.party_id == "P9" and row.recipient == "margaret@email.com"


async def test_the_full_list_asked_for_after_an_overview_is_in_the_email(many_deps, omar, owner_conn):
    await say(many_deps, omar, "all my claims", make_form(how_many="all"))
    await say(many_deps, omar, "just email me the full list",
              make_form(wants_email_summary=True, how_many="same_as_before"))
    await say(many_deps, omar, "that's all", make_form(done=True))
    await say(many_deps, omar, "yes", SEND)
    body = (await outbox_row(owner_conn, omar)).body
    assert "The full list you asked for" in body and all(f"CL-400{i}" in body for i in range(1, 9))


async def test_a_handed_over_conversation_gets_no_email_offer(deps, denial_case):
    r = await say(deps, denial_case, "let me talk to a person", make_form(wants_person=True))
    assert "offer_email" not in instructions(r)


# ---- more -----------------------------------------------------------------------------------------------------

async def test_the_address_never_reaches_a_model_a_reply_or_the_record(deps, wrapping):
    r = await say(deps, wrapping, "yes, send it", SEND)
    blob = str(deps.wording.briefs) + r.reply + str(r.inspector)
    assert "margaret@email.com" not in blob and "@" not in r.reply


async def test_the_summary_is_sent_once(deps, wrapping, owner_conn):
    await say(deps, wrapping, "yes", SEND)
    await say(deps, wrapping, "yes again", SEND)  # after the end, a "yes" asks nothing new: a goodbye, never a second send
    assert await outbox_count(owner_conn, wrapping) == 1


async def test_a_question_about_another_claim_goes_back_to_choosing_one(deps, wrapping):
    r = await say(deps, wrapping, "what about CL-2011?",
                  make_form(email_offer_reply="something_else", mentions_claim=True, quoted_claim_number=True,
                            search_relation="new_search"), stated(claim_number="CL-2011"))
    # RESOLVE_INTENT and PROCESS_CASE answer it, and PROCESS_CASE offers the email again on the way back to wrap-up.
    assert (await memory(deps, wrapping, "focus.claim_ids")) == ["CL-2011"]
    assert any(f.claim == "CL-2011" for f in facts(r)) and "offer_email" in instructions(r)


async def test_whats_in_it_describes_the_email_and_asks_again(deps, wrapping):
    r = await say(deps, wrapping, "what's in it?", make_form(email_offer_reply="whats_in_it"))
    assert "email_contents" in instructions(r) and r.state == State.POST_PROCESS
    assert (await memory(deps, wrapping, "pending")).kind == "offer_email"
