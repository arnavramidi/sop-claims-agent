"""Whole turns through PROCESS_CASE: openings, follow-up offers, not-on-file and the hand-over, over real Postgres."""

import json
from datetime import date

import pytest
from helpers import (BRIEF_MSG, BRIEF_SAID, FORM, GAVE, escalation_reasons, facts, focus, instructions, make_deps,
                     make_form, memory, new_conversation, parts, say, stated)

from claims_agent import engine
from claims_agent.db import conversations as store
from claims_agent.db.identity import clear_verified
from claims_agent.engine import TurnContext
from claims_agent.form.fields import Pick
from claims_agent.memory import Pending
from claims_agent.settings import load_settings
from claims_agent.states import State

TODAY = date(2026, 3, 5)


async def _verified(deps, name, dob, last4):
    cid = await new_conversation(deps)
    r = await say(deps, cid, f"{name}, born {dob}, ID ends {last4}", make_form(**GAVE),
                  stated(policyholder_name=name, date_of_birth=[dob], id_digits=last4))
    return cid, r


async def _in_case(deps, cid, need, claim_id):
    """Talk the conversation into PROCESS_CASE with one claim in focus (quoted by number) and the opening given."""
    r = await say(deps, cid, f"about {claim_id}", make_form(quoted_claim_number=True, need=need),
                  stated(claim_number=claim_id))
    assert r.state == State.PROCESS_CASE and await focus(deps, cid) == [claim_id]
    return r


@pytest.fixture
async def margaret(deps):
    """Verified as P9 and has said nothing about a claim, so it sits in RESOLVE_INTENT."""
    cid, r = await _verified(deps, "Margaret Chen", "1985-03-15", "4472")
    assert r.state == State.RESOLVE_INTENT
    return cid


@pytest.fixture
async def denial_case(deps, margaret):
    """P9 asking why CL-2048 was denied: in PROCESS_CASE, the opening given, the first follow-up offered."""
    await _in_case(deps, margaret, "denial_question", "CL-2048")
    assert (await memory(deps, margaret, "pending")).kind == ALTERNATIVES_OFFER
    return margaret


@pytest.fixture
async def closed_case(deps, margaret):
    """P9 asking about the closed claim CL-2011 (no documents outstanding): in PROCESS_CASE."""
    await _in_case(deps, margaret, "status_inquiry", "CL-2011")
    return margaret


ALTERNATIVES_OFFER = "offer_followup:CL-2048:missing_required_material_alternatives"
# Parts that put a question to the caller.
QUESTIONS = {"ask_need", "anything_else", "offer_followup", "offer_person", "ask_which_answer", "ask_which_document",
             "which_claim",
             "offer_email", "nearest_matches", "overview", "narrow_down"}


# ---- core behaviour ----------------------------------------------------------------------------------------

async def test_entry_gives_the_opening_then_offers_the_first_followup(deps, margaret):
    r = await say(deps, margaret, "why was my denied healthcare claim denied?", make_form(need="denial_question", claim_statuses={"denied": 0.95}, how_many="one"))
    # The answers module names these facts "documents" and "deadline" (the answers asked for are
    # documents_needed and appeal_deadline).
    assert [f.kind for f in facts(r) if f.claim == "CL-2048" and f.kind != "found_claim"][:4] == ["status", "denial_reason", "documents", "deadline"]
    # An offer names the claim it was made for.
    assert (await memory(deps, margaret, "pending")).kind == "offer_followup:CL-2048:missing_required_material_alternatives"


async def test_the_reply_says_the_claim_is_not_denied_and_gives_its_status(deps, margaret):
    r = await say(deps, margaret, "why was CL-2011 denied?", make_form(quoted_claim_number=True, need="denial_question"), stated(claim_number="CL-2011"))
    kinds = [f.kind for f in facts(r)]; assert kinds[:2] == ["not_denied", "status"] and "denial_reason" not in kinds


async def test_yes_to_the_offer_gives_that_followup(deps, denial_case):
    r = await say(deps, denial_case, "yes please", make_form(pending_reply="yes"))
    assert "missing_required_material_alternatives" in (await memory(deps, denial_case, "request.answered"))["CL-2048"]


async def test_cant_get_it_twice_hands_over(deps, denial_case):
    cant = make_form(cant_get_document=True, which_document="pathology report")
    assert (await say(deps, denial_case, "I can't get the pathology report", cant)).state == State.PROCESS_CASE
    r = await say(deps, denial_case, "the lab still won't give it to me", cant)
    assert r.state == State.ESCALATED and await escalation_reasons(deps, denial_case) == ["document_unavailable"]


async def test_cant_get_one_document_gives_alternatives_for_that_one_only(deps, denial_case):
    r = await say(deps, denial_case, "I have the pathology report but can't get the office note", make_form(cant_get_document=True, which_document="office note"))
    t = " ".join(f.text for f in facts(r)); assert "visit summary" in t and "certified copy" not in t


async def test_an_unlinked_followup_is_answered_and_the_mismatch_recorded(deps, denial_case):
    r = await say(deps, denial_case, "what file format do you need?", make_form(answer_wanted="file_format_requirements"))
    assert "specimen details" in " ".join(f.text for f in facts(r)) and {"mismatch": "file_format_requirements"} in r.inspector["fired"]


async def test_a_followup_on_a_clean_claim_is_not_on_file(deps, closed_case):         # CL-2011 in focus
    r = await say(deps, closed_case, "when do I need to send it?", make_form(answer_wanted="submission_timing"))
    assert "not_on_file" in instructions(r) and "within a week" not in r.reply and "reasonable expectation" not in r.reply


async def test_the_same_unanswerable_question_three_times(deps, denial_case):
    stalls = []
    for i in range(3):
        r = await say(deps, denial_case, "what's the fax number?", make_form(answer_wanted="something_else"))
        assert "not_on_file" in instructions(r) and "offer_person" in instructions(r)
        stalls.append(await memory(deps, denial_case, "counters.stalled_in_a_row"))
    assert r.state == State.PROCESS_CASE
    assert stalls == [1, 2, 0]  # each turn stalls; the third reaches the limit, which offers a person and resets it


async def test_answers_given_are_remembered_per_claim_across_a_switch(deps, denial_case):
    await say(deps, denial_case, "yes please", make_form(pending_reply="yes"))
    await say(deps, denial_case, "and my auto claim?", make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one"))
    assert await focus(deps, denial_case) == ["CL-2102"]
    await say(deps, denial_case, "back to the denied one", make_form(answer_wanted="something_else", mentions_claim=True, claim_statuses={"denied": 0.95}, search_relation="new_search", how_many="one"))
    assert "missing_required_material_alternatives" in (await memory(deps, denial_case, "request.answered"))["CL-2048"]


async def test_a_new_claim_mid_answer_is_not_a_stalled_turn(deps, denial_case):
    await say(deps, denial_case, "and my auto claim?", make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one"))
    assert await memory(deps, denial_case, "counters.stalled_in_a_row") == 0


async def test_a_changed_claim_row_shows_in_the_next_reply(deps, denial_case, loader_conn):
    await loader_conn.execute("UPDATE claims SET status = 'open' WHERE case_id = 'CL-2048'"); await loader_conn.commit()
    r = await say(deps, denial_case, "what's the status now?", make_form(answer_wanted="status"))
    assert "open" in " ".join(f.text for f in facts(r) if f.kind == "status")


async def test_an_unsure_pick_asks_the_caller_naming_two_readings(deps, denial_case):
    r = await say(deps, denial_case, "what about the timing?", make_form(answer_wanted=Pick(value="submission_timing", p=0.45, options={"submission_timing": 0.45, "processing_time_after_submission": 0.4})))
    assert "ask_which_answer" in instructions(r)


async def test_a_wary_caller_message_is_passed_to_the_wording_model(deps, denial_case):
    await say(deps, denial_case, "this is ridiculous, what's the status?", make_form(answer_wanted="status", mood="angry"))
    b = deps.wording.briefs[-1]; assert b.mood == "angry" and b.caller_message.startswith("this is ridiculous")


# ---- added ----------------------------------------------------------------------------------------------------

def _texts(r) -> str:
    return " ".join(f.text for f in facts(r))


async def test_the_opening_is_marked_answered_and_the_offer_is_one_yes_no_question(deps, denial_case):
    answered = (await memory(deps, denial_case, "request.answered"))["CL-2048"]
    assert answered == ["status", "denial_reason", "documents_needed", "appeal_deadline"]
    assert await memory(deps, denial_case, "request.need") == "denial_question"
    pend = await memory(deps, denial_case, "pending")
    assert pend.yes_no and "CL-2048" not in pend.label


async def test_yes_offers_the_next_followup_and_no_asks_if_there_is_anything_else(deps, denial_case):
    r = await say(deps, denial_case, "yes please", make_form(pending_reply="yes"))
    assert "answer" in instructions(r) and r.inspector["memory"]["pending"]["kind"] == "offer_followup:CL-2048:submission_timing"
    r = await say(deps, denial_case, "no thanks", make_form(pending_reply="no"))
    assert instructions(r) == ["anything_else"] and (await memory(deps, denial_case, "pending")).kind == "anything_else"


async def test_with_every_linked_followup_answered_it_asks_if_there_is_anything_else(deps, denial_case):
    await say(deps, denial_case, "yes please", make_form(pending_reply="yes"))
    r = await say(deps, denial_case, "yes", make_form(pending_reply="yes"))
    assert "within a week" in _texts(r) and "offer_followup" not in instructions(r) and "anything_else" in instructions(r)


async def test_alternatives_are_recorded_for_the_document_on_the_claim(deps, denial_case):
    await say(deps, denial_case, "I can't get the office note", make_form(cant_get_document=True, which_document="office note"))
    assert await memory(deps, denial_case, "request.alternatives_offered") == {"CL-2048": ["office note"]}


async def test_an_unsure_cant_get_it_never_hands_over(deps, denial_case):
    await say(deps, denial_case, "I can't get the office note", make_form(cant_get_document=True, which_document="office note"))
    r = await say(deps, denial_case, "maybe I can't get it", make_form(cant_get_document=0.55, which_document="office note"))
    assert r.state == State.PROCESS_CASE and await escalation_reasons(deps, denial_case) == []


# A real form can only answer the which-document question with an offered document, "all", "not_listed" (the caller
# named a document that is not on the list) or "none_stated".

async def test_a_document_not_on_the_claim_never_gets_another_documents_alternatives(deps, denial_case):
    cant = make_form(cant_get_document=True, which_document="not_listed")
    for _ in range(2):
        r = await say(deps, denial_case, "I can't get the repair estimate", cant)
        t = _texts(r)
        assert "replacement copy" in t  # the general alternatives
        assert "repair estimate" not in t and "visit summary" not in t and "certified copy" not in t
        assert await memory(deps, denial_case, "request.alternatives_offered") == {}
    assert r.state == State.PROCESS_CASE and await escalation_reasons(deps, denial_case) == []


async def test_a_document_not_on_the_list_never_hands_over_even_with_one_outstanding(deps):
    cid, _ = await _verified(deps, "Ma Tian", "1964-09-10", "6688")    # CL-3001, one document: diagnosis report
    cant = make_form(cant_get_document=True, which_document="not_listed")
    for _ in range(3):
        r = await say(deps, cid, "I can't get the repair estimate", cant)
        assert "replacement copy" in _texts(r) and r.state == State.PROCESS_CASE
    assert await memory(deps, cid, "request.alternatives_offered") == {} and await escalation_reasons(deps, cid) == []


async def test_no_document_named_with_one_outstanding_means_that_one_and_twice_hands_over(deps):
    cid, _ = await _verified(deps, "Ma Tian", "1964-09-10", "6688")
    cant = make_form(cant_get_document=True, which_document="none_stated")
    assert (await say(deps, cid, "I can't get it", cant)).state == State.PROCESS_CASE
    assert await memory(deps, cid, "request.alternatives_offered") == {"CL-3001": ["diagnosis report"]}
    r = await say(deps, cid, "I still can't get it", cant)
    assert r.state == State.ESCALATED and await escalation_reasons(deps, cid) == ["document_unavailable"]


async def test_the_only_document_outstanding_needs_no_name(deps):
    cid, _ = await _verified(deps, "Ma Tian", "1964-09-10", "6688")    # CL-3001, one document: diagnosis report
    r = await say(deps, cid, "I can't get it", make_form(cant_get_document=True))
    assert await memory(deps, cid, "request.alternatives_offered") == {"CL-3001": ["diagnosis report"]}
    assert "answer" in instructions(r)


async def test_a_not_on_file_answer_is_not_grounded_and_offers_a_person(deps, denial_case):
    r = await say(deps, denial_case, "what's the fax number?", make_form(answer_wanted="something_else"))
    assert "offer_person" in instructions(r) and "offer_followup" not in instructions(r)
    assert (await memory(deps, denial_case, "pending")).kind == "offer_person"
    assert await memory(deps, denial_case, "counters.stalled_in_a_row") == 1


async def test_a_confident_answer_is_grounded_progress(deps, denial_case):
    r = await say(deps, denial_case, "how much did they allow?", make_form(answer_wanted="amounts"))
    assert "1450.00" in _texts(r) and await memory(deps, denial_case, "counters.stalled_in_a_row") == 0
    assert "amounts" in (await memory(deps, denial_case, "request.answered"))["CL-2048"]


async def test_a_linked_followup_records_no_mismatch(deps, denial_case):
    r = await say(deps, denial_case, "when do I need to send them?", make_form(answer_wanted="submission_timing"))
    assert "within a week" in _texts(r) and not any("mismatch" in f for f in r.inspector["fired"])


async def test_a_passed_deadline_offers_a_person_and_never_the_week(deps, margaret):
    await set_today(deps, margaret, "2026-03-20")
    r = await say(deps, margaret, "why was my denied healthcare claim denied?", make_form(need="denial_question", claim_statuses={"denied": 0.95}, how_many="one"))
    assert "has passed" in _texts(r) and "offer_person" in instructions(r) and "offer_followup" not in instructions(r)
    r = await say(deps, margaret, "when do I need to send it?", make_form(answer_wanted="submission_timing"))
    assert "within a week" not in r.reply and "offer_person" in instructions(r)


async def set_today(deps, cid, day):
    """The inspector's "today" control (the app's own function, no row written by hand)."""
    async with deps.db.conversation(cid) as c:
        await store.set_controls(c, cid, today=date.fromisoformat(day))


async def test_the_form_is_offered_the_answers_and_documents_of_the_claims_in_focus(deps, denial_case):
    await say(deps, denial_case, "hmm", make_form())
    req = deps.filler.requests[-1]
    names = [a for a, _ in req.answers]
    assert names[:6] == ["status", "denial_reason", "documents_needed", "appeal_deadline", "amounts", "next_steps"]
    assert "not_on_file" not in names and "missing_required_material_alternatives" in names
    assert req.documents == ("pathology report", "office note")


async def test_a_clean_claim_offers_the_form_no_followup_and_no_document(deps, closed_case):
    await say(deps, closed_case, "hmm", make_form())
    req = deps.filler.requests[-1]
    assert [a for a, _ in req.answers] == ["status", "amounts", "next_steps"] and req.documents == ()


async def test_a_brand_new_conversation_offers_the_form_no_claim_answers_and_no_documents(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "hello there", make_form())
    req = deps.filler.requests[-1]
    names = [a for a, _ in req.answers]
    assert not req.verified and req.documents == () and req.listed_claims == ()
    assert names[:6] == list(engine.ANSWER_KINDS)  # the vocabulary only, never narrowed by a claim


async def test_nothing_about_a_claim_reaches_the_form_before_verification(deps, denial_case, monkeypatch):
    # The form request's own refusal is switched off, so only the engine's gate stands: claims are in focus in memory
    # but the database no longer counts the conversation as verified.
    monkeypatch.setattr(engine.FormRequest, "__post_init__", lambda self: None)
    async with deps.db.conversation(denial_case) as c:
        await clear_verified(c, denial_case)
    assert await focus(deps, denial_case) == ["CL-2048"]
    await say(deps, denial_case, "hello there", make_form())
    req = deps.filler.requests[-1]
    assert not req.verified and req.documents == () and req.listed_claims == ()
    assert [a for a, _ in req.answers][:6] == list(engine.ANSWER_KINDS) and "amounts" in [a for a, _ in req.answers]


async def test_a_new_need_with_no_answer_named_goes_back_to_settle_it(deps, closed_case):
    r = await say(deps, closed_case, "what do I do next?", make_form(need="next_steps"))
    assert await memory(deps, closed_case, "request.need") == "next_steps" and r.state == State.PROCESS_CASE
    assert any(f.kind == "next_steps" for f in facts(r))


async def test_a_claim_that_is_no_longer_visible_sends_the_caller_back_and_the_reply_asks_what_they_need(
        deps, denial_case, loader_conn):
    await loader_conn.execute("DELETE FROM claim_documents WHERE case_id = 'CL-2048'")
    await loader_conn.execute("DELETE FROM claims WHERE case_id = 'CL-2048'"); await loader_conn.commit()
    r = await say(deps, denial_case, "ok, and then?", make_form())
    assert r.reply.strip() and "ask_need" in instructions(r) and r.state == State.RESOLVE_INTENT
    assert await focus(deps, denial_case) == [] and "CL-2048" not in _texts(r)


async def _step_directly(deps, cid, form, mutate=None, message="a question about my claim"):
    """Run PROCESS_CASE's step in the conversation's own transaction, for memory talking cannot reach yet (the email
    offer is POST_PROCESS's)."""
    from claims_agent.rules.process_case import step as process_step
    async with deps.db.conversation(cid) as c:
        conv, mem = await store.load_for_turn(c, cid)
        if mutate is not None:
            mutate(mem)
        return await process_step(TurnContext(c, conv, mem, message, form, stated(), deps, TODAY, False)), mem


@pytest.mark.parametrize("role,phrase", [("policyholder", "the email on your policy"),
                                         ("third_party", "the policyholder's email")])
async def test_coming_back_from_the_email_offer_offers_it_again(deps, denial_case, role, phrase):
    def came_back(mem):
        mem.wrapup.offers_made = 1
        mem.identity.caller_role = role
    out, _ = await _step_directly(deps, denial_case, make_form(answer_wanted="amounts"), came_back)
    assert [p.instruction for p in out.parts][-1] == "offer_email" and out.next_state == State.POST_PROCESS
    assert out.parts[-1].pending.kind == "offer_email" and out.parts[-1].pending.yes_no
    assert phrase in [f.text for f in out.parts[-1].facts if f.kind == "destination"]
    assert not any(p.instruction in ("offer_followup", "anything_else") for p in out.parts)


@pytest.mark.parametrize("form", [
    make_form(answer_wanted="amounts"), make_form(answer_wanted="next_steps"),
    make_form(answer_wanted="file_format_requirements"), make_form(answer_wanted="processing_time_after_submission"),
    make_form(answer_wanted="submission_method"), make_form(answer_wanted="receipt_confirmation"),
    make_form(answer_wanted="something_else"), make_form(pending_reply="yes"),
    make_form(cant_get_document=True, which_document="office note"),
    make_form(answer_wanted=Pick(value="submission_timing", p=0.45,
                                 options={"submission_timing": 0.45, "processing_time_after_submission": 0.4})),
])
async def test_plain_replies_pass_the_reply_check(deps, denial_case, form):
    r = await say(deps, denial_case, "a question about my claim", form)
    assert r.inspector["reply_source"] == "live" and all(c["passed"] for c in r.inspector["checks"])


async def test_the_opening_plain_reply_passes_the_reply_check(deps, margaret):
    r = await say(deps, margaret, "why was my denied healthcare claim denied?", make_form(need="denial_question", claim_statuses={"denied": 0.95}, how_many="one"))
    assert r.inspector["reply_source"] == "live" and all(c["passed"] for c in r.inspector["checks"])


async def test_nothing_from_another_customer_reaches_a_model(deps, denial_case):
    await say(deps, denial_case, "yes please", make_form(pending_reply="yes"))
    blob = json.dumps([str(b) for b in deps.wording.briefs] + [str(q) for q in deps.filler.requests])
    assert "CL-3001" not in blob and "diagnosis report" not in blob


async def test_the_single_claim_hop_now_runs_the_real_opening(deps):
    cid, r = await _verified(deps, "Ma Tian", "1964-09-10", "6688")
    assert r.state == State.PROCESS_CASE and "answer" in instructions(r)
    assert any(p.pending is not None and p.pending.kind.startswith("offer_followup:") for p in parts(r))


# ---- further cases ------------------------------------------------------------------------------------------------

def _questions(r) -> list[str]:
    return [i for i in instructions(r) if i in QUESTIONS]


@pytest.fixture
async def two_claims(deps, margaret):
    """P9 asking about all her healthcare claims: CL-2048 (denied, documents) and CL-2011 (closed) in focus; the
    first follow-up is offered for CL-2048."""
    r = await say(deps, margaret, "all my healthcare claims", make_form(how_many="all", claim_types={"healthcare": 0.95}))
    assert r.state == State.PROCESS_CASE and sorted(await focus(deps, margaret)) == ["CL-2011", "CL-2048"]
    assert (await memory(deps, margaret, "pending")).kind == ALTERNATIVES_OFFER
    return margaret


async def test_a_yes_gives_the_followup_only_for_the_claim_it_was_offered_for(deps, two_claims):
    r = await say(deps, two_claims, "yes please", make_form(pending_reply="yes"))
    assert "not_on_file" not in instructions(r) and "offer_person" not in instructions(r)
    assert {f.claim for f in facts(r) if f.kind == "guidance"} == {"CL-2048"}
    answered = await memory(deps, two_claims, "request.answered")
    assert "missing_required_material_alternatives" in answered["CL-2048"]
    assert "missing_required_material_alternatives" not in answered.get("CL-2011", [])


async def test_an_offer_does_not_outlive_a_switch_to_another_claim(deps, denial_case):
    await say(deps, denial_case, "and my auto claim?", make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one"))
    assert await focus(deps, denial_case) == ["CL-2102"]
    assert not (await memory(deps, denial_case, "pending")).kind.startswith("offer_followup:")
    r = await say(deps, denial_case, "yes", make_form(pending_reply="yes"))
    assert "not_on_file" not in instructions(r) and "offer_person" not in instructions(r)
    assert not any(f.kind == "guidance" for f in facts(r))


async def test_a_stale_offer_for_a_claim_not_in_focus_is_ignored(deps, denial_case):
    def stale(mem):
        mem.pending = Pending(kind="offer_followup:CL-2102:submission_timing", label="hear about submission timing?",
                              yes_no=True)
    out, mem = await _step_directly(deps, denial_case, make_form(pending_reply="yes"), stale)
    assert not any(f.kind in ("guidance", "not_on_file") for p in out.parts for f in p.facts)
    assert "submission_timing" not in mem.request.answered["CL-2048"]


async def test_an_offer_naming_an_unknown_topic_is_ignored(deps, denial_case):
    def odd(mem):
        mem.pending = Pending(kind="offer_followup:CL-2048:no_such_topic", label="hear about it?", yes_no=True)
    out, mem = await _step_directly(deps, denial_case, make_form(pending_reply="yes"), odd)
    assert not any(f.kind in ("guidance", "not_on_file") for p in out.parts for f in p.facts)
    assert "no_such_topic" not in mem.request.answered["CL-2048"]


async def test_a_declined_followup_is_not_offered_again(deps, denial_case):
    await say(deps, denial_case, "no thanks", make_form(pending_reply="no"))
    assert await memory(deps, denial_case, "request.declined") == {"CL-2048": ["missing_required_material_alternatives"]}
    r = await say(deps, denial_case, "how much did they allow?", make_form(answer_wanted="amounts"))
    assert r.inspector["memory"]["pending"]["kind"] == "offer_followup:CL-2048:submission_timing"
    assert "missing_required_material_alternatives" not in (await memory(deps, denial_case, "request.answered"))["CL-2048"]


async def test_a_declined_followup_can_still_be_asked_for(deps, denial_case):
    await say(deps, denial_case, "no thanks", make_form(pending_reply="no"))
    r = await say(deps, denial_case, "actually, what can I do if I can't get them?",
                  make_form(answer_wanted="missing_required_material_alternatives"))
    assert "replacement copy" in _texts(r)
    assert "missing_required_material_alternatives" in (await memory(deps, denial_case, "request.answered"))["CL-2048"]


async def test_declines_are_wiped_by_a_different_person_reset(deps, denial_case):
    await say(deps, denial_case, "no thanks", make_form(pending_reply="no"))
    await say(deps, denial_case, "what about my husband's policy?", make_form(other_person_policy=True))
    m = await memory(deps, denial_case)
    assert m.request.declined == {} and m.request.alternatives_offered == {} and m.request.answered == {}


async def test_cant_get_any_of_them_gives_each_documents_alternatives_and_twice_hands_over(deps, denial_case):
    every = make_form(cant_get_document=True, which_document="all")
    r = await say(deps, denial_case, "I can't get any of these", every)
    t = _texts(r)
    assert "visit summary" in t and "certified copy" in t and r.state == State.PROCESS_CASE
    assert await memory(deps, denial_case, "request.alternatives_offered") == {"CL-2048": ["pathology report", "office note"]}
    r = await say(deps, denial_case, "I still can't get any of them", every)
    assert r.state == State.ESCALATED and await escalation_reasons(deps, denial_case) == ["document_unavailable"]


async def test_one_new_document_in_the_set_is_not_a_hand_over(deps, denial_case):
    await say(deps, denial_case, "I can't get the office note", make_form(cant_get_document=True, which_document="office note"))
    r = await say(deps, denial_case, "or any of them", make_form(cant_get_document=True, which_document="all"))
    assert r.state == State.PROCESS_CASE and "certified copy" in _texts(r)
    r = await say(deps, denial_case, "none of them", make_form(cant_get_document=True, which_document="all"))
    assert r.state == State.ESCALATED


async def test_an_unnamed_document_among_several_is_asked_about(deps, denial_case):
    r = await say(deps, denial_case, "I can't get it", make_form(cant_get_document=True))
    assert "ask_which_document" in instructions(r) and (await memory(deps, denial_case, "pending")).kind == "which_document"
    assert {f.claim for f in facts(r) if f.kind == "requested_document"} == {"CL-2048"}
    assert "pathology report" in _texts(r) and "office note" in _texts(r) and "visit summary" not in _texts(r)
    assert await memory(deps, denial_case, "request.alternatives_offered") == {}
    # The reply names the document: it counts as the can't-get for it, though it does not say "can't" again.
    r = await say(deps, denial_case, "the office note", make_form(which_document="office note"))
    assert "visit summary" in _texts(r)
    assert await memory(deps, denial_case, "request.alternatives_offered") == {"CL-2048": ["office note"]}
    r = await say(deps, denial_case, "I really can't get the office note", make_form(cant_get_document=True, which_document="office note"))
    assert r.state == State.ESCALATED


async def test_any_other_reply_to_the_which_document_question_is_an_ordinary_message(deps, denial_case):
    await say(deps, denial_case, "I can't get it", make_form(cant_get_document=True))
    r = await say(deps, denial_case, "actually, when's my deadline?", make_form(answer_wanted="appeal_deadline"))
    assert "ask_which_document" not in instructions(r) and any(f.kind == "deadline" for f in facts(r))
    assert await memory(deps, denial_case, "request.alternatives_offered") == {}


async def test_all_of_them_answers_the_which_document_question(deps, denial_case):
    await say(deps, denial_case, "I can't get it", make_form(cant_get_document=True))
    await say(deps, denial_case, "all of them", make_form(which_document="all"))
    assert await memory(deps, denial_case, "request.alternatives_offered") == {"CL-2048": ["pathology report", "office note"]}


async def test_alternatives_survive_a_switch_and_back(deps, denial_case):
    cant = make_form(cant_get_document=True, which_document="office note")
    await say(deps, denial_case, "I can't get the office note", cant)
    await say(deps, denial_case, "and my auto claim?", make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one"))
    await say(deps, denial_case, "back to the denied one", make_form(answer_wanted="something_else", mentions_claim=True, claim_statuses={"denied": 0.95}, search_relation="new_search", how_many="one"))
    assert await focus(deps, denial_case) == ["CL-2048"]
    r = await say(deps, denial_case, "I still can't get the office note", cant)
    assert r.state == State.ESCALATED


async def test_a_document_beyond_the_offered_cap_is_found_by_its_exact_name(load_dataset):
    s = load_settings()
    s = s.model_copy(update={"limits": s.limits.model_copy(update={"documents_offered": 1})})
    deps = make_deps(await load_dataset("starter"), settings=s)
    cid, _ = await _verified(deps, "Margaret Chen", "1985-03-15", "4472")
    await _in_case(deps, cid, "denial_question", "CL-2048")
    await say(deps, cid, "hmm", make_form())
    assert deps.filler.requests[-1].documents == ("pathology report",)
    r = await say(deps, cid, "I can't get the Office Note", make_form(cant_get_document=True, which_document="not_listed"))
    assert "visit summary" in _texts(r)
    assert await memory(deps, cid, "request.alternatives_offered") == {"CL-2048": ["office note"]}


async def test_a_question_about_another_claim_goes_to_that_claim(deps, denial_case):
    r = await say(deps, denial_case, "what's the status of my auto claim?", make_form(answer_wanted="status", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one"))
    assert await focus(deps, denial_case) == ["CL-2102"]
    assert {f.claim for f in facts(r) if f.kind == "status"} == {"CL-2102"}


async def test_a_question_about_this_claim_stays_on_it(deps, denial_case):
    r = await say(deps, denial_case, "what's the status of this claim?", make_form(answer_wanted="status", mentions_claim=True, search_relation="new_search", how_many="one"))
    assert await focus(deps, denial_case) == ["CL-2048"] and r.state == State.PROCESS_CASE
    assert {f.claim for f in facts(r) if f.kind == "status"} == {"CL-2048"}


async def test_no_thanks_but_a_question_answers_the_question(deps, denial_case):
    r = await say(deps, denial_case, "No thanks, but when do I need to send it?", make_form(pending_reply="no", answer_wanted="submission_timing"))
    assert "within a week" in _texts(r) and "answer" in instructions(r)
    assert await memory(deps, denial_case, "request.declined") == {"CL-2048": ["missing_required_material_alternatives"]}


async def test_yes_and_cant_get_a_named_document_takes_the_document(deps, denial_case):
    r = await say(deps, denial_case, "Yes, I can't get the office note", make_form(pending_reply="yes", cant_get_document=True, which_document="office note"))
    assert "visit summary" in _texts(r)
    assert await memory(deps, denial_case, "request.alternatives_offered") == {"CL-2048": ["office note"]}


async def test_the_assignment_message_asks_one_question_and_only_after_the_answer(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, BRIEF_MSG, FORM, BRIEF_SAID)
    names = instructions(r)
    assert names[0] == "verified" and "answer" in names
    assert _questions(r) == ["offer_followup"] and names[-1] == "offer_followup"
    assert names.index("answer") < names.index("offer_followup")
    assert "?" not in parts(r)[0].plain and r.reply.count("?") == 1


async def test_a_caller_who_verifies_and_asks_nothing_gets_one_question(deps):
    cid, r = await _verified(deps, "Margaret Chen", "1985-03-15", "4472")
    assert instructions(r) == ["verified", "ask_need"] and r.reply.count("?") == 1


async def test_a_single_claim_customer_is_not_asked_how_to_help_before_the_answer(deps):
    cid, r = await _verified(deps, "Ma Tian", "1964-09-10", "6688")
    assert _questions(r) == ["offer_followup"] and r.reply.count("?") == 1


async def test_the_need_note_is_stated_only_for_a_claim_whose_row_holds_it(deps, margaret, loader_conn):
    await say(deps, margaret, "why was CL-2011 denied?", make_form(quoted_claim_number=True, need="denial_question"), stated(claim_number="CL-2011"))
    await loader_conn.execute("UPDATE claims SET denial_reason = 'late filing' WHERE case_id = 'CL-2011'")
    await loader_conn.commit()

    from claims_agent.rules.process_case import step as process_step
    async with deps.db.conversation(margaret) as c:
        conv, mem = await store.load_for_turn(c, margaret)
        assert mem.focus.need_note == "not_denied"
        out = await process_step(TurnContext(c, conv, mem, "again", make_form(), stated(), deps, TODAY, True))
    assert not any(f.kind == "not_denied" for p in out.parts for f in p.facts)


async def test_an_offer_the_engine_drops_is_recorded_as_dropped(deps, denial_case):
    r = await say(deps, denial_case, "how much did they allow, or maybe a person?", make_form(answer_wanted="amounts", wants_person=0.5))
    assert "offer_followup" not in instructions(r) and "offer_person" in instructions(r)
    assert any(f.get("rule") == "offer_dropped" and f.get("kind", "").startswith("offer_followup:")
               for f in r.inspector["fired"])


async def test_a_state_that_keeps_claim_facts_from_models_gets_no_documents_and_every_answer(load_dataset):
    s = load_settings()
    s = s.model_copy(update={"states": {**s.states, "PROCESS_CASE": s.states["PROCESS_CASE"].model_copy(update={"freedom": "strict"})}})
    deps = make_deps(await load_dataset("starter"), settings=s)
    cid, _ = await _verified(deps, "Margaret Chen", "1985-03-15", "4472")
    await _in_case(deps, cid, "denial_question", "CL-2048")
    await say(deps, cid, "hmm", make_form())
    req = deps.filler.requests[-1]
    assert req.documents == () and [a for a, _ in req.answers][:6] == list(engine.ANSWER_KINDS)
    assert "receipt_confirmation" in [a for a, _ in req.answers]


async def test_one_usable_reading_is_asked_about(deps, denial_case):
    r = await say(deps, denial_case, "what about the timing?", make_form(answer_wanted=Pick(value="submission_timing", p=0.45, options={"submission_timing": 0.45, "none": 0.4})))
    assert "ask_which_answer" in instructions(r) and "submission timing" in _texts(r)
    assert "anything_else" not in instructions(r)


async def test_a_turn_that_ends_in_done_gets_no_need_question(deps, denial_case, monkeypatch):
    from claims_agent.engine import STEPS, StepOutcome

    async def silent(ctx):
        return StepOutcome(State.DONE)
    monkeypatch.setitem(STEPS, State.PROCESS_CASE, silent)
    r = await say(deps, denial_case, "thanks", make_form())
    assert r.state == State.DONE and "ask_need" not in instructions(r)


# ---- further cases ------------------------------------------------------------------------------------------------

AUTO = dict(mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one")


async def test_a_specific_question_about_another_claim_gets_that_answer(deps, denial_case):
    r = await say(deps, denial_case, "how much did they allow on my auto claim?", make_form(answer_wanted="amounts", **AUTO))
    assert await focus(deps, denial_case) == ["CL-2102"]
    texts = [f.text for f in facts(r) if f.claim == "CL-2102"]
    assert any("allowed max amount: 3500.00" in t for t in texts)
    assert not any(f.kind == "status" for f in facts(r))  # the answer in place of the opening, not beside it
    assert "amounts" in (await memory(deps, denial_case, "request.answered"))["CL-2102"]


async def test_a_followup_asked_about_another_claim_gets_that_followup(deps, closed_case):
    r = await say(deps, closed_case, "and when do I need to send the documents for my denied claim?",
                  make_form(answer_wanted="submission_timing", mentions_claim=True, claim_statuses={"denied": 0.95},
                            search_relation="new_search", how_many="one"))
    assert await focus(deps, closed_case) == ["CL-2048"]
    assert "within a week" in _texts(r) and not any(f.kind == "denial_reason" for f in facts(r))


async def test_an_answer_that_does_not_exist_for_the_other_claim_is_not_on_file(deps, denial_case):
    r = await say(deps, denial_case, "and when do I send the auto claim's documents?",
                  make_form(answer_wanted="submission_timing", **AUTO))
    assert await focus(deps, denial_case) == ["CL-2102"] and "not_on_file" in instructions(r)
    assert "within a week" not in r.reply


async def test_the_assignment_message_still_gets_the_opening(deps):
    cid = await new_conversation(deps)
    r = await say(deps, cid, BRIEF_MSG, FORM, BRIEF_SAID)
    assert any(f.kind == "status" and f.claim == "CL-2048" for f in facts(r))


async def test_an_offer_is_matched_whole_never_by_its_start(deps, denial_case):
    def longer(mem):
        mem.pending = Pending(kind=ALTERNATIVES_OFFER + ":extra", label="hear about it?", yes_no=True)
    out, mem = await _step_directly(deps, denial_case, make_form(pending_reply="yes"), longer)
    assert not any(f.kind in ("guidance", "not_on_file") for p in out.parts for f in p.facts)
    assert {"rule": "offer_not_standing", "kind": ALTERNATIVES_OFFER + ":extra"} in out.fired


async def test_the_open_need_question_replaces_a_stale_offer(deps, denial_case, loader_conn):
    await loader_conn.execute("DELETE FROM claim_documents WHERE case_id = 'CL-2048'")
    await loader_conn.execute("DELETE FROM claims WHERE case_id = 'CL-2048'"); await loader_conn.commit()
    r = await say(deps, denial_case, "ok, and then?", make_form())
    assert "ask_need" in instructions(r) and (await memory(deps, denial_case, "pending")).kind == "ask_need"
    stalled = await memory(deps, denial_case, "counters.stalled_in_a_row")
    await say(deps, denial_case, "yes", make_form(pending_reply="yes"))
    assert await memory(deps, denial_case, "counters.stalled_in_a_row") == stalled + 1  # a bare yes is not progress


async def test_the_which_document_question_has_its_own_instruction_and_text(deps, denial_case):
    r = await say(deps, denial_case, "I can't get it", make_form(cant_get_document=True))
    part = next(p for p in parts(r) if p.instruction == "ask_which_document")
    assert part.plain.startswith("Which document can't you get") and r.reply.count("?") == 1
    assert all(f.kind == "requested_document" for f in part.facts)


# ---- a question asked as the claim is chosen ------------------------------------------------------------------

DENIAL_KINDS = ["status", "denial_reason", "documents", "deadline"]


async def test_a_confident_pick_the_opening_already_gives_gets_the_whole_opening(deps, margaret):
    r = await say(deps, margaret, "why was my denied healthcare claim denied?",
                  make_form(need="denial_question", claim_statuses={"denied": 0.95}, claim_types={"healthcare": 0.95},
                            how_many="one", answer_wanted="denial_reason"))
    assert await focus(deps, margaret) == ["CL-2048"]
    assert [f.kind for f in facts(r) if f.claim == "CL-2048" and f.kind != "found_claim"][:4] == DENIAL_KINDS
    assert {"rule": "question_in_opening", "need": "denial_question", "answer": "denial_reason"} in r.inspector["fired"]


async def test_a_denial_question_about_a_claim_not_denied_gets_the_note_then_the_status(deps, margaret):
    r = await say(deps, margaret, "why was CL-2011 denied?",
                  make_form(quoted_claim_number=True, need="denial_question", answer_wanted="denial_reason"),
                  stated(claim_number="CL-2011"))
    kinds = [f.kind for f in facts(r)]
    assert kinds[:2] == ["not_denied", "status"] and "denial_reason" not in kinds and "not_on_file" not in kinds


async def test_an_amount_asked_as_the_claim_is_chosen_replaces_the_opening(deps, margaret):
    r = await say(deps, margaret, "how much did they allow on my auto claim?",
                  make_form(answer_wanted="amounts", mentions_claim=True, claim_types={"auto": 0.95}, how_many="one",
                            need="general_claim_question"))
    assert await focus(deps, margaret) == ["CL-2102"]
    assert any("allowed max amount: 3500.00" in f.text for f in facts(r) if f.claim == "CL-2102")
    assert not any(f.kind == "status" for f in facts(r))


# ---- an upset caller --------------------------------------------------------------------------------------------

SPLIT_UPSET = Pick(value="upset", p=0.35, options={"calm": 0.0, "uneasy": 0.0, "upset": 0.51, "angry": 0.49})


async def test_an_upset_caller_asking_about_a_denial_is_worded_as_upset_and_gets_the_facts(deps, margaret):
    r = await say(deps, margaret, "I'm really upset. Why on earth was my healthcare claim denied?",
                  make_form(need="denial_question", claim_statuses={"denied": 0.95}, claim_types={"healthcare": 0.95},
                            how_many="one", answer_wanted="denial_reason", mood=SPLIT_UPSET))
    assert deps.wording.briefs[-1].mood == "upset"
    assert [f.kind for f in facts(r) if f.claim == "CL-2048" and f.kind != "found_claim"][:4] == DENIAL_KINDS


async def test_a_message_about_another_claim_goes_to_that_claim_before_any_document_help(deps, denial_case):
    # The form reads "can't get a document" into a message that only turns to the auto claim (a reading carried over
    # from the message before); the message is about the auto claim, so that claim is chosen, not a document asked.
    r = await say(deps, denial_case, "What about my auto claim?",
                  make_form(mentions_claim=True, claim_types={"auto": 1.0}, search_relation="new_search",
                            cant_get_document=0.95, which_document=Pick(value="not_listed", p=0.3)))
    assert await focus(deps, denial_case) == ["CL-2102"]
    assert "ask_which_document" not in instructions(r)
    assert "CL-2102" in {f.claim for f in facts(r)}
