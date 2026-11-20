"""Messages that ask more than one thing: each part answered in one reply, over real Postgres."""

import pytest
from helpers import GAVE, assert_no_claim_data, facts, focus, instructions, make_form, memory, new_conversation, parts, say, stated

from claims_agent.engine import run_turn
from claims_agent.states import State

MULTI = make_form(multi_question=True)
ALTERNATIVES_OFFER = "offer_followup:CL-2048:missing_required_material_alternatives"


async def say_parts(deps, conv, text, parts):
    """One whole turn: the whole message's form says it asks several things, the extractor splits it into `parts`
    (each a (text, form) pair), then each part's form; whatever the turn did not use is dropped."""
    deps.filler.queue(MULTI)
    deps.extractor.queue(stated(question_parts=[q for q, _ in parts]))
    for _, form in parts:
        deps.filler.queue(form)
    try:
        return await run_turn(deps, conv, text)
    finally:
        deps.filler.discard_unused()
        deps.extractor.discard_unused()


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
    """P9 asking why CL-2048 was denied: in PROCESS_CASE, the opening given, the first follow-up offered."""
    r = await say(deps, margaret, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                  stated(claim_number="CL-2048"))
    assert r.state == State.PROCESS_CASE and await focus(deps, margaret) == ["CL-2048"]
    assert (await memory(deps, margaret, "pending")).kind == ALTERNATIVES_OFFER
    return margaret


# ---- each part answered, in one reply ------------------------------------------------------------------------

async def test_a_glossary_question_and_a_document_question_are_both_answered_in_one_reply(deps, denial_case):
    before = len(deps.wording.briefs)          # the fixture's own turns are already in the list
    r = await say_parts(deps, denial_case, "What does allowed amount mean, and what if the clinic won't reissue the report?",
        [("What does allowed amount mean?", make_form(answer_wanted="amounts")),
         ("What if the clinic won't reissue the report?", make_form(cant_get_document=True, which_document="pathology report"))])
    # Correction: an amounts answer gives facts of kind "amount" and a follow-up gives facts of kind "guidance"; which
    # answers were given is recorded per claim under the answer's own name.
    kinds = [f.kind for f in facts(r)]; assert "amount" in kinds and "guidance" in kinds
    assert {"amounts", "missing_required_material_alternatives"} <= set((await memory(deps, denial_case, "request.answered"))["CL-2048"])
    assert len(deps.wording.briefs) == before + 1 and len(r.inspector["checks"]) >= 1


async def test_each_part_is_answered_for_its_own_claim(deps, margaret):
    r = await say_parts(deps, margaret, "Status of my auto claim, and why was my healthcare one denied?",
        [("Status of my auto claim", make_form(need="status_inquiry", claim_types={"auto": 0.95}, how_many="one", search_relation="new_search")),
         ("Why was my healthcare one denied?", make_form(need="denial_question", claim_types={"healthcare": 0.95}, how_many="one", search_relation="new_search", mentions_claim=True, answer_wanted="something_else"))])     # something_else is what sends PROCESS_CASE back to find the other claim
    by_claim = {(f.claim, f.kind) for f in facts(r)}; assert ("CL-2102", "status") in by_claim and ("CL-2048", "denial_reason") in by_claim


async def test_reason_documents_and_how_to_send_are_all_answered(deps, denial_case):
    assert "submission_method" not in (await memory(deps, denial_case, "request.answered"))["CL-2048"]
    r = await say_parts(deps, denial_case, "Why was it denied, what do you need, and how do I send it?",
        [("Why was it denied?", make_form(answer_wanted="denial_reason")), ("What do you need?", make_form(answer_wanted="documents_needed")),
         ("How do I send it?", make_form(answer_wanted="submission_method"))])
    # Correction: the documents answer gives facts of kind "documents", and a follow-up such as submission_method gives
    # facts of kind "guidance"; the answer given is recorded per claim under its own name.
    assert {"denial_reason", "documents", "guidance"} <= {f.kind for f in facts(r)}
    assert "submission_method" in (await memory(deps, denial_case, "request.answered"))["CL-2048"]


async def test_an_off_topic_part_is_declined_and_the_rest_answered(deps, denial_case):
    r = await say_parts(deps, denial_case, "What's the status, and what's the weather in Tokyo?",
        [("What's the status?", make_form(answer_wanted="status")), ("What's the weather in Tokyo?", make_form(about="off_topic", answer_wanted="none"))])
    assert "status" in [f.kind for f in facts(r)] and "decline_off_topic" in instructions(r) and await memory(deps, denial_case, "counters.stalled_in_a_row") == 0


async def test_more_than_three_parts_answers_three_and_asks_for_the_rest(deps, denial_case):
    r = await say_parts(deps, denial_case, "four questions…", [(f"q{i}", make_form(answer_wanted=a)) for i, a in enumerate(["status", "denial_reason", "documents_needed", "amounts"])])
    kinds = [f.kind for f in facts(r)]; assert "amounts" not in kinds and "more_questions_later" in instructions(r)


async def test_a_handover_in_one_part_stops_the_rest(deps, denial_case):
    r = await say_parts(deps, denial_case, "Get me a person, and what's the status?", [("Get me a person", make_form(wants_person=True)), ("What's the status?", make_form(answer_wanted="status"))])
    assert r.state == State.ESCALATED and "status" not in [f.kind for f in facts(r)]


async def test_nothing_is_split_before_verification(deps, conv):
    await say(deps, conv, "DOB is 1985-03-15, also what's the weather?", make_form(multi_question=True, **GAVE, about="off_topic"), stated(date_of_birth=["1985-03-15"]))
    assert len(deps.filler.requests) == 1 and "dob" in await memory(deps, conv, "identity.held")


# ---- the rules every part keeps ------------------------------------------------------------------------------

async def test_the_extractor_is_asked_for_the_parts_once_and_only_after_verification(deps, denial_case):
    asked = len(deps.extractor.requests)
    await say_parts(deps, denial_case, "What's the status, and why was it denied?",
                    [("What's the status?", make_form(answer_wanted="status")),
                     ("Why was it denied?", make_form(answer_wanted="denial_reason"))])
    new = deps.extractor.requests[asked:]
    assert len(new) == 1 and new[0].want_parts and not new[0].want_identity
    assert [r.message for r in deps.filler.requests[-2:]] == ["What's the status?", "Why was it denied?"]


async def test_fewer_than_two_parts_is_handled_whole(deps, denial_case):
    asked = len(deps.filler.requests)
    r = await say_parts(deps, denial_case, "What's the status?", [("What's the status?", make_form(answer_wanted="none"))])
    assert len(deps.filler.requests) == asked + 1 and "status" not in [f.kind for f in facts(r)]


async def test_one_question_per_reply_the_last_one_asked(deps, denial_case):
    r = await say_parts(deps, denial_case, "Why was it denied, what do you need, and how do I send it?",
        [("Why was it denied?", make_form(answer_wanted="denial_reason")), ("What do you need?", make_form(answer_wanted="documents_needed")),
         ("How do I send it?", make_form(answer_wanted="submission_method"))])
    asking = [p for p in parts(r) if p.pending is not None]
    assert len(asking) == 1 and asking[0] == parts(r)[-1] and await memory(deps, denial_case, "pending") == asking[0].pending


async def test_the_stall_count_moves_once_per_message(deps, denial_case):
    r = await say_parts(deps, denial_case, "What's the weather, and who won the match?",
        [("What's the weather?", make_form(about="off_topic")), ("Who won the match?", make_form(about="off_topic"))])
    assert instructions(r).count("decline_off_topic") == 1
    assert await memory(deps, denial_case, "counters.stalled_in_a_row") == 1


async def test_yes_to_the_offer_and_a_switch_to_another_claim(deps, denial_case):
    """"Yes, and what about my auto claim?": the offer is accepted and the switch is made."""
    await say_parts(deps, denial_case, "yes, and what about my auto claim?",
        [("Yes, please tell me.", make_form(pending_reply="yes")),
         ("What about my auto claim?", make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one"))])
    assert "missing_required_material_alternatives" in (await memory(deps, denial_case, "request.answered"))["CL-2048"]
    assert await focus(deps, denial_case) == ["CL-2102"]


async def test_a_different_person_in_a_later_part_drops_everything_before_it(deps, denial_case):
    r = await say_parts(deps, denial_case, "What's the status, and can we look at my husband's policy?",
        [("What's the status?", make_form(answer_wanted="status")),
         ("Can we look at my husband's policy?", make_form(other_person_policy=True))])
    assert r.state == State.VERIFY_ID and not [f for f in facts(r) if f.claim]
    assert_no_claim_data(r.reply)
