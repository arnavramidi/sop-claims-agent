"""Two readings of one question settled by what each answer gives: a follow-up that carries the deadline answers a
days-left question too, and a document named by a word that only one outstanding document has is that document."""

from helpers import facts, instructions, make_form, memory, say, stated


async def _on_2048(deps, cid):
    await say(deps, cid, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
              stated(claim_number="CL-2048"))


async def test_how_long_to_send_them_gets_the_timing_follow_up_with_its_deadline(deps, verified_conv):
    await _on_2048(deps, verified_conv)
    r = await say(deps, verified_conv, "How long do I have to send them?",
                  make_form(answer_wanted="submission_timing", calculation="days_left"))
    assert "submission_timing" in (await memory(deps, verified_conv, "request.answered"))["CL-2048"]
    assert any(f.kind == "guidance" for f in facts(r)) and any("2026-03-18" in f.text for f in facts(r))


async def test_a_days_left_question_with_no_follow_up_asked_is_still_worked_out(deps, verified_conv):
    await _on_2048(deps, verified_conv)
    r = await say(deps, verified_conv, "How many days are left?", make_form(calculation="days_left"))
    assert any(f.get("rule") == "calculation" for f in r.inspector["fired"])


async def test_the_report_names_the_one_outstanding_document_that_is_a_report(deps, verified_conv):
    await _on_2048(deps, verified_conv)
    r = await say(deps, verified_conv, "What if the clinic won't reissue the report?",
                  make_form(cant_get_document=True, which_document="none_stated"))
    assert "ask_which_document" not in instructions(r)
    assert (await memory(deps, verified_conv, "request.alternatives_offered"))["CL-2048"] == ["pathology report"]


async def test_a_word_that_two_documents_share_still_asks_which(deps, verified_conv):
    await _on_2048(deps, verified_conv)
    r = await say(deps, verified_conv, "What if I can't get the paperwork?",
                  make_form(cant_get_document=True, which_document="none_stated"))
    assert "ask_which_document" in instructions(r)
