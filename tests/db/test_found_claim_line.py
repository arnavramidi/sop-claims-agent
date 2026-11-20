"""The line that names the claim found: number, type and date. Its summary on file is quoted only where it is the
only description the caller gets; when the reply goes on to give the claim's status and reason, the summary would
only repeat them."""

from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, facts, instructions, focus, make_form, new_conversation, say, stated

from claims_agent.form.fields import Pick


async def test_the_brief_message_names_the_claim_without_repeating_its_summary(deps, conv):
    form = make_form(**GAVE, need="denial_question", how_many="one", claim_types={"healthcare": 0.97},
                     claim_statuses={"denied": 0.96}, months={1: 0.95}, no_year_said=True)
    r = await say(deps, conv, BRIEF_MSG, form, BRIEF_SAID)
    found = next(f for f in facts(r) if f.kind == "found_claim")
    assert found.text == "Claim CL-2048: healthcare, opened 2026-01-12."
    assert any(f.kind == "denial_reason" for f in facts(r))


async def test_a_claim_found_when_the_need_must_be_asked_keeps_its_summary(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, 1985-03-15, SSN 4472", make_form(gave_identity_detail=True, caller_role="policyholder"),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    two_needs = Pick(value="status_inquiry", p=0.45, options={"status_inquiry": 0.45, "next_steps": 0.4})
    r = await say(deps, cid, "my dental claim", make_form(mentions_claim=True, how_many="one",
                                                          claim_types={"dental": 0.95}, need=two_needs))
    assert await focus(deps, cid) == ["CL-1899"] and "ask_need" in instructions(r)
    found = next(f for f in facts(r) if f.kind == "found_claim")
    assert 'Summary on file: "Dental claim completed"' in found.text
