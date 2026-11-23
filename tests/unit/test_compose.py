from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.reply.compose import LAST_RESORT, compose
from claims_agent.reply.instructions import DESTINATION_RULE, INSTRUCTIONS, MISSING
from claims_agent.reply.wording import ScriptedWording
from claims_agent.settings import load_settings
from claims_agent.states import State

SETTINGS = load_settings()
STATUSES = ("denied", "closed", "open")
F = (Fact("amounts", "Allowed maximum for claim CL-2048: 1450.00", "CL-2048"),
     Fact("appeal_deadline", "Appeal deadline for claim CL-2048: 2026-03-18 (13 days left)", "CL-2048"))


def brief(facts=(), verified=True, numbers=(), plain="Your appeal deadline is 2026-03-18."):
    return ReplyBrief(state=State.PROCESS_CASE,
                      parts=(ReplyPart("answer", facts=tuple(facts), plain=plain),
                             ReplyPart("anything_else", plain="Anything else?")),
                      mood="calm", caller_message="when is my deadline", previous_reply=None, english=True,
                      verified=verified, caller_claim_numbers=tuple(numbers))


async def test_a_failed_reply_is_rewritten_once_then_the_template_is_sent():
    w = ScriptedWording(["See https://x.example", "Still https://x.example"])
    c = await compose(brief(facts=F), w, SETTINGS, STATUSES)
    assert c.source == "template" and c.text == brief(facts=F).plain_text() and len(w.rewrites) == 1
    assert len(c.checks) == 3 and not c.checks[0].passed and not c.checks[1].passed and c.checks[2].passed


async def test_a_clean_rewrite_is_used():
    w = ScriptedWording(["See https://x.example", "Your deadline is March 18."])
    c = await compose(brief(facts=F), w, SETTINGS, STATUSES)
    assert c.source == "rewrite" and c.text == "Your deadline is March 18."
    assert w.rewrites[0][1] == "See https://x.example" and w.rewrites[0][2] == ("web_address",)


async def test_a_clean_first_reply_is_used_without_a_rewrite():
    w = ScriptedWording(["Your deadline is March 18."])
    c = await compose(brief(facts=F), w, SETTINGS, STATUSES)
    assert c.source == "live" and w.rewrites == [] and len(c.checks) == 1


async def test_a_fixed_reply_makes_no_model_call():
    w = ScriptedWording([])
    c = await compose(brief(plain="Hello."), w, SETTINGS, STATUSES, fixed=True)
    assert c.source == "fixed" and w.briefs == [] and c.text == brief(plain="Hello.").plain_text()
    assert len(c.checks) == 1 and c.checks[0].passed


async def test_an_unscripted_wording_echoes_the_plain_text_which_passes():
    c = await compose(brief(facts=F), ScriptedWording(), SETTINGS, STATUSES)
    assert c.source == "live" and c.text == brief(facts=F).plain_text()


async def test_the_scripted_wording_can_be_queued_later():
    w = ScriptedWording([])
    w.queue(["Your deadline is March 18."])
    c = await compose(brief(facts=F), w, SETTINGS, STATUSES)
    assert c.text == "Your deadline is March 18." and len(w.briefs) == 1


def test_the_brief_joins_facts_and_plain_text():
    b = brief(facts=F)
    assert b.facts() == F
    assert b.plain_text() == "Your appeal deadline is 2026-03-18.\n\nAnything else?"


def test_every_instruction_key_the_plans_use_exists():
    keys = """ask_again ask_identity ask_another_detail ask_full_value restate_value explain_step decline_off_topic
    unsupported_offer_person offer_person handover closed final_notice cannot_read_out_personal_detail verified
    consent_requested consent_pending consent_denied not_listed no_claims_on_file claim_not_on_policy which_claim
    nearest_matches overview narrow_down ask_need ask_which_answer ask_which_document ask_which_figure answer not_on_file offer_followup anything_else
    offer_email email_sent email_skipped email_contents email_address_not_read_out different_address_not_possible
    goodbye model_trouble offer_person_in_language found_claim more_questions_later ask_same_policyholder
    email_already_sent email_at_wrap_up
    acknowledge_refusal ask_caller_name offer_person_before_closing unreadable_detail ask_which_of_these""".split()
    assert set(keys) == set(INSTRUCTIONS) and all(INSTRUCTIONS[k].strip() for k in keys)
    assert MISSING == ("addresses", "links", "receipt status", "appeal steps", "amounts owed")


LINK_FACT = Fact("guidance", "Upload it at https://portal.example.com")


async def test_a_template_that_fails_the_check_is_replaced_by_the_last_resort_line():
    b = brief(facts=(LINK_FACT,), plain="Upload it at https://portal.example.com")
    w = ScriptedWording(["bad https://a.example", "bad https://b.example"])
    c = await compose(b, w, SETTINGS, STATUSES)
    assert c.source == "last_resort" and c.text == LAST_RESORT and len(w.rewrites) == 1
    assert not c.checks[-2].passed and c.checks[-1].passed


async def test_a_fixed_text_that_fails_the_check_is_replaced_too():
    w = ScriptedWording([])
    c = await compose(brief(plain="Call 650-521-2836."), w, SETTINGS, STATUSES, fixed=True)
    assert c.source == "last_resort" and c.text == LAST_RESORT and w.briefs == []


def test_the_last_resort_line_passes_the_check():
    assert check_reply(LAST_RESORT, brief(), wary_status_words=SETTINGS.reply_check.wary_status_words,
                       data_statuses=STATUSES, status_forms=SETTINGS.reply_check.status_forms).passed


def test_no_instruction_asks_for_a_value_to_be_repeated():
    for key, text in INSTRUCTIONS.items():
        assert "read back the" not in text.lower() and "as they stated" not in text.lower(), key
    r = INSTRUCTIONS["restate_value"].lower()
    assert "kind" in r and "never repeat" in r


def test_the_person_instructions_and_the_language_line():
    for key in ("offer_person", "unsupported_offer_person"):
        assert "?" in INSTRUCTIONS[key] and "yes-or-no" in INSTRUCTIONS[key]
    t = INSTRUCTIONS["offer_person_in_language"]
    assert "?" not in t and "own language" in t and "English" in t


async def test_a_fixed_text_whose_values_are_in_its_facts_passes_and_makes_no_model_call():
    w = ScriptedWording([])
    b = brief(facts=F, plain="Your appeal deadline is 2026-03-18, and the allowed maximum is $1,450.00.")
    c = await compose(b, w, SETTINGS, STATUSES, fixed=True)
    assert c.source == "fixed" and c.text == b.plain_text() and w.briefs == [] and c.checks[-1].passed


def test_no_instruction_has_a_destination_phrase_of_its_own():
    for key, text in INSTRUCTIONS.items():
        assert "address on file" not in text.lower(), key
        if "email" in text.lower() or "summary" in text.lower():
            assert "the phrase given in the facts" in text and DESTINATION_RULE in text, key
