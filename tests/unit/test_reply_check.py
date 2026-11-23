from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.settings import load_settings
from claims_agent.states import State

_RC = load_settings().reply_check
WARY = _RC.wary_status_words
FORMS = _RC.status_forms
STATUSES = ("denied", "closed", "open")
F = (Fact("amounts", "Allowed maximum for claim CL-2048: 1450.00", "CL-2048"),
     Fact("appeal_deadline", "Appeal deadline for claim CL-2048: 2026-03-18 (13 days left)", "CL-2048"))


def brief(facts=(), verified=True, numbers=()):
    return ReplyBrief(state=State.PROCESS_CASE, parts=(ReplyPart("answer", facts=tuple(facts), plain="Plain."),),
                      mood="calm", caller_message="", previous_reply=None, english=True, verified=verified,
                      caller_claim_numbers=tuple(numbers))


def failures(text, **kw):
    return set(check_reply(text, brief(**kw), wary_status_words=WARY, data_statuses=STATUSES,
                           status_forms=FORMS).failures)


def test_addresses_numbers_and_emails_always_fail():
    assert "web_address" in failures("Upload at https://portal.example.com", facts=F)
    assert "phone_number" in failures("Call 650-521-2836", facts=F)
    assert "email_address" in failures("Write to help@example.com", facts=F)


def test_an_amount_not_in_the_facts_fails():
    assert "number_not_in_facts" in failures("You will receive $1,200.", facts=F)
    assert failures("The allowed maximum on CL-2048 is $1,450.", facts=F) == set()


def test_a_claim_number_not_in_the_facts_fails():
    assert "number_not_in_facts" in failures("Claim CL-2011 is closed.", facts=F)


def test_dates_must_come_from_the_facts():
    assert failures("You have 13 days, until March 18, 2026.", facts=F) == set()
    assert "month_not_in_facts" in failures("You have until April.", facts=F)


def test_approved_fails_unless_the_facts_say_it():
    assert "status_word" in failures("Good news, it was approved.", facts=F)
    assert failures("Consent is still pending.", facts=(Fact("consent", "Consent status: pending"),)) == set()


def test_a_claim_number_cannot_be_echoed_before_verification():
    assert "claim_detail_before_verification" in failures("I'll look at CL-2048 once verified.", verified=False, numbers=("CL-2048",))


def test_the_verification_rule_may_be_stated_before_verification():
    rule = (Fact("verification_rule", "At least 3 of these 5 are needed: full name, date of birth, phone number, email address, last 4 digits of the ID on file."),)
    assert failures("I need 3 of these 5 details before I can discuss a denied claim.", facts=rule, verified=False) == set()


def test_the_two_code_checks_are_reported_separately():
    r = check_reply("Call 650-521-2836 about your $9,999.", brief(facts=F), wary_status_words=WARY, data_statuses=STATUSES)
    assert r.leak == ("phone_number",) and r.invented == ("number_not_in_facts",)
    assert not r.passed


# ---- the stricter readings: the text may be adversarial or odd ----

def test_web_addresses_in_odd_shapes():
    for text in ["Visit WWW.Portal.Example.COM today", "go to portal . example . com", "portal[.]example(.)com",
                 "portal dot example dot com", "h t t p s : / / x", "see claims.example today", "Try 10.0.0.1",
                 "visit ex​ample.com", "ｅｘａｍｐｌｅ．ｃｏｍ"]:
        assert "web_address" in failures(text, facts=F), text


def test_ordinary_sentences_are_not_web_addresses():
    assert failures("Thanks. Your deadline is March 18, 2026. Anything else?", facts=F) == set()
    assert failures("Claim CL-2048 has a maximum of $1,450.00.", facts=F) == set()


def test_email_addresses_in_odd_shapes():
    for text in ["write to HELP@EXAMPLE.COM", "help [at] example dot com", "help(at)example.org", "help＠example.com",
                 "mail me @ the office"]:
        assert "email_address" in failures(text, facts=F), text


def test_phone_numbers_in_odd_shapes():
    for text in ["call (650) 521 2836", "call 650.521.2836", "call +1 650 521 2836", "call 6505212836",
                 "call 555-1234", "call six five zero five two one two eight three six",
                 "call ٦٥٠٥٢١٢٨٣٦", "call ６５０－５２１－２８３６"]:
        assert "phone_number" in failures(text, facts=F), text


def test_a_date_is_not_a_phone_number():
    assert "phone_number" not in failures("The deadline is 2026-03-18.", facts=F)


def test_numbers_with_separators_and_trailing_zeros():
    assert failures("The maximum is 1,450.00 dollars.", facts=F) == set()
    assert failures("The maximum is $1450.", facts=F) == set()
    assert "number_not_in_facts" in failures("The maximum is $1,451.", facts=F)
    assert "number_not_in_facts" in failures("The maximum is $14,500.", facts=F)
    assert "number_not_in_facts" in failures("The maximum is 1.450,50.", facts=F)


def test_a_number_inside_a_longer_run_is_not_a_match():
    assert "number_not_in_facts" in failures("Claim CL-20480 is open.", facts=F)
    assert "number_not_in_facts" in failures("Claim CL-204 is open.", facts=F)


def test_unicode_digits_are_read_as_digits():
    assert "number_not_in_facts" in failures("Claim CL-２０１１ is open.", facts=F)
    assert failures("Claim CL-２０４８ is open.", facts=F) == set()


def test_spelled_out_numbers_must_come_from_the_facts():
    assert "number_not_in_facts" in failures("You have twelve days left.", facts=F)
    assert "number_not_in_facts" in failures("It is one thousand two hundred dollars.", facts=F)
    assert failures("You have thirteen days left.", facts=F) == set()
    assert "number_not_in_facts" in failures("You have twelve days left.", facts=F)
    assert failures("One moment please.", facts=F) == set()


def test_dates_in_other_forms():
    assert failures("Until 18 March 2026.", facts=F) == set()
    assert failures("Until 2026-03-18.", facts=F) == set()
    assert failures("Until 03/18/2026.", facts=F) == set()
    assert "number_not_in_facts" in failures("Until 03/19/2026.", facts=F)
    assert "number_not_in_facts" in failures("Until 3/18/26.", facts=F)
    assert "month_not_in_facts" in failures("Until 18 MAR... no, 18 DEC.", facts=F)
    assert "month_not_in_facts" in failures("Until Sept 18.", facts=F)
    assert "month_not_in_facts" in failures("Hasta el 18 de abril.", facts=F)
    assert "month_not_in_facts" in failures("Jusqu'au 18 février.", facts=F)
    assert failures("Hasta el 18 de marzo.", facts=F) == set()


def test_may_is_a_month_only_next_to_a_number():
    assert failures("You may appeal until March 18.", facts=F) == set()
    assert "month_not_in_facts" in failures("Until May 18.", facts=F)
    assert "month_not_in_facts" in failures("Until the 18th of May.", facts=F)


def test_status_words_in_other_forms_and_case():
    for text in ["It was APPROVED.", "It was Approved!", "They will approve it.", "The approval came.",
                 "It is under   review.", "It was rejected.", "Still processing."]:
        assert "status_word" in failures(text, facts=F), text


def test_a_status_word_the_data_has_is_allowed():
    r = check_reply("It is pending.", brief(facts=F), wary_status_words=WARY + ["denied"], data_statuses=("pending",))
    assert r.passed
    r = check_reply("It is denied.", brief(facts=F), wary_status_words=WARY + ["denied"], data_statuses=("denied",))
    assert r.passed


def test_claim_numbers_before_verification_in_odd_shapes():
    for text in ["about cl-2048", "about CL 2048", "about CL‑2048", "about C L - 2 0 4 8"]:
        assert "claim_detail_before_verification" in failures(text, verified=False, numbers=("CL-2048",)), text
    assert "claim_detail_before_verification" not in failures("about CL-2048", verified=True, numbers=("CL-2048",), facts=F)


def test_before_verification_any_unfactual_number_fails():
    assert "number_not_in_facts" in failures("You mentioned claim CL-9999.", verified=False)


def test_invisible_characters_do_not_hide_things():
    assert "phone_number" in failures("call 650​-521‍-2836", facts=F)
    assert "email_address" in failures("help​@example.com", facts=F)


def test_non_english_text_is_checked_too():
    assert "number_not_in_facts" in failures("Su reclamo CL-2011 está cerrado.", facts=F)
    assert failures("Su fecha límite es 2026-03-18.", facts=F) == set()


def test_failures_are_listed_once_in_a_fixed_order():
    r = check_reply("Call 650-521-2836 or 650-521-2837 at help@x, $9,999 approved in April", brief(facts=F),
                    wary_status_words=WARY, data_statuses=STATUSES, status_forms=FORMS)
    assert r.failures == ("phone_number", "email_address", "number_not_in_facts", "month_not_in_facts", "status_word")


# ---- further cases ----
import time

import pytest

GUIDE = Fact("guidance", "A human representative should review what other proof may be accepted.")
STATUS = Fact("status", "Claim status: open")


def test_a_date_is_one_unit():
    for text in ["You have until March 13.", "You have until March 3, 2026.", "Until 3/13.", "You have 18 days left.",
                 "You have 3 days left.", "Until the nineteenth.", "Until Tuesday.",
                 "Until next Tuesday.", "Until tomorrow.", "Next week.", "In a couple of weeks.", "Until March 18, 2027."]:
        assert failures(text, facts=F) & {"number_not_in_facts", "month_not_in_facts"}, text
    assert failures("Until March 18. That is 13 days left.", facts=F) == set()
    assert failures("Until March 18th, 2026 (13 days left).", facts=F) == set()


def test_numbers_are_compared_by_type():
    for text in ["You will receive $13.", "You will receive $2,048.", "You will receive $2048.", "13%", "thirteen percent",
                 "between 13 and 18 days", "Until 2048.", "That is 13 dollars."]:
        assert "number_not_in_facts" in failures(text, facts=F), text
    assert failures("Claim CL-2048 allows $1,450.00 and 13 days remain.", facts=F) == set()


def test_facts_side_by_side_are_not_a_phone_number():
    for text in ["Until March 18 2026 (13 days left).", "Until March 18 2026 - 13 days left.", "Claim CL-2048 (1450.00 maximum)",
                 "Claim CL-2048. 1450.00 is the maximum."]:
        assert "phone_number" not in failures(text, facts=F), text
        assert failures(text, facts=F) == set(), text


def test_obfuscated_addresses():
    for text in ["portal (dot) example (dot) com", "portal [dot] example [dot] com", "portal-dot-example-dot-com",
                 "portal dot example dot claims", "portal dot example dot ai", "portal dot example dot co dot uk",
                 "portal . example . claims", "portal\u3002example\u3002com", "portal\uff61example\uff61com",
                 "portal\u00b7example\u00b7com", "portal\u2027example\u2027com", "go to example dot claims slash appeals"]:
        assert "web_address" in failures(text, facts=F), text
    for text in ["email help at example dot claims", "help AT example DOT co", "help (at) example (dot) org"]:
        assert "email_address" in failures(text, facts=F) or "web_address" in failures(text, facts=F), text
    assert "email_address" in failures("help AT example DOT co", facts=F)


def test_status_words_by_exact_form_and_use():
    assert "status_word" in failures("Your claim was accepted.", facts=(GUIDE,))
    assert failures("Other proof may be accepted.", facts=(GUIDE,)) == set()
    assert "status_word" in failures("Your claim CL-2048 is pending.", facts=(Fact("guidance", "what is still pending"),))
    assert failures("Your claim is pending.", facts=(Fact("status", "Claim status: pending"),)) == set()
    assert failures("The appeal process takes time.", facts=F) == set()
    assert failures("Let me explain the process.", facts=F) == set()
    assert failures("The average processing time is short.", facts=(Fact("guidance", "average processing time"),)) == set()
    assert "status_word" in failures("Your claim is processing now.", facts=(Fact("guidance", "average processing time"),))
    assert "status_word" in failures("I'm processing your request.", facts=F)
    assert "status_word" in failures("Your claim was approved.", facts=(Fact("guidance", "Surgery needs pre-approval"),))
    assert "status_word" in failures("It was accepted.", facts=(Fact("guidance", "we accept scanned copies"),))


def test_a_reply_that_is_too_long_is_rejected_before_any_pattern_runs():
    r = check_reply("a" * 20000, brief(facts=F), wary_status_words=WARY, data_statuses=STATUSES, status_forms=FORMS)
    assert r.failures == ("too_long",) and r.leak == () and r.invented == () and r.other == ("too_long",)
    assert failures("a " * 749 + "ok", facts=F) == set()
    assert failures("a " * 750 + "ok", facts=F) == {"too_long"}


def test_the_check_on_a_long_reply_is_fast():
    texts = ["a" * 1500, "a " * 750, "1 " * 750, "a." * 750, ("Claim CL-2048 is open. " * 70)[:1500]]
    t0 = time.perf_counter()
    for text in texts:
        failures(text, facts=F)
    assert time.perf_counter() - t0 < 1.0


# ---- further cases ----
RULE = (Fact("verification_rule", "At least 3 of these 5 are needed: full name, date of birth, phone number, email address, last 4 digits of the ID on file."),)
DOCS = (Fact("documents", "Required documents (2): pathology report, office note"),)


def test_a_number_words_are_numbers():
    assert failures("At least three of these five details are needed.", facts=RULE, verified=False) == set()
    assert failures("You have thirteen days left.", facts=F) == set()
    for text in ["Until the first of March.", "A dozen days.", "In a day.", "Until the nineteenth.", "Twelve days.",
                 "It is fourteen hundred dollars."]:
        assert failures(text, facts=F) & {"number_not_in_facts", "month_not_in_facts"}, text
    assert failures("It is fourteen hundred dollars.", facts=(Fact("amounts", "Allowed maximum: 1400.00"),)) == set()


def test_a_vague_quantities_and_relative_dates_need_the_same_words():
    for text in ["It takes a few days.", "Several days.", "A couple of days.", "Several documents.", "Tonight.", "Yesterday.",
                 "On Mon.", "By Fri.", "Next month.", "This week.", "Last year.", "By end of month.", "By the end of the month.",
                 "Around mid-March."]:
        assert failures(text, facts=F) & {"number_not_in_facts", "month_not_in_facts"}, text
    assert failures("It can take a few days.", facts=(Fact("guidance", "It takes a few days."),)) == set()
    assert failures("Until next Tuesday.", facts=(Fact("guidance", "Send it by next Tuesday."),)) == set()


def test_b_what_a_bare_number_may_match():
    assert failures("The maximum is 1,450.", facts=F) == set()
    assert failures("Until the 18th.", facts=F) == set()
    assert failures("Until the eighteenth of March.", facts=F) == set()
    assert failures("Thirteen.", facts=F) == set()
    assert failures("In 2026.", facts=F) == set()
    for text in ["In 2027.", "The amount is 2,048.", "That is $13.", "That is 2048.", "The 19th."]:
        assert "number_not_in_facts" in failures(text, facts=F), text


def test_c_bullets_and_other_languages_are_not_addresses():
    assert failures("You need two documents:\n\u2022 the pathology report\n\u2022 the office note", facts=DOCS) == set()
    assert failures("Pode pedir para falar com uma pessoa a qualquer momento.", facts=F) == set()
    assert failures("Puede pedir hablar con una persona en cualquier momento.", facts=F) == set()
    assert "web_address" in failures("portal (dot) example (dot) com", facts=F)
    assert "web_address" in failures("portal \u3002 example\u3002com", facts=F)


def test_d_a_claim_state_in_other_shapes():
    facts = (GUIDE, *F)
    for text in ["CL-2048 has been accepted.", "Your case was accepted.", "Your claim\u2019s accepted.",
                 "Your claim for the dental visit was accepted.", "We have accepted your claim."]:
        assert "status_word" in failures(text, facts=facts), text
    assert failures("Other proof may be accepted.", facts=(GUIDE,)) == set()


def test_e_processing_with_and_without_the_words_in_the_facts():
    proc = (Fact("guidance", "average processing time"),)
    assert failures("I'm processing your request.", facts=proc) == set()
    assert "status_word" in failures("Your claim is processing.", facts=proc)
    assert "status_word" in failures("I'm processing your request.", facts=F)


# ---- further cases ----

def test_an_ordinal_may_only_equal_the_day_of_a_fact_date():
    for text in ["You have until the 13th to appeal.", "Your appeal is due on the thirteenth.", "Until the 13th.",
                 "By the second.", "By the 1st.", "Due on the 2nd."]:
        assert "number_not_in_facts" in failures(text, facts=F), text
    for text in ["Until the 18th.", "Until the eighteenth.", "Your appeal is due on the 18th.", "The deadline is the eighteenth of March."]:
        assert failures(text, facts=F) == set(), text


def test_only_claim_units_are_durations_and_today_is_filler():
    for text in ["Give me a minute while I check.", "Just a minute, please.", "One second.", "Hold on, an hour at most is not needed.",
                 "Anything else I can help you with today?", "I have a couple of questions."]:
        assert failures(text, facts=F) == set(), text
    assert "number_not_in_facts" in failures("In a couple of weeks.", facts=F)
    assert "number_not_in_facts" in failures("You have five business days.", facts=F)
    assert failures("You have 13 days left.", facts=F) == set()


def test_vague_words_count_only_before_claim_units():
    assert "number_not_in_facts" in failures("It takes several documents.", facts=F)
    assert "number_not_in_facts" in failures("In a few weeks.", facts=F)
    assert failures("It takes a few days.", facts=(Fact("guidance", "It takes a few days."),)) == set()


_ORDINARY_FACTS = (
    Fact("status", "Claim CL-2048: status denied.", "CL-2048"),
    Fact("denial_reason", "Denial reason for claim CL-2048: the review file did not include the pathology report and the treating provider office note.", "CL-2048"),
    Fact("documents", "Documents needed for claim CL-2048 (2): pathology report, office note.", "CL-2048"),
    Fact("appeal_deadline", "Appeal deadline for claim CL-2048: 2026-03-18 (13 days left).", "CL-2048"),
    Fact("amounts", "Amounts for claim CL-2048: expected reimbursement 0.00; allowed maximum 1450.00; paid 0.00; fee schedule rate 1450.00.", "CL-2048"),
    Fact("guidance", "For claim CL-2048, please submit the pathology report and the office note within a week.", "CL-2048"),
    Fact("guidance", "The average processing time is usually less than a week."),
)
_RULE_ONLY = (Fact("verification_rule", "At least 3 of these 5 are needed: full name, date of birth, phone number, email address, last 4 digits of the ID on file."),)


@pytest.mark.parametrize("text", [
    "Your claim CL-2048 was denied because the review file did not include the pathology report and the office note.",
    "You have 13 days left to appeal; the deadline is March 18, 2026.",
    "The deadline is 18 March.",
    "You'll need to send two documents: the pathology report and the office note.",
    "Please send both documents within a week.",
    "Once we receive them, processing usually takes less than a week.",
    "The allowed maximum is $1,450.00 and nothing has been paid yet.",
    "I'm sorry for the frustration. I understand this has been a long process.",
    "Is there anything else I can help you with today?",
    "I can connect you with a person if you'd like.",
    "Thanks, Margaret. One moment while I look at that.",
    "First, send the pathology report. Second, send the office note.",
    "That was the first step; the next step is to send the documents.",
    "Would you like a summary sent to the email on your policy?",
    "Give me a minute while I check.",
    "First of all, I'm sorry about this.",
    "Let me first check your claim.",
])
def test_ordinary_replies_about_a_denied_claim_must_keep_passing(text):
    assert failures(text, facts=_ORDINARY_FACTS) == set()


@pytest.mark.parametrize("text", [
    "I need at least three of these five details: your full name, date of birth, phone number, email address, or the last four digits of the ID on file.",
    "Could you give me one more detail?",
    "First I need to verify your identity.",
])
def test_ordinary_replies_before_verification_must_keep_passing(text):
    assert failures(text, facts=_RULE_ONLY, verified=False) == set()


def test_before_verification_a_claim_detail_is_not_repeated_back():
    text = "I understand you're calling about a denied healthcare claim from January. I can't share anything about a claim until I've verified you."
    assert "month_not_in_facts" in failures(text, facts=_RULE_ONLY, verified=False)


def test_a_comma_makes_a_sequence_only_when_the_ordinal_opens_a_sentence():
    for text in ["Send it by the thirteenth, please.", "The deadline is the thirteenth, so please hurry.",
                 "Please send it by the second, if you can.", "On the third, at the latest."]:
        assert "number_not_in_facts" in failures(text, facts=F), text
    assert failures("First, send the pathology report. Second, send the office note.", facts=F) == set()
    assert failures("Steps to take:\nFirst, send the report.\nSecond, send the note.", facts=F) == set()


def test_first_is_a_date_only_after_the_or_a_month_or_before_of_a_month():
    for text in ["By the first.", "Until the first of March.", "Until March first.", "Until first of March."]:
        assert failures(text, facts=F) & {"number_not_in_facts", "month_not_in_facts"}, text
    for text in ["First of all, I'm sorry about this.", "Let me first check your claim."]:
        assert failures(text, facts=F) == set(), text


# ---- an acknowledging sentence for an upset caller ------------------------------------------------------------

_EMPATHY = [
    "I understand this is frustrating, and I'm sorry you've had to repeat yourself.",
    "I hear you, and I know this has been a stressful wait.",
    "I'm sorry this has been so upsetting.",
    "I can tell this is the second time you've asked, and I understand why you're annoyed.",
    "That sounds really worrying. I'm here to help you get this sorted out.",
    "I know this feels like one more hoop to jump through.",
    "Thanks for bearing with me for a moment.",
]
_DENIAL_FACTS = (Fact("status", "Claim CL-2048 status: denied.", "CL-2048"),
                 Fact("denial_reason", "Claim CL-2048 was denied because the review file did not include the "
                                       "pathology report.", "CL-2048"))


@pytest.mark.parametrize("sentence", _EMPATHY)
def test_an_acknowledging_sentence_passes_before_and_after_verification(sentence):
    ask = " To verify who you are, I need 3 of these 5 details: full name, date of birth, phone number, email address or the ID on file."
    assert failures(sentence + ask, facts=_RULE_ONLY, verified=False) == set()
    answer = " Claim CL-2048 was denied because the review file did not include the pathology report."
    assert failures(sentence + answer, facts=_DENIAL_FACTS) == set()


@pytest.mark.parametrize("text, failure", [
    ("I'm sorry for the trouble. Your claim was approved after all.", "status_word"),
    ("I understand this is frustrating. Your claim is now under review.", "status_word"),
    ("I'm sorry about this. You'll hear back within 5 days.", "number_not_in_facts"),
    ("I know this is stressful. You should get $500 back.", "number_not_in_facts"),
    ("I'm so sorry. It will be sorted by Friday.", "number_not_in_facts"),
    ("I understand. You can call 650-521-2836 for help.", "phone_number"),
])
def test_an_acknowledging_sentence_does_not_let_an_invented_fact_through(text, failure):
    assert failure in failures(text, facts=_DENIAL_FACTS)
    assert failure in failures(text, facts=_RULE_ONLY, verified=False)


@pytest.mark.parametrize("text", [
    "I'm sorry your claim was denied, but I need to verify you first.",
    "I understand you're upset that the claim has been denied.",
    "I'm sorry to hear your claim is closed. Could you give me your full name?",
    "I know it's frustrating that they denied your claim.",
])
def test_before_verification_a_claims_status_is_never_stated(text):
    assert "claim_detail_before_verification" in failures(text, facts=_RULE_ONLY, verified=False)


@pytest.mark.parametrize("text", [
    "I need 3 of these 5 details before I can discuss a denied claim.",
    "I can only share claim details, including why a claim was or wasn't paid, once you're verified.",
    "Claim details are protected, so I need to verify who you are first.",
])
def test_before_verification_talking_about_verification_is_not_a_status(text):
    assert failures(text, facts=_RULE_ONLY, verified=False) == set()
