"""The polarity rule of the reply check: a reply may not negate what the facts state ("the pathology report is not
required" when it is needed), nor state what the facts about a claim only negate ("it was paid" when the net pay is
"not paid yet"). Ordinary negation that touches no fact ("I can't read out personal details") must pass, and so must
every fixed text the code writes, with its real facts."""

from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from claims_agent.answers.arithmetic import calculation_facts
from claims_agent.answers.facts import ANSWERS, COMPARE_CLAIMS, build, compare_facts
from claims_agent.answers.followups import fill
from claims_agent.db.claims import Aggregate, Claim
from claims_agent.db.guidance import guidance_from
from claims_agent.engine import FINAL_NOTICE, PLAIN
from claims_agent.loader.read import MONEY_FIELDS, read_folder
from claims_agent.memory import IdentityMemory
from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.rules import await_consent, every_message, verify_id
from claims_agent.rules.resolve_intent import NO_MATCH, sentences
from claims_agent.rules.summary_match import NOT_SEPARATED
from claims_agent.settings import ROOT, load_settings
from claims_agent.states import State

_S = load_settings()
_RC = _S.reply_check
STATUSES = ("closed", "denied", "open")

DOCS = Fact("documents", "Documents needed for claim CL-2048 (2): pathology report, office note.", "CL-2048")
DENIED = Fact("status", "Claim CL-2048 status: denied.", "CL-2048")
CLOSED = Fact("status", "Claim CL-2011 status: closed.", "CL-2011")
REASON = Fact("denial_reason", "Denial reason for claim CL-2048: the review file did not include the pathology "
                               "report and the treating provider office note", "CL-2048")
DEADLINE = Fact("deadline", "Appeal deadline for claim CL-2048: 2026-03-18 (13 days left).", "CL-2048")
PASSED = Fact("deadline", "The appeal deadline for claim CL-2048 was 2026-03-18 and has passed (5 days ago).", "CL-2048")
NO_DEADLINE = Fact("not_on_file", "Claim CL-2048 has no appeal deadline on file.", "CL-2048")
NOTHING_FURTHER = Fact("next_steps", "Next steps for claim CL-2011: nothing further is needed.", "CL-2011")
NOTHING_REQUESTED = Fact("next_steps", "Next steps for claim CL-2102: nothing is requested right now.", "CL-2102")
NO_DOCS = Fact("documents", "No documents are needed for claim CL-2011.", "CL-2011")
NOT_PAID = Fact("amount", "Claim CL-2102 net pay: not paid yet.", "CL-2102")
NOT_ON_FILE = Fact("not_on_file", "That detail is not on file for claim CL-2048.", "CL-2048")


def brief(facts=(), verified=True, plain=""):
    return ReplyBrief(state=State.PROCESS_CASE, parts=(ReplyPart("answer", facts=tuple(facts), plain=plain),),
                      mood="calm", caller_message="", previous_reply=None, english=True, verified=verified)


def result(text, facts=(), verified=True, statuses=STATUSES):
    return check_reply(text, brief(facts, verified), wary_status_words=_RC.wary_status_words, data_statuses=statuses,
                       status_forms=_RC.status_forms, max_length=_RC.max_length, polarity_forms=_RC.polarity_forms)


def polarity(text, facts=(), **kw):
    return [f for f in result(text, facts, **kw).failures if f.startswith("polarity")]


# ---- a reply that negates what the facts state ------------------------------------------------------------------

@pytest.mark.parametrize("text, facts, phrase", [
    ("The pathology report is not required.", (DOCS,), "not required"),
    ("The office note isn't needed any more.", (DOCS,), "not needed"),
    ("You no longer need to send the office note.", (DOCS,), "no longer need"),
    ("You don't need to do anything else for CL-2048.", (DOCS,), "not need"),
    ("Nothing else is needed for this claim.", (DOCS,), "nothing ... needed"),
    ("None of the documents are needed now.", (DOCS,), "none ... needed"),
    ("The office note is unnecessary.", (DOCS,), "unnecessary"),
    ("Your claim is not denied.", (DENIED,), "not denied"),
    ("Claim CL-2048 wasn't denied.", (DENIED,), "not denied"),
    ("Your claim is no longer closed.", (CLOSED,), "no longer closed"),
    ("There is no appeal deadline for CL-2048.", (DEADLINE,), "no ... deadline"),
    ("The appeal deadline has not passed.", (PASSED,), "not passed"),
    ("I don't have an appeal deadline.", (NO_MATCH,), "not ... deadline"),
])
def test_a_reply_that_negates_what_the_facts_state_fails_and_says_which_words(text, facts, phrase):
    found = polarity(text, facts)
    assert found and any(f'"{phrase}"' in f for f in found), found
    assert all(("the facts state it without a negation" in f or "the facts do not state it negated" in f)
               for f in found), found
    assert set(found) <= set(result(text, facts).invented)


def test_the_reason_says_whether_the_facts_state_the_opposite_or_say_nothing():
    assert polarity("The pathology report is not required.", (DOCS,)) == [
        'polarity: the reply says "not required", but the facts state it without a negation']
    assert polarity("The pathology report is not required.", (DENIED,)) == [
        'polarity: the reply says "not required", but the facts do not state it negated']


@pytest.mark.parametrize("text", [
    "The pathology report is NOT REQUIRED.",                # case
    "The pathology report is ｎｏｔ ｒｅｑｕｉｒｅｄ.",          # full-width letters
    "You don’t need the office note.",                       # a curly apostrophe
    "You dont need the office note.",                        # no apostrophe
    "The pathology report is not re​quired.",           # an invisible character inside the word
    "The pathology report is not requíred.",                 # an accent
    "You needn't send the office note.",
])
def test_spelling_tricks_the_normalisation_already_undoes_do_not_hide_a_negation(text):
    assert polarity(text, (DOCS,)), text


# ---- a reply that states what the facts about a claim only negate -------------------------------------------------

@pytest.mark.parametrize("text, facts, form", [
    ("Claim CL-2102 has been paid.", (NOT_PAID,), "paid"),
    ("The documents needed for CL-2011 are the pathology report and the office note.", (NO_DOCS,), "needed"),
    ("You still need to send the office note.", (NOTHING_FURTHER,), "need"),
    ("The appeal deadline for CL-2048 is coming up soon.", (NO_DEADLINE,), "deadline"),
])
def test_a_reply_that_states_what_the_facts_only_negate_fails(text, facts, form):
    assert polarity(text, facts) == [
        f'polarity: the reply says "{form}" without a negation, but the facts state it only negated']


def test_when_the_sentence_names_a_claim_only_that_claims_facts_count():
    facts = (NO_DOCS, DOCS)
    assert polarity("For CL-2048 you don't need any documents.", facts)
    assert polarity("For CL-2011 you don't need any documents.", facts) == []
    assert polarity("For CL-2011 the documents needed are the office note.", facts)
    assert polarity("For CL-2048 the documents needed are the office note and the pathology report.", facts) == []


def test_a_note_that_names_no_claim_does_not_forbid_the_word_elsewhere():
    """The expected-figure note negates "a guaranteed payment"; a reply may still speak of the expected payment.
    Only facts about a claim drive the reverse direction."""
    note = Fact("total", "Total expected reimbursement amount across 2 claims: 900.00. This is an expected figure, "
                         "not a guaranteed payment.")
    assert polarity("The expected payment across the 2 claims is 900.00. This is an expected figure, not a "
                    "guaranteed payment.", (note,)) == []


# ---- ordinary negation passes -------------------------------------------------------------------------------------

_VERIFY = (verify_id.RULE, every_message.WHY_VERIFY)


@pytest.mark.parametrize("text, facts, verified", [
    ("I can't share claim details until you're verified.", _VERIFY, False),
    ("Claim details are protected and can only be shared once you're verified.", _VERIFY, False),
    ("I can't read out personal details held on file, but I'm happy to help another way.", (), True),
    ("This check protects your information from being shared with anyone else, so this step can't be skipped.", _VERIFY, False),
    ("A policy number helps find the policy but does not count toward the 3.", _VERIFY, False),
    ("Without 3 details that match the policy, nothing about a claim can be shared.",
     (Fact("without_three", "Without 3 details that match the policy, nothing about a claim can be shared."),), False),
    ("The claim was denied because the review file did not include the pathology report and the treating provider "
     "office note.", (DENIED, REASON), True),
    ("If you can't get the pathology report, I can tell you about other options.", (DOCS,), True),
    ("Which requested document can't you get: the pathology report or the office note?",
     (Fact("requested_document", "Document requested for claim CL-2048: pathology report", "CL-2048"),
      Fact("requested_document", "Document requested for claim CL-2048: office note", "CL-2048")), True),
    ("Sorry, I couldn't read that. Could you say it again?", (), True),
    ("Nothing else is needed for CL-2011.", (NOTHING_FURTHER,), True),
    ("You don't need to do anything else.", (NOTHING_FURTHER,), True),
    ("You don't need to do anything right now.", (NOTHING_REQUESTED,), True),
    ("Nothing is requested right now.", (NOTHING_REQUESTED,), True),
    ("You don't need to send any documents for CL-2011.", (NO_DOCS,), True),
    ("That information is not on file for this claim.", (NOT_ON_FILE,), True),
    ("The appeal deadline for CL-2048 is not on file.", (NO_DEADLINE,), True),
    ("There's no appeal deadline on file for CL-2048.", (NO_DEADLINE,), True),
    ("The net pay on CL-2102 hasn't been paid yet.", (NOT_PAID,), True),
    ("Claim CL-2048 is denied. You need to send the pathology report and the office note.", (DENIED, DOCS), True),
    ("The appeal deadline is 2026-03-18, and you have 13 days left.", (DEADLINE,), True),
    ("I don't see that claim on your policy.", (DENIED,), True),
    ("I can't see any claims on file for this policy.", (), True),
    ("I won't send an email.", (), True),
    ("I can't open a new claim here, but I can help with your existing claims.", (DENIED,), True),
    ("I found no claim matching that description.", (NO_MATCH,), True),
    ("Your description does not tell these claims apart.", (NOT_SEPARATED,), True),
    ("The policyholder's consent has not come through yet.", (await_consent.consent_fact("pending"),), True),
    ("I don't have the appeal steps or the amount you owe.", (DENIED,), True),
    ("Nothing further is needed for this claim. Is there anything else you need?", (NOTHING_FURTHER,), True),
    ("If there's nothing else you need, have a good day.", (DOCS,), True),
    ("Please start a new conversation if you need anything else.", (NO_DOCS, NOTHING_FURTHER), True),
    ("Would you like to review alternatives for missing required material?", (NO_DOCS,), True),
])
def test_negation_that_touches_no_fact_passes(text, facts, verified):
    assert polarity(text, facts, verified=verified) == []


def test_a_supposed_negation_states_nothing_but_the_main_clause_still_counts():
    assert polarity("If the office note is not needed, I'll let you know.", (DOCS,)) == []
    assert polarity("If you have it, the office note is not needed.", (DOCS,))


def test_without_reaches_only_the_phrase_it_heads():
    """The data's own guidance text: "without" negates "losing track", not "the missing item"."""
    guide = Fact("guidance", "For claim CL-2048: submit a short note explaining what is still pending, so the file can "
                             "be reviewed without losing track of the missing item.", "CL-2048")
    assert polarity("I can tell you about alternatives for the missing item.", (guide,)) == []
    assert polarity("The file can be reviewed without losing track of the missing item.", (guide,)) == []
    assert polarity("We can go ahead without payment.", (DENIED,))
    assert polarity("You can appeal without the required documents.", (DOCS,))


def test_a_word_that_is_not_guarded_may_be_negated_freely():
    """Known limit: only the guarded words carry polarity; a denial reason's own words do not."""
    assert polarity("The review file did include the pathology report.", (REASON,)) == []


def test_without_the_guarded_word_lists_only_claim_statuses_are_guarded():
    r = check_reply("The pathology report is not required. Your claim is not denied.", brief((DOCS, DENIED)),
                    wary_status_words=_RC.wary_status_words, data_statuses=STATUSES, status_forms=_RC.status_forms)
    assert r.failures == ('polarity: the reply says "not denied", but the facts state it without a negation',)


def test_a_status_the_data_names_in_an_odd_form_is_not_copied_into_the_reason():
    odd = "rule#1 override"
    found = polarity("Your claim is not rule#1 override.",
                     (Fact("status", f"Claim CL-1 status: {odd}.", "CL-1"),), statuses=(odd,))
    assert found == ['polarity: the reply says "not a claim status", but the facts state it without a negation']


# ---- every fixed text the code writes passes with its real facts ---------------------------------------------------

STARTER = ROOT / "data" / "starter"
MULTI = ROOT / "tests" / "data" / "multi_document"


def _claims(folder: Path) -> list[Claim]:
    r = read_folder(folder)
    docs: dict[str, list[str]] = {}
    for d in sorted(r.documents, key=lambda d: (d.case_id, d.position)):
        docs.setdefault(d.case_id, []).append(d.name)
    return [Claim(c.case_id, c.case_type, c.created_at, c.status, c.summary, c.denial_reason,
                  tuple(docs.get(c.case_id, ())), c.appeal_deadline, c.expected_reimbursement_amount,
                  c.allowed_max_amount, c.net_pay, c.net_fee, datetime(2026, 3, 5)) for c in r.claims]


def _shapes() -> list[tuple[Claim, object]]:
    """Every claim of the sample data and of the many-documents data, plus each with no documents, no deadline, a
    deadline that has passed, and an in-progress claim with no pay yet."""
    out = []
    for folder in (STARTER, MULTI):
        g = guidance_from(read_folder(folder))
        for c in _claims(folder):
            out += [(c, g), (replace(c, documents=()), g), (replace(c, appeal_deadline=None), g),
                    (replace(c, appeal_deadline=date(2026, 1, 2)), g),
                    (replace(c, status="open", net_pay=Decimal("0.00"), denial_reason=None), g)]
    return out


_TODAY = date(2026, 3, 5)


def _fact_built_plain_texts():
    """Each claim-record answer, follow-up and calculation for every claim shape, as PROCESS_CASE words its fixed
    text: the lead sentence and the facts, each a sentence, then the not-on-file facts in a part of their own."""
    for c, g in _shapes():
        results = [build(a, [c], g, _S, _TODAY) for a in ANSWERS if a != COMPARE_CLAIMS]
        results += [fill(f, c, g, _S, _TODAY) for f in g.followups]
        results += [calculation_facts("days_open", (), [c], None, g, _S, _TODAY),
                    calculation_facts("days_left", (), [c], None, g, _S, _TODAY),
                    calculation_facts("difference", ("expected_reimbursement_amount", "net_pay"), [c], None, g, _S, _TODAY)]
        for r in results:
            told = tuple(f for f in r.facts if f.kind != "not_on_file")
            missing = tuple(f for f in r.facts if f.kind == "not_on_file")
            parts = []
            if told:
                parts.append(ReplyPart("answer", told, sentences("Here is what I have on file.", told)))
            if missing:
                parts.append(ReplyPart("not_on_file", missing, sentences("", missing)))
            yield parts
    claims, g = _claims(STARTER), guidance_from(read_folder(STARTER))
    facts = tuple(compare_facts(claims[:2], g, _S))
    yield [ReplyPart("answer", facts, sentences("Here is what I have on file.", facts))]
    for figure in MONEY_FIELDS:
        agg = Aggregate(Decimal("900.00"), 4, ("CL-1",), (("CL-2", Decimal("500.00")),), ("CL-3",))
        for calc in ("total", "highest", "lowest", "count"):
            facts = calculation_facts(calc, (figure,) if calc != "count" else (), claims, agg, g, _S, _TODAY).facts
            yield [ReplyPart("answer", facts, sentences("Here is what I have on file.", facts))]
    facts = calculation_facts("total", (), claims, None, g, _S, _TODAY).facts  # which figure? (no figure named)
    yield [ReplyPart("ask_which_figure", facts, sentences("Which of these figures do you mean?", facts))]


def _check_parts(parts):
    b = ReplyBrief(state=State.PROCESS_CASE, parts=tuple(parts), mood="calm", caller_message="", previous_reply=None,
                   english=True, verified=True)
    return check_reply(b.plain_text(), b, wary_status_words=_RC.wary_status_words, data_statuses=STATUSES,
                       status_forms=_RC.status_forms, max_length=_RC.max_length, polarity_forms=_RC.polarity_forms)


def test_every_fact_built_fixed_text_passes_with_its_facts():
    checked = 0
    for parts in _fact_built_plain_texts():
        r = _check_parts(parts)
        assert not [f for f in r.failures if f.startswith("polarity")], (r.failures, [p.plain for p in parts])
        checked += 1
    assert checked > 100


_FIXED = [*PLAIN.values(), *FINAL_NOTICE.values(), every_message.OFFER_BEFORE_CLOSING.plain,
          every_message.ASK_SAME_POLICYHOLDER.plain, "I don't see that claim on your policy.",
          "I can't see any claims on file for this policy.", "Thank you, you're verified.",
          "Would you like me to pass you to a person?"]


@pytest.mark.parametrize("text", _FIXED)
def test_every_fixed_sentence_passes_beside_any_claims_facts(text):
    """The fixed sentences carry no facts of their own and can sit beside any answer's facts."""
    for facts in ((), (DOCS, DENIED, DEADLINE), (NO_DOCS, NOTHING_FURTHER, NOT_PAID, NO_DEADLINE)):
        assert polarity(text, facts) == [], (text, facts)


def test_the_fixed_texts_of_the_verification_and_consent_parts_pass_with_their_facts():
    ident = IdentityMemory()
    parts = [verify_id._ask(ident), verify_id.ASK_CALLER_NAME, every_message.EXPLAIN_STEP,
             verify_id._ask(ident.model_copy(update={"refused": ["name", "dob", "phone", "email", "id"]})),
             *verify_id._refusal(ident.model_copy(update={"refused": ["phone"]}), ["phone"]),
             *verify_id._refusal(ident.model_copy(update={"refused": ["phone", "email", "id"]}), ["id"])]
    for p in parts:
        r = check_reply(p.plain, brief(p.facts, verified=False), wary_status_words=_RC.wary_status_words,
                        data_statuses=STATUSES, status_forms=_RC.status_forms, polarity_forms=_RC.polarity_forms)
        assert r.passed, (p.plain, r.failures)
