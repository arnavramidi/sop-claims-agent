"""Claim-record answers: facts built by code from a claim and the guidance data."""

from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from claims_agent.db.claims import Claim
from claims_agent.db.guidance import guidance_from
from claims_agent.loader.read import read_folder
from claims_agent.settings import ROOT, load_settings

STARTER = ROOT / "data" / "starter"
MULTI = ROOT / "tests" / "data" / "multi_document"
UNMARKED = ROOT / "tests" / "data" / "unmarked_followup"


def _claims(folder: Path) -> dict[str, Claim]:
    r = read_folder(folder)
    docs: dict[str, list[str]] = {}
    for d in sorted(r.documents, key=lambda d: (d.case_id, d.position)):
        docs.setdefault(d.case_id, []).append(d.name)
    return {c.case_id: Claim(c.case_id, c.case_type, c.created_at, c.status, c.summary, c.denial_reason,
                             tuple(docs.get(c.case_id, ())), c.appeal_deadline, c.expected_reimbursement_amount,
                             c.allowed_max_amount, c.net_pay, c.net_fee, datetime(2026, 3, 5))
            for c in r.claims}


_S, _M = _claims(STARTER), _claims(MULTI)
CL2048, CL2011, CL2102 = _S["CL-2048"], _S["CL-2011"], _S["CL-2102"]
CL1899_WITH_ONE_DOC = replace(_S["CL-1899"], documents=("diagnosis report",))
CL5001 = _M["CL-5001"]
G, GM, S, TODAY = guidance_from(read_folder(STARTER)), guidance_from(read_folder(MULTI)), load_settings(), date(2026, 3, 5)


def text(r):
    return " ".join(f.text for f in r.facts)


def followup(g, topic):
    return next(f for f in g.followups if f.topic == topic)


from claims_agent.answers.documents import match_document
from claims_agent.answers.facts import available, build, opening
from claims_agent.answers.followups import fill


def test_a1_days_left_are_counted_against_today():
    assert "2026-03-18" in text(build("appeal_deadline", [CL2048], G, S, TODAY)) and "13 days" in text(build("appeal_deadline", [CL2048], G, S, TODAY))


def test_p1_document_names_that_differ_between_files_still_match():
    assert match_document("pathology report", G.document_text) == "original pathology report"
    assert match_document("office note", G.document_text) == "treating provider office note"


def test_p2_a_document_with_no_guidance_uses_the_default():
    assert match_document("diagnosis report", G.document_text) is None


def test_several_matches_use_the_default():
    assert match_document("photos", GM.document_text) is None


def test_p3_an_auto_claim_with_nothing_outstanding_has_no_followups():
    a = available(CL2102, G, 1, S)
    assert "documents_needed" not in a and not {f.topic for f in G.followups} & set(a)
    assert "photos" not in text(build("next_steps", [CL2102], G, S, TODAY)) and "estimate" not in text(build("next_steps", [CL2102], G, S, TODAY))


def test_p4_the_fallback_line_exists_only_with_documents_outstanding():
    assert "fallback" in available(CL2048, G, 1, S) and "fallback" not in available(CL2011, G, 1, S)


def test_p5_no_type_advice_is_invented_for_dental():
    assert "medical claims" not in text(build("documents_needed", [CL1899_WITH_ONE_DOC], G, S, TODAY))


def test_type_advice_and_count_are_in_the_documents_fact():
    t = text(build("documents_needed", [CL2048], G, S, TODAY))
    assert "medical claims" in t and "(2)" in t and "pathology report, office note" in t


def test_p11_zero_paid_on_an_open_claim_is_not_paid_yet():
    assert "not paid yet" in text(build("amounts", [CL2102], G, S, TODAY))


def test_a_zero_paid_on_a_denied_claim_is_stated_as_a_figure():
    t = text(build("amounts", [CL2048], G, S, TODAY))
    assert "not paid yet" not in t and "0.00" in t and "1450.00" in t


def test_a9_a_figure_that_failed_to_load_is_unavailable():
    assert "unavailable" in text(build("amounts", [replace(CL2011, net_pay=None)], G, S, TODAY))


def test_the_glossary_caveat_is_kept_whole():
    assert "In some systems" in text(build("amounts", [CL2011], G, S, TODAY))


def test_not_on_file_always_offers_a_person_and_is_not_grounded():
    r = build("not_on_file", [CL2048], G, S, TODAY)
    assert r.offer_person and not r.grounded


def test_status_uses_only_the_datas_words():
    assert "approved" not in text(build("status", [CL2011], G, S, TODAY)) and "closed" in text(build("status", [CL2011], G, S, TODAY))


def test_status_facts_have_the_status_kind():
    assert {f.kind for f in build("status", [CL2011], G, S, TODAY).facts} == {"status"}


def test_two_claims_are_each_labelled():
    t = text(build("status", [CL2048, CL2102], G, S, TODAY))
    assert "CL-2048" in t and "CL-2102" in t


def test_a_passed_deadline_says_so_and_offers_a_person():
    r = build("appeal_deadline", [CL2048], G, S, date(2026, 3, 20))
    assert "passed" in text(r) and r.offer_person and "left" not in text(r)


def test_next_steps_for_a_denied_claim_name_documents_and_deadline():
    t = text(build("next_steps", [CL2048], G, S, TODAY))
    assert "pathology report" in t and "2026-03-18" in t


def test_next_steps_for_an_open_claim_say_nothing_is_requested():
    assert "nothing is requested right now" in text(build("next_steps", [CL2102], G, S, TODAY))
    assert "nothing further" in text(build("next_steps", [CL2011], G, S, TODAY))


def test_no_amount_is_ever_stated_as_owed():
    t = text(build("amounts", [CL2011], G, S, TODAY)).lower()
    assert "you owe" not in t and "you will receive" not in t


def test_p1_file_format_fill_keeps_each_documents_requirements():
    t = text(fill(followup(G, "file_format_requirements"), CL2048, G, S, TODAY))
    assert "specimen details" in t and "visit date" in t


def test_opening_for_a_denial_question():
    assert opening("denial_question", CL2048, S) == ["status", "denial_reason", "documents_needed", "appeal_deadline"]


def test_opening_for_a_status_inquiry_on_a_denied_claim():
    assert opening("status_inquiry", CL2048, S) == ["status", "denial_reason", "documents_needed"]


def test_an_unknown_need_opens_with_status():
    assert opening("appeal_question", CL2011, S) == ["status"]


def test_opening_drops_answers_that_do_not_exist():
    assert opening("denial_question", CL2011, S) == ["status"]


def test_type_advice_is_found_with_the_same_normalising():
    assert "medical claims" in text(build("documents_needed", [replace(CL2048, case_type=" Healthcare ")], G, S, TODAY))


def test_opening_never_comes_back_empty_for_a_known_need():
    assert opening("document_submission", CL2102, S) == ["status"]


def test_a_deadline_one_day_away_says_tomorrow():
    assert "tomorrow" in text(build("appeal_deadline", [CL2048], G, S, date(2026, 3, 17)))
    assert "tomorrow" not in text(build("appeal_deadline", [CL2048], G, S, TODAY))
