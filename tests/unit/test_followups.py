"""Data-driven follow-ups: which exist, which comes next, how one is filled."""

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


from claims_agent.answers.followups import exists_for, fill, next_followup


def test_a_passed_deadline_says_so_and_offers_a_person():
    r = fill(followup(G, "submission_timing"), CL2048, G, S, date(2026, 3, 20))
    assert "passed" in text(r) and r.offer_person and "within a week" not in text(r)


def test_the_week_and_the_deadline_are_both_said():
    t = text(fill(followup(G, "submission_timing"), CL2048, G, S, TODAY))
    assert "within a week" in t and "2026-03-18" in t


def test_the_file_format_answer_includes_each_documents_own_requirements():
    t = text(fill(followup(G, "file_format_requirements"), CL2048, G, S, TODAY))
    assert "specimen details" in t and "visit date" in t


def test_documents_outstanding_but_not_denied_is_reworded_live_with_its_real_status():
    r = fill(followup(GM, "submission_method"), CL5001, GM, S, TODAY)
    assert r.reword_live and "open" in text(r) and "denied" not in text(r)


def test_a_denied_claim_is_not_reworded_live():
    assert not fill(followup(G, "submission_method"), CL2048, G, S, TODAY).reword_live


def test_alternatives_are_for_that_document_only():
    t = text(fill(followup(G, "missing_required_material_alternatives"), CL2048, G, S, TODAY, document="office note"))
    assert "visit summary" in t and "certified copy" not in t


def test_alternatives_without_a_document_use_the_default():
    t = text(fill(followup(G, "missing_required_material_alternatives"), CL2048, G, S, TODAY))
    assert "replacement copy" in t and "visit summary" not in t


def test_every_fact_names_its_claim():
    for topic in ("submission_method", "missing_required_material_alternatives", "fallback"):
        assert all("CL-2048" in f.text for f in fill(followup(G, topic), CL2048, G, S, TODAY).facts)


def test_a_template_is_filled_by_plain_replacement_only():
    f = followup(G, "submission_method")
    r = fill(replace(f, template="{case_id} {documents.__class__}"), CL2048, G, S, TODAY)
    assert "<class" not in text(r) and not r.grounded
    assert "the pathology report and the office note" in text(fill(replace(f, template="{documents}"), CL2048, G, S, TODAY))


def test_a_disabled_followup_is_never_filled():
    f = replace(followup(G, "submission_method"), enabled=False)
    r = fill(f, CL2048, G, S, TODAY)
    assert "member portal" not in text(r) and not r.grounded and r.offer_person


def test_the_first_remaining_followup_in_order():
    answered = {"missing_required_material_alternatives", "submission_timing", "processing_time_after_submission"}
    assert next_followup("document_submission", CL2048, answered, G, S).topic == "submission_method"


def test_nothing_is_offered_once_all_are_answered():
    assert next_followup("denial_question", CL2048, {"missing_required_material_alternatives", "submission_timing"}, G, S) is None


def test_settings_order_and_comes_after_are_honoured_and_unknown_names_ignored():
    s = S.model_copy(deep=True)
    s.followups.order = ["receipt_confirmation", "no_such_topic"]
    s.followups.comes_after = {"receipt_confirmation": "submission_method"}
    assert next_followup("document_submission", CL2048, set(), G, s).topic != "receipt_confirmation"
    assert next_followup("document_submission", CL2048, {"submission_method"}, G, s).topic == "receipt_confirmation"


def test_comes_after_as_a_list_and_a_cycle_does_not_dead_end():
    s = S.model_copy(deep=True)
    s.followups.comes_after = {"submission_timing": ["submission_method"], "submission_method": ["submission_timing"]}
    assert next_followup("document_submission", CL2048, set(), G, s) is not None


def test_a_followup_not_needing_documents_exists_on_a_clean_claim():
    ug = guidance_from(read_folder(UNMARKED))
    assert exists_for(followup(ug, "processing_time_after_submission"), CL2102, ug, S)


def test_a_followup_does_not_exist_for_a_claim_with_nothing_outstanding():
    assert not exists_for(followup(G, "submission_timing"), CL2102, G, S)


def test_a_text_naming_documents_does_not_exist_for_a_claim_without_any():
    f = replace(followup(G, "processing_time_after_submission"), requires_documents=False)
    assert not exists_for(f, CL2102, G, S)
    r = fill(f, CL2102, G, S, TODAY)
    assert not r.grounded and "not on file" in text(r)


def test_exists_and_fill_agree_for_every_followup_and_claim():
    for g in (G, GM):
        for c in (CL2048, CL2011, CL2102, CL5001):
            for f in g.followups:
                assert exists_for(f, c, g, S) == fill(f, c, g, S, TODAY).grounded or not f.enabled


def test_a_followup_without_denial_is_reworded_live_with_the_real_status():
    ug = guidance_from(read_folder(UNMARKED))
    r = fill(followup(ug, "processing_time_after_submission"), CL2102, ug, S, TODAY)
    t = text(r)
    assert r.reword_live and "open" in t and "  " not in t and "denial" not in t


def test_a_document_not_on_the_claim_uses_the_default_alternatives():
    f = followup(G, "missing_required_material_alternatives")
    t = text(fill(f, CL2048, G, S, TODAY, document="repair estimate"))
    assert "replacement copy" in t and "body shop" not in t
    t = text(fill(f, CL2048, G, S, TODAY, document="  Office  Note "))
    assert "visit summary" in t


def test_a_followup_naming_documents_carries_the_count():
    r = fill(followup(G, "submission_timing"), CL2048, G, S, TODAY)
    assert "Documents outstanding for claim CL-2048: 2." in text(r)
    assert "medical claims" not in text(r)


def test_a_settings_value_holding_a_blank_is_not_expanded_again():
    g = replace(G, followup_settings={**G.followup_settings, "average_processing_time_after_submission": "{case_id}"})
    assert "{case_id}" in text(fill(followup(g, "processing_time_after_submission"), CL2048, g, S, TODAY))


def test_a_malformed_template_is_not_filled():
    r = fill(replace(followup(G, "submission_method"), template="a { b"), CL2048, G, S, TODAY)
    assert not r.grounded


def _no_default(g):
    return replace(g, default_text="")


def test_the_count_is_the_claims_total_and_a_left_out_document_is_named():
    g = _no_default(G)
    claim = replace(CL2048, documents=("pathology report", "mystery form"))
    r = fill(followup(g, "file_format_requirements"), claim, g, S, TODAY)
    t = text(r)
    assert "Documents outstanding for claim CL-2048: 2." in t
    assert "No specific file requirements are on file for: mystery form." in t
    assert [f.kind for f in r.facts if "No specific" in f.text] == ["not_on_file"]


def test_when_every_document_is_named_there_is_no_left_out_fact():
    t = text(fill(followup(G, "file_format_requirements"), CL2048, G, S, TODAY))
    assert "No specific file requirements" not in t
    g = _no_default(G)
    assert "No specific file requirements" not in text(fill(followup(g, "file_format_requirements"), CL2048, g, S, TODAY))


def test_left_out_names_come_only_from_the_claims_own_documents():
    g = _no_default(G)
    claim = replace(CL2048, documents=("pathology report", "mystery form", "other form"))
    t = text(fill(followup(g, "file_format_requirements"), claim, g, S, TODAY))
    assert "for: mystery form, other form." in t and "repair estimate" not in t


def test_with_no_default_and_no_match_the_offer_and_the_answer_agree_for_every_followup():
    from claims_agent.answers.facts import available
    g = _no_default(G)
    claim = replace(CL2048, documents=("mystery form",))
    for f in g.followups:
        assert exists_for(f, claim, g, S) == fill(f, claim, g, S, TODAY).grounded
        assert (f.topic in available(claim, g, 1, S)) == exists_for(f, claim, g, S)
    assert "file_format_requirements" not in available(claim, g, 1, S)
