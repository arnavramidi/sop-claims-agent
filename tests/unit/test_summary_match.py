from datetime import date

from claims_agent.db.claims import Claim
from claims_agent.rules.summary_match import found_claim_fact, summary_question, what_differs


def _claim(case_id, case_type, created, status, summary):
    return Claim(case_id, case_type, created, status, summary, None, (), None, None, None, None, None, None)


CL2048 = _claim("CL-2048", "healthcare", date(2026, 1, 12), "denied", "Healthcare claim denied")
CL2011 = _claim("CL-2011", "healthcare", date(2025, 1, 28), "closed", "Healthcare claim paid")
CL6001_ER = _claim("CL-6001", "healthcare", date(2025, 8, 9), "closed", "Emergency room visit for a broken tooth")
CL6002_CLEAN = _claim("CL-6002", "dental", date(2025, 8, 21), "closed", "Routine cleaning and x-rays")
CL6001_DENTAL = _claim("CL-6001", "dental", date(2025, 8, 14), "closed", "Dental claim completed")


def test_what_differs_between_the_two_january_claims():
    assert what_differs([CL2048, CL2011]) == ["status", "year"]


def test_what_differs_names_type_and_month_when_they_differ():
    assert what_differs([CL6001_ER, CL6002_CLEAN]) == ["case_type"]
    assert what_differs([CL6001_DENTAL, CL2048]) == ["case_type", "status", "year", "month"]


def test_the_summary_question_offers_each_claim_and_cannot_tell():
    q = summary_question([CL6001_ER, CL6002_CLEAN])
    assert dict(q.choices)["CL-6001"].startswith("Emergency room") and "cannot_tell" in dict(q.choices)


def test_the_summary_question_instructions_hold_no_claim_text():
    q = summary_question([CL6001_ER, CL6002_CLEAN])
    assert "Emergency" not in q.instructions and "CL-6001" not in q.instructions


def test_the_found_claim_fact_quotes_the_summary_and_adds_nothing():
    assert found_claim_fact(CL6001_DENTAL).text == \
        "Claim CL-6001: dental, opened 2025-08-14. Summary on file: \"Dental claim completed\"."
