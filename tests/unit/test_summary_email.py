"""The summary email's text: built by code from claim rows and the answers given, with no personal detail."""

from claims_agent.answers.summary_email import write_summary
from claims_agent.memory import RequestMemory
from test_answers import CL2011, CL2048, CL2102, G, S, TODAY  # the starter claims and guidance, built once


def test_w1_the_summary_covers_discussion_status_and_follow_ups():
    request = RequestMemory(need="denial_question", answered={
        "CL-2048": ["status", "denial_reason", "documents_needed", "appeal_deadline"]})
    subject, body = write_summary([CL2048], request, G, S, TODAY)
    assert subject
    assert body.index("What we discussed") < body.index("Status") < body.index("Follow-up")
    assert "CL-2048" in body and "denied" in body and "pathology report" in body and "2026-03-18" in body


def test_p15_two_claims_discussed_are_both_in_the_email():
    _, body = write_summary([CL2048, CL2102], RequestMemory(answered={"CL-2048": ["status"], "CL-2102": ["status"]}),
                            G, S, TODAY)
    assert "CL-2048" in body and "CL-2102" in body


def test_v11_the_email_holds_no_personal_details():
    _, body = write_summary([CL2048], RequestMemory(answered={"CL-2048": ["status"]}), G, S, TODAY)
    for secret in ("Margaret", "1985", "4472", "margaret@email.com", "6505212836", "POL-9921"):
        assert secret not in body


def test_a_typed_full_id_number_never_appears():
    from dataclasses import replace
    claim = replace(CL2048, denial_reason="Member 123456789 not found")
    _, body = write_summary([claim], RequestMemory(answered={"CL-2048": ["denial_reason"]}), G, S, TODAY)
    assert "123456789" not in body and "*****6789" in body


def test_the_full_list_is_a_fourth_section_one_line_per_claim():
    _, body = write_summary([], RequestMemory(), G, S, TODAY, full_list=[CL2048, CL2011])
    assert body.index("Follow-up") < body.index("The full list you asked for")
    assert any("CL-2048" in line and "healthcare" in line and "2026-01-12" in line and "denied" in line
               for line in body.splitlines())
    assert "CL-2011" in body


def test_no_full_list_section_without_a_list():
    _, body = write_summary([CL2048], RequestMemory(answered={"CL-2048": ["status"]}), G, S, TODAY)
    assert "The full list you asked for" not in body


def test_a_follow_up_topic_is_named_in_words_not_read_out():
    _, body = write_summary([CL2048], RequestMemory(answered={"CL-2048": ["missing_required_material_alternatives"]}),
                            G, S, TODAY)
    assert "missing required material alternatives" in body
