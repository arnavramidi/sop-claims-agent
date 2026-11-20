from datetime import date

from claims_agent.settings import load_settings, load_thresholds


def test_limits_match_the_spec():
    s = load_settings()
    assert (s.limits.stalled_turns, s.limits.wrong_values, s.limits.consent_checks) == (3, 3, 5)
    assert (s.limits.claims_in_focus, s.limits.claims_prepared, s.limits.unclear_email_replies) == (5, 3, 2)
    assert (s.limits.model_retries, s.limits.model_timeout_seconds) == (1, 20)
    assert s.limits.message_chars == 4000 and s.limits.documents_offered == 25


def test_today_and_data_folder_defaults():
    s = load_settings()
    assert s.today.date == date(2026, 3, 5) and s.data.folder == "data/starter" and s.data.reset_on_start is True


def test_freedom_levels_are_three_switches():
    s = load_settings()
    assert {k: v.freedom for k, v in s.states.items()} == {
        "VERIFY_ID": "strict", "AWAIT_CONSENT": "strict", "RESOLVE_INTENT": "free",
        "PROCESS_CASE": "free", "POST_PROCESS": "guided"}
    f = s.freedom
    assert (f["strict"].claim_facts, f["strict"].settle_unsure, f["strict"].prepared_replies) == (False, False, False)
    assert (f["free"].claim_facts, f["free"].settle_unsure, f["free"].prepared_replies) == (True, True, True)
    assert (f["guided"].claim_facts, f["guided"].settle_unsure, f["guided"].prepared_replies) == (True, False, False)


def test_needs_open_with_the_spec_answers():
    n = load_settings().needs
    assert n["denial_question"].opens_with == ["status", "denial_reason", "documents_needed", "appeal_deadline"]
    assert n["document_submission"].opens_with == ["documents_needed", "appeal_deadline"]
    assert n["next_steps"].opens_with == ["next_steps"]
    assert n["status_inquiry"].opens_with == ["status"] and n["status_inquiry"].opens_with_when_denied == ["status", "denial_reason", "documents_needed"]
    assert n["general_claim_question"].opens_with == ["status"]


def test_followup_special_behaviour_is_settings_not_code():
    sp = load_settings().followups.special
    assert sp["submission_timing"].with_deadline and sp["submission_timing"].drop_when_deadline_passed
    assert sp["file_format_requirements"].documents_as_requirements
    assert sp["missing_required_material_alternatives"].per_document_alternatives


def test_thresholds_are_placeholders_until_calibrated():
    t = load_thresholds()
    assert (t.yes, t.unsure_low, t.unsure_high, t.pick_confident) == (0.5, 0.35, 0.65, 0.6)
    assert (t.send_email_min, t.prepared_reply_min) == (0.8, 0.8)


def test_model_ids_are_pinned():
    m = load_settings().models
    assert (m.fast, m.text, m.larger) == ("gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra")


def test_reply_check_settings():
    rc = load_settings().reply_check
    assert rc.max_length == 1500
    assert rc.status_forms["approved"] == ["approved", "approve", "approves", "approval"]
    assert all(w in rc.status_forms for w in rc.wary_status_words)


def test_the_wording_and_extractor_limits_are_settings():
    m = load_settings().models
    assert m.wording_max_output_tokens > 0 and m.extractor_max_output_tokens > 0
    assert m.wording_reasoning_effort and m.wording_verbosity in ("low", "medium", "high") and m.extractor_reasoning_effort
