import pytest

from claims_agent.form.fields import Pick
from claims_agent.form.filler import ScriptedExtractor, ScriptedFiller
from claims_agent.form.questions import FormRequest, build_questions, form_from_answers
from claims_agent.form.read import PickAnswer, Refused, YesNoAnswer, confident_yes, picked, top_two, unsure, yes
from claims_agent.settings import load_thresholds
from claims_agent.states import State
from helpers import make_form, stated


def req(**kw) -> FormRequest:
    base = dict(message="hello", state=State.VERIFY_ID, verified=False,
                recent_caller_messages=tuple(f"earlier message {i}" for i in range(6)),
                claim_types=("healthcare", "dental", "auto"),
                claim_statuses=("denied", "closed", "open"),
                needs=("denial_question", "document_submission", "next_steps", "status_inquiry",
                       "general_claim_question"),
                relationships=("son", "daughter", "spouse"))
    base.update(kw)
    return FormRequest(**base)


def question(qs, name):
    return next(q for q in qs if q.name == name)


def test_one_yes_no_per_type_status_month_and_year():
    names = {q.name for q in build_questions(req(years=(2022, 2023, 2024, 2025, 2026)))}
    assert {"claim_type:healthcare", "claim_type:dental", "claim_type:auto"} <= names
    assert {"claim_status:denied", "claim_status:closed", "claim_status:open"} <= names
    assert {f"month:{m}" for m in range(1, 13)} <= names and {"year:2022", "year:2026", "no_year_said"} <= names
    assert {f"shaped_need:{i}" for i in range(6)} <= names


def test_claim_rows_cannot_be_put_in_a_request_before_verification():
    with pytest.raises(ValueError):
        build_questions(req(verified=False, listed_claims=(("CL-2048", "healthcare"),)))
    with pytest.raises(ValueError):
        build_questions(req(verified=False, documents=("pathology report",)))


def test_d6_a_new_claim_type_becomes_a_question():
    assert "claim_type:travel" in {q.name for q in build_questions(req(claim_types=("travel",)))}


def test_identity_questions_stop_once_verified():
    names = {q.name for q in build_questions(req(verified=True, state=State.PROCESS_CASE))}
    assert not {"caller_role", "gave_identity_detail", "correcting_detail", "partial_detail"} & names


def test_consent_recheck_only_while_awaiting_consent():
    assert "consent_recheck" in {q.name for q in build_questions(req(state=State.AWAIT_CONSENT))}
    assert "consent_recheck" not in {q.name for q in build_questions(req(state=State.VERIFY_ID))}


def test_email_offer_reply_only_in_post_process():
    assert "email_offer_reply" in {q.name for q in build_questions(req(verified=True, state=State.POST_PROCESS))}
    assert "email_offer_reply" not in {q.name for q in build_questions(req(verified=True, state=State.PROCESS_CASE))}


def test_example_phrases_ride_in_the_option_description():
    q = question(build_questions(req(answers=(("submission_timing", ""),),
                                     examples={"answer_wanted:submission_timing": ("when do i need to send",)})),
                 "answer_wanted")
    assert "when do i need to send" in dict(q.choices)["submission_timing"]


def test_a_refused_answer_reads_as_unsure():
    form = form_from_answers(req(), {"wants_person": Refused()})
    t = load_thresholds()
    assert form.wants_person.p is None and unsure(form.wants_person, t) and not yes(form.wants_person, t)


def test_the_whole_form_fits_one_request():
    assert len(build_questions(req(verified=True, state=State.POST_PROCESS))) <= 400


def test_picked_needs_the_confidence_threshold():
    t = load_thresholds()
    assert picked(Pick(value="status", p=0.59), t) is None and picked(Pick(value="status", p=0.6), t) == "status"


def test_a_pick_outside_the_offered_options_is_stored_as_none():
    form = form_from_answers(req(), {"mood": PickAnswer(choice="furious", probabilities={}, confidence=0.9)})
    assert form.mood.value is None


def test_answers_fill_the_form_and_unasked_fields_are_none():
    form = form_from_answers(req(), {
        "wants_person": YesNoAnswer(probability=0.9),
        "claim_type:dental": YesNoAnswer(probability=0.8),
        "mood": PickAnswer(choice="upset", probabilities={"upset": 0.7, "angry": 0.2, "calm": 0.1}, confidence=0.7),
    })
    assert form.wants_person.p == 0.9 and form.claim_types["dental"].p == 0.8
    assert form.mood.value == "upset" and top_two(form.mood) == ["upset", "angry"]
    assert form.email_offer_reply is None and form.consent_recheck is None


async def test_scripted_filler_and_extractor_replay_and_record():
    filler, extractor = ScriptedFiller(), ScriptedExtractor()
    filler.queue(make_form(wants_person=True))
    extractor.queue(stated(policy_number="P-1"))
    r = req()
    assert (await filler.fill(r)).wants_person.p == 1.0 and filler.requests == [r]
    assert (await extractor.extract("x", want_identity=True, want_claim_number=False)).policy_number == "P-1"
    filler.fail_with(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await filler.fill(r)


def test_make_form_defaults_are_neutral():
    f = make_form(mood="angry", claim_types={"healthcare": 0.9})
    assert f.english.p == 1.0 and f.wants_person.p == 0.0 and f.mood.value == "angry"
    assert f.claim_types["healthcare"].p == 0.9


def test_shaped_need_is_one_question_per_recent_message():
    qs = build_questions(req(recent_caller_messages=("my claim was denied", "what do I send")))
    shaped = [q for q in qs if q.name.startswith("shaped_need:")]
    assert [q.name for q in shaped] == ["shaped_need:0", "shaped_need:1"]
    assert "number 0" in shaped[0].instructions and "number 1" in shaped[1].instructions
    assert all("my claim was denied" not in q.instructions and "what do I send" not in q.instructions for q in shaped)
    form = form_from_answers(req(recent_caller_messages=("a", "b")), {"shaped_need:1": YesNoAnswer(probability=0.8)})
    assert [y.p for y in form.shaped_need] == [None, 0.8]


def test_only_the_last_six_recent_messages_are_used():
    from claims_agent.form.questions import RECENT_WINDOW
    assert RECENT_WINDOW == 6
    qs = build_questions(req(recent_caller_messages=tuple(f"m{i}" for i in range(1, 8))))
    shaped = [q for q in qs if q.name.startswith("shaped_need:")]
    assert [q.name for q in shaped] == [f"shaped_need:{i}" for i in range(6)]
    assert "number 0" in shaped[0].instructions and "number 5" in shaped[5].instructions


async def test_the_record_holds_the_last_six_of_seven_messages_numbered_from_zero():
    import json
    from claims_agent.form.decisions_filler import DecisionsFiller

    class Client:
        class decisions:
            calls = []

            @staticmethod
            async def create(**kw):
                Client.decisions.calls.append(kw)
                raise RuntimeError("stop")

    r = req(recent_caller_messages=tuple(f"m{i}" for i in range(1, 8)))
    with pytest.raises(RuntimeError):
        await DecisionsFiller(Client, "m", retries=0, timeout=5).fill(r)
    record = json.loads(Client.decisions.calls[0]["input"][1]["content"])
    assert record["earlier_caller_messages"] == [{"number": i, "text": f"m{i + 2}"} for i in range(6)]
    assert [q["name"] for q in Client.decisions.calls[0]["questions"] if q["name"].startswith("shaped_need:")] == \
        [f"shaped_need:{i}" for i in range(6)]


def test_a_request_with_claim_rows_cannot_be_built_before_verification():
    with pytest.raises(ValueError):
        FormRequest(message="x", verified=False, listed_claims=(("CL-1", "dental"),))
    with pytest.raises(ValueError):
        FormRequest(message="x", verified=False, documents=("pathology report",))
    assert FormRequest(message="x", verified=True, documents=("pathology report",)).documents


async def test_the_scripted_filler_can_never_be_handed_such_a_request():
    filler = ScriptedFiller()
    filler.queue(make_form())
    with pytest.raises(ValueError):
        await filler.fill(FormRequest(message="x", verified=False, documents=("d",)))
    assert filler.requests == []


def test_confident_yes_is_at_or_above_the_top_of_the_unsure_band():
    from claims_agent.form.fields import YesNo
    t = load_thresholds()
    assert confident_yes(YesNo(p=t.unsure_high), t)
    assert not confident_yes(YesNo(p=0.55), t)
    assert not confident_yes(YesNo(p=None), t)


# ---- Task 13 fix round 1: a listed claim is offered with what the caller was shown ----------------------------

from claims_agent.form.fields import ListedClaim  # noqa: E402

SHOWN = (ListedClaim("CL-2048", "healthcare", "denied", "2026-01-12"),
         ListedClaim("CL-2011", "healthcare", "closed", "2025-01-28"))


def test_each_listed_claim_option_carries_its_date_and_status_in_the_order_shown():
    q = next(q for q in build_questions(req(verified=True, listed_claims=SHOWN)) if q.name == "listed_claim")
    assert [v for v, _ in q.choices] == ["CL-2048", "CL-2011", "none"]
    first, second = q.choices[0][1], q.choices[1][1]
    assert "2026-01-12" in first and "denied" in first and "number 1" in first
    assert "2025-01-28" in second and "closed" in second and "number 2" in second


def test_an_undated_listed_claim_says_so():
    q = next(q for q in build_questions(req(verified=True, listed_claims=(ListedClaim("CL-1", "auto", "open", None),)))
             if q.name == "listed_claim")
    assert "date not on file" in q.choices[0][1]


def test_listed_claims_are_still_refused_before_verification():
    with pytest.raises(ValueError):
        FormRequest(message="x", verified=False, listed_claims=SHOWN)


def test_the_which_document_question_can_say_a_document_was_named_that_is_not_on_the_list():
    qs = {q.name: q for q in build_questions(req(verified=True, state=State.PROCESS_CASE,
                                                 documents=("pathology report", "office note")))}
    values = [v for v, _ in qs["which_document"].choices]
    assert values == ["pathology report", "office note", "all", "not_listed", "none_stated"]
    described = dict(qs["which_document"].choices)
    assert described["not_listed"] != described["none_stated"] and "not" in described["not_listed"]


def test_a_representative_is_asked_whether_they_moved_to_another_person():
    from claims_agent.form.questions import FormRequest, build_questions
    names = lambda req: {q.name for q in build_questions(req)}  # noqa: E731
    assert "other_than_represented" in names(FormRequest(message="hi", acting_for_policyholder=True))
    assert "other_than_represented" not in names(FormRequest(message="hi"))


def test_each_figure_question_points_at_its_meaning_in_the_data_record():
    from claims_agent.form.questions import FormRequest, build_questions
    req = FormRequest(message="hi", verified=True, figure_meanings={"net_pay": "The finalized amount the insurer paid."})
    q = next(q for q in build_questions(req) if q.name == "figure:net_pay")
    assert "money_figures" in q.instructions and "The finalized amount the insurer paid." not in q.instructions


def test_a_list_of_claims_is_worded_with_every_detail_that_tells_them_apart():
    from claims_agent.reply.instructions import INSTRUCTIONS
    assert "every detail" in INSTRUCTIONS["which_claim"] and "every detail" in INSTRUCTIONS["nearest_matches"]
