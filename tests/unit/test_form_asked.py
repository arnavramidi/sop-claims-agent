"""The form as the fast model was asked it, as the turn record keeps it for the inspector."""

import json

import pytest

from claims_agent.form import asked
from claims_agent.form.asked import OTHER, Reading, answer_of, form_asked, topic
from claims_agent.form.fields import ListedClaim, Pick, YesNo
from claims_agent.form.questions import FormRequest, Question, build_questions, form_from_answers
from claims_agent.form.read import PickAnswer, YesNoAnswer
from claims_agent.loader.read import MONEY_FIELDS
from claims_agent.settings import load_thresholds
from claims_agent.states import State
from helpers import make_form

T = load_thresholds()

BEFORE = FormRequest(message="hello", state=State.AWAIT_CONSENT, claim_types=("healthcare", "dental"),
                     claim_statuses=("denied",), years=(2025, 2026), needs=("denial_question",),
                     relationships=("son",), answers=(("status", "the claim's status"),),
                     recent_caller_messages=("one", "two"), acting_for_policyholder=True)
AFTER = FormRequest(message="hello", state=State.POST_PROCESS, verified=True, claim_types=("healthcare",),
                    claim_statuses=("denied",), years=(2026,), needs=("denial_question",),
                    answers=(("status", "the claim's status"),),
                    listed_claims=(ListedClaim("CL-2048", "healthcare", "denied", "2026-01-12"),),
                    documents=("pathology report",), figure_meanings={m: "what it means" for m in MONEY_FIELDS})


def _every_answer(req: FormRequest):
    """A form filled with an answer to every question, as a filler fills it."""
    answers = {q.name: PickAnswer(choice=q.choices[0][0], confidence=0.9,
                                  probabilities={q.choices[0][0]: 0.9})
               if q.type == "choice" else YesNoAnswer(probability=0.9) for q in build_questions(req)}
    return form_from_answers(req, answers)


@pytest.mark.parametrize("req", [BEFORE, AFTER], ids=["before_verification", "after_verification"])
def test_every_question_the_form_can_ask_finds_its_answer_and_a_topic(req):
    form = _every_answer(req)
    for q in build_questions(req):
        assert answer_of(form, q.name) is not None, q.name
        assert topic(q.name) != OTHER, q.name


def test_each_question_is_listed_in_the_words_it_was_sent_with_its_options():
    rec = form_asked([Reading(AFTER, _every_answer(AFTER))], T, cap=10 ** 6)
    sent = build_questions(AFTER)
    shown = rec["readings"][0]["questions"]
    assert [q["id"] for q in shown] == [q.name for q in sent]
    assert [q["text"] for q in shown] == [q.instructions for q in sent]
    listed = next(q for q in shown if q["id"] == "listed_claim")
    assert listed["type"] == "pick" and listed["options"][0]["value"] == "CL-2048"
    assert rec["readings"][0]["count"] == len(sent) and rec["readings"][0]["verified"] is True


def test_marks_follow_the_thresholds():
    form = make_form(wants_person=0.5, done=0.9, declines=YesNo(p=None), need="denial_question",
                     answer_wanted=Pick(value="status", p=0.55, options={"status": 0.55, "deadline": 0.4}),
                     how_many="none")
    rec = form_asked([Reading(AFTER, form)], T, cap=10 ** 6)
    by_id = {q["id"]: q for q in rec["readings"][0]["questions"]}
    assert by_id["wants_person"]["mark"] == "unsure" and by_id["wants_person"]["reading"] == "yes"
    assert by_id["done"]["mark"] == "confident" and by_id["done"]["answered"] is True
    assert by_id["declines"]["mark"] == "no_answer" and by_id["declines"]["answered"] is False
    assert by_id["english"]["answered"] is True and by_id["skips_step"]["answered"] is False  # a confident no
    assert by_id["need"]["mark"] == "confident" and by_id["need"]["answered"] is True
    assert by_id["answer_wanted"]["mark"] == "below" and by_id["answer_wanted"]["answer"]["options"]["deadline"] == 0.4
    assert by_id["how_many"]["mark"] == "confident" and by_id["how_many"]["answered"] is False  # "none" says nothing
    assert rec["thresholds"] == {"yes": T.yes, "unsure_low": T.unsure_low, "unsure_high": T.unsure_high,
                                 "pick_confident": T.pick_confident}


def test_a_question_added_later_is_listed_under_other(monkeypatch):
    real = asked.build_questions
    monkeypatch.setattr(asked, "build_questions",
                        lambda req: [*real(req), Question("brand_new", "predicate", "A question added later?")])
    rec = form_asked([Reading(BEFORE, make_form())], T, cap=10 ** 6)
    new = rec["readings"][0]["questions"][-1]
    assert new["id"] == "brand_new" and new["topic"] == OTHER and new["mark"] == "no_answer"


def test_what_the_model_could_see_changes_with_verification():
    before = form_asked([Reading(BEFORE, make_form())], T, cap=10 ** 6)["readings"][0]
    after = form_asked([Reading(AFTER, make_form())], T, cap=10 ** 6)["readings"][0]
    assert any("not verified" in line for line in before["what_it_saw"])
    assert any("is verified" in line for line in after["what_it_saw"])
    assert not any(q["id"] == "listed_claim" for q in before["questions"])
    assert any(q["id"].startswith("figure:") for q in after["questions"])


def test_a_part_carries_its_words_masked_and_a_settled_answer_says_how():
    part = FormRequest(message="my id is 123456789, mail me at a@b.example", state=State.PROCESS_CASE, verified=True)
    reading = Reading(part, make_form(), part=2, settled={"need": {"value": "status", "steps": ["larger"],
                                                                   "larger": None}})
    rec = form_asked([Reading(AFTER, make_form()), reading], T, cap=10 ** 6)
    whole, shown = rec["readings"]
    assert whole["part"] is None and whole["message"] is None
    assert shown["part"] == 2 and "123456789" not in shown["message"] and "*****6789" in shown["message"]
    assert "a@b.example" not in json.dumps(rec)
    need = next(q for q in shown["questions"] if q["id"] == "need")
    assert need["settled"] == {"value": "status", "steps": ["larger"], "larger": None}


def test_each_reading_is_cut_to_the_cap():
    rec = form_asked([Reading(AFTER, make_form())], T, cap=4000)
    reading = rec["readings"][0]
    assert len(json.dumps(reading)) <= 4000 and reading["truncated"] is True


def test_no_form_means_no_record():
    assert form_asked([], T, cap=10 ** 6) is None
