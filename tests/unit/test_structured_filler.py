import json
from dataclasses import replace
from typing import get_args

import pytest
from pydantic import BaseModel

from claims_agent.form import structured_filler
from claims_agent.form.filler import ScriptedFiller
from claims_agent.form.fields import Pick
from claims_agent.form.questions import build_questions
from claims_agent.form.structured_filler import StructuredFiller, answer_model, field_name
from claims_agent.form.switching import SwitchingFiller, build_switching_filler
from claims_agent.models.errors import ModelNotPermitted
from claims_agent.settings import load_settings
from test_model_clients import REQ, FakeOpenAI as _Fake, _Out


def answers(req=REQ, **chosen):
    m = answer_model(build_questions(req))
    values = {}
    for q in build_questions(req):
        values[field_name(q.name)] = chosen.get(q.name, "no" if q.type == "predicate" else q.choices[0][0])
    return m(**values)


class FakeOpenAI(_Fake):
    """The shared fake, but the Responses endpoint returns parsed answer models (one per script item)."""

    def __init__(self, parsed=None, script=None):
        super().__init__()
        items = script if script is not None else [parsed if parsed is not None else answers()]
        self.responses.script = [i if isinstance(i, BaseException) else _Out.model_construct(output_parsed=i, status="completed", output=[], output_text="")
                                 for i in items]


def test_one_field_per_question_each_with_unsure():
    m = answer_model(build_questions(REQ)); f = m.model_fields
    assert len(f) == len(build_questions(REQ)) and "unsure" in get_args(f["about"].annotation) and "unsure" in get_args(f["mood"].annotation)
    assert "unsure" in get_args(f["wants_person"].annotation)
    assert field_name("claim_type:healthcare") == "claim_type__healthcare"


async def test_yes_no_and_unsure_map_to_one_zero_and_none():
    form = await StructuredFiller(FakeOpenAI(answers(wants_person="yes", done="no", skips_step="unsure")), "text-model", retries=1, timeout=20).fill(REQ)
    assert (form.wants_person.p, form.done.p, form.skips_step.p) == (1.0, 0.0, None) and form.wants_person.by == "structured"


async def test_a_pick_maps_to_certain_and_unsure_to_no_value():
    form = await StructuredFiller(FakeOpenAI(answers(mood="angry", need="unsure")), "text-model", retries=1, timeout=20).fill(REQ)
    assert (form.mood.value, form.mood.p) == ("angry", 1.0) and form.need.value is None
    assert form.mood.by == "structured" and form.need.by == "structured"


async def test_claim_rows_still_cannot_be_sent_before_verification():
    with pytest.raises(ValueError):
        await StructuredFiller(FakeOpenAI(), "text-model", retries=1, timeout=20).fill(replace(REQ, verified=False, documents=("pathology report",)))


async def test_caller_text_is_one_data_record_and_not_in_the_developer_message():
    client = FakeOpenAI()
    await StructuredFiller(client, "text-model", retries=1, timeout=20).fill(REQ)
    developer, user = client.responses.calls[0]["input"]
    assert REQ.message not in developer["content"] and json.loads(user["content"])["caller_message"] == REQ.message


async def test_an_oversize_shape_is_sent_in_two_calls_and_merged(monkeypatch):
    monkeypatch.setattr(structured_filler, "MAX_PROPERTIES", len(build_questions(REQ)) - 1)
    client = FakeOpenAI(script=[None, None])
    qs = build_questions(REQ)
    half = len(qs) // 2
    client.responses.script = [_Out.model_construct(output_parsed=answers_for(qs[:half]), status="completed", output=[]),
                               _Out.model_construct(output_parsed=answers_for(qs[half:]), status="completed", output=[])]
    form = await StructuredFiller(client, "text-model", retries=0, timeout=20).fill(REQ)
    assert len(client.responses.calls) == 2 and form.english.p == 0.0


def answers_for(qs):
    m = answer_model(qs)
    return m(**{field_name(q.name): "no" if q.type == "predicate" else q.choices[0][0] for q in qs})


async def test_a_malformed_response_is_a_model_failure_and_a_bug_of_ours_is_not_retried():
    from claims_agent.models.errors import ModelUnavailable
    bad = FakeOpenAI(script=[None, None])
    bad.responses.script = [_Out(output_parsed=None)] * 2
    with pytest.raises(ModelUnavailable):
        await StructuredFiller(bad, "text-model", retries=1, timeout=20).fill(REQ)
    client = FakeOpenAI(script=[answers()])
    import claims_agent.form.structured_filler as sf
    orig = sf.form_from_answers
    sf.form_from_answers = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("our bug"))
    try:
        with pytest.raises(RuntimeError):
            await StructuredFiller(client, "text-model", retries=1, timeout=20).fill(REQ)
    finally:
        sf.form_from_answers = orig
    assert len(client.responses.calls) == 1


async def test_fill_one_maps_a_choice_and_unsure():
    q = build_questions(REQ)[5]  # mood
    one = answer_model([q])
    f = StructuredFiller(FakeOpenAI(script=[one(mood="upset")]), "text-model", retries=0, timeout=20)
    assert (await f.fill_one(REQ, q)).value == "upset"
    f = StructuredFiller(FakeOpenAI(script=[one(mood="unsure")]), "text-model", retries=0, timeout=20)
    assert (await f.fill_one(REQ, q)).value is None


async def test_scripted_fill_one_defaults_to_no_value_and_records_the_question():
    s = ScriptedFiller()
    q = build_questions(REQ)[0]
    assert (await s.fill_one(REQ, q)).value is None and s.one_questions == [q]
    s.queue_one(Pick(value="x", p=1.0))
    assert (await s.fill_one(REQ, q)).value == "x"


async def test_switching_filler_stays_on_decisions_when_asked_and_switches_for_good_when_refused(monkeypatch):
    p, f = ScriptedFiller(name="decisions"), ScriptedFiller(name="structured")
    sw = SwitchingFiller(p, f)
    p.fail_with(ModelNotPermitted("m")); f.queue("form-a"); f.queue("form-b")
    assert await sw.fill(REQ) == "form-a" and sw.name == "structured"
    assert await sw.fill(REQ) == "form-b" and len(p.requests) == 1


def test_the_builder_follows_form_filler(monkeypatch):
    s = load_settings()
    for mode, expected in (("decisions", "decisions"), ("structured", "structured"), ("auto", "decisions")):
        monkeypatch.setenv("FORM_FILLER", mode)
        assert build_switching_filler(s, object()).name == expected
    monkeypatch.setenv("FORM_FILLER", "bogus")
    with pytest.raises(ValueError):
        build_switching_filler(s, object())
