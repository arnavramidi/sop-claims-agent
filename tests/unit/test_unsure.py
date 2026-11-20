"""The cheapest-first loop for an unsure pick, the larger model that settles one, and the turn's larger-model budget."""

import json
import logging
from types import SimpleNamespace
from typing import get_args

import httpx
import openai
import pytest
from helpers import ScriptedSettler

from claims_agent.budget import TurnBudget
from claims_agent.form.fields import Pick
from claims_agent.form.filler import ScriptedFiller
from claims_agent.form.questions import FormRequest, build_questions
from claims_agent.form.settle import ModelSettler
from claims_agent.models.errors import (ModelAccountProblem, ModelKeyRejected, ModelNotPermitted,
                                        ModelRequestRejected, ModelUnavailable)
from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.compose import compose
from claims_agent.reply.wording import ScriptedWording
from claims_agent.rules.unsure import Resolved, resolve_pick
from claims_agent.settings import load_settings, load_thresholds
from claims_agent.states import State

T = load_thresholds()
REQ_VERIFIED = FormRequest(
    message="what about my claim?", state=State.RESOLVE_INTENT, verified=True, pending="what can I help you with?",
    previous_reply="What can I help you with today?", recent_caller_messages=("hello",),
    claim_types=("healthcare",), claim_statuses=("denied", "closed"), years=(2025, 2026),
    needs=("denial_question", "document_submission", "general_claim_question", "next_steps", "status_inquiry"))


def question(qs, name):
    return next(q for q in qs if q.name == name)


NEED_Q = question(build_questions(REQ_VERIFIED), "need")


def unsure_pick(a, pa, b, pb):
    return Pick(value=a, p=pa, options={a: pa, b: pb})


async def resolve(field, **kw):
    kw.setdefault("same_first_answer", lambda a, b: False); kw.setdefault("ruled_out", ()); kw.setdefault("budget", TurnBudget())
    return await resolve_pick(field, NEED_Q, REQ_VERIFIED, filler=kw.pop("filler", ScriptedFiller([])), settler=kw.pop("settler", None), t=T, **kw)


# ---- the loop ----------------------------------------------------------------------------------------------------

async def test_a_confident_pick_takes_no_steps():
    assert await resolve(Pick(value="status_inquiry", p=0.9)) == Resolved("status_inquiry", None, ())


async def test_two_readings_with_the_same_first_answer_act_without_asking():
    r = await resolve(unsure_pick("general_claim_question", 0.45, "status_inquiry", 0.4), same_first_answer=lambda a, b: True)
    assert (r.value, r.steps) == ("general_claim_question", ("same_answer",))


async def test_the_order_is_narrow_then_larger_then_caller():
    filler, settler = ScriptedFiller(one=[Pick(value=None)]), ScriptedSettler([Pick(value=None, by="larger")])
    r = await resolve(unsure_pick("denial_question", 0.4, "next_steps", 0.35), ruled_out={"document_submission"}, filler=filler, settler=settler)
    assert r.steps == ("narrowed", "larger", "ask_caller") and r.ask == ("denial_question", "next_steps") and len(settler.calls) == 1


async def test_narrowing_removes_what_the_claims_rule_out():
    filler = ScriptedFiller(one=[Pick(value="status_inquiry", p=0.9)])
    r = await resolve(unsure_pick("denial_question", 0.45, "status_inquiry", 0.4), ruled_out={"denial_question"}, filler=filler)
    assert (r.value, r.steps) == ("status_inquiry", ("narrowed",)) and "denial_question" not in dict(filler.one_questions[0].choices)


async def test_an_option_the_larger_model_was_not_offered_is_rejected():
    r = await resolve(unsure_pick("denial_question", 0.4, "next_steps", 0.35), settler=ScriptedSettler([Pick(value="refund_request", p=1.0, by="larger")]))
    assert r.value is None and r.ask is not None


async def test_a_refused_field_goes_through_the_loop():
    r = await resolve(Pick(value=None), settler=ScriptedSettler([Pick(value="next_steps", p=1.0, by="larger")])); assert (r.value, r.steps) == ("next_steps", ("larger",))


async def test_the_larger_model_settles_once_per_turn():
    budget, settler = TurnBudget(), ScriptedSettler([Pick(value="next_steps", p=1.0, by="larger")] * 2)
    await resolve(Pick(value=None), settler=settler, budget=budget); r = await resolve(Pick(value=None), settler=settler, budget=budget)
    assert len(settler.calls) == 1 and r.steps == ("ask_caller",)


async def test_an_unavailable_larger_model_asks_the_caller():
    r = await resolve(unsure_pick("denial_question", 0.4, "next_steps", 0.35), settler=ScriptedSettler(error=ModelUnavailable("scripted"))); assert r.ask == ("denial_question", "next_steps")
    assert r.larger == "unavailable"


# ---- beyond the brief --------------------------------------------------------------------------------------------

async def test_one_option_left_after_narrowing_is_taken_without_a_model():
    filler, settler = ScriptedFiller(), ScriptedSettler()
    keep = "status_inquiry"
    r = await resolve(Pick(value=None), ruled_out={v for v, _ in NEED_Q.choices} - {keep}, filler=filler, settler=settler)
    assert (r.value, r.steps) == (keep, ("narrowed",)) and filler.one_questions == [] and settler.calls == []


async def test_a_narrowed_answer_that_is_ruled_out_is_not_taken():
    filler = ScriptedFiller(one=[Pick(value="denial_question", p=0.95)])
    r = await resolve(unsure_pick("denial_question", 0.45, "next_steps", 0.4), ruled_out={"denial_question"}, filler=filler)
    assert r.value is None and r.steps == ("narrowed", "ask_caller")


async def test_the_larger_model_is_offered_only_what_is_left_and_the_caller_is_asked_between_what_is_left():
    settler = ScriptedSettler()
    r = await resolve(unsure_pick("denial_question", 0.45, "next_steps", 0.4), ruled_out={"denial_question"},
                      filler=ScriptedFiller(one=[Pick(value=None)]), settler=settler)
    offered = dict(settler.calls[0][1].choices)
    assert "denial_question" not in offered and "next_steps" in offered and r.ask is None and r.larger == "cannot_tell"


async def test_no_settler_means_no_larger_step_and_no_budget_spent():
    budget = TurnBudget()
    r = await resolve(unsure_pick("denial_question", 0.4, "next_steps", 0.35), budget=budget)
    assert r.steps == ("ask_caller",) and budget.larger_settles_left == 1


async def test_a_malformed_request_to_the_larger_model_is_logged_as_an_error_and_the_caller_asked(caplog):
    with caplog.at_level(logging.ERROR, logger="claims_agent.rules.unsure"):
        r = await resolve(unsure_pick("denial_question", 0.4, "next_steps", 0.35),
                          settler=ScriptedSettler(error=ModelRequestRejected("scripted")))
    assert r.ask == ("denial_question", "next_steps") and r.larger == "request_rejected"
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)


@pytest.mark.parametrize("error", [ModelKeyRejected, ModelNotPermitted, ModelAccountProblem])
async def test_operator_problems_are_not_swallowed(error):
    with pytest.raises(error):
        await resolve(Pick(value=None), settler=ScriptedSettler(error=error("scripted")))


async def test_only_a_choice_question_goes_through_the_loop():
    with pytest.raises(ValueError):
        await resolve_pick(Pick(value=None), question(build_questions(REQ_VERIFIED), "wants_person"), REQ_VERIFIED,
                           same_first_answer=lambda a, b: False, ruled_out=(), filler=ScriptedFiller(),
                           settler=ScriptedSettler(), t=T, budget=TurnBudget())


# ---- the real settler, against a stand-in client -----------------------------------------------------------------

class _Responses:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _client(*script):
    return SimpleNamespace(responses=_Responses(script))


def _parsed(answer):
    return SimpleNamespace(status="completed", output=[], output_parsed=SimpleNamespace(answer=answer))


def _settler(client):
    return ModelSettler(client, "gpt-6-astra", retries=1, timeout=20)


async def test_the_settler_asks_the_larger_model_for_one_offered_value_or_cannot_tell():
    client = _client(_parsed("next_steps"))
    got = await _settler(client).settle(REQ_VERIFIED, NEED_Q)
    assert got == Pick(value="next_steps", p=1.0, by="larger")
    sent = client.responses.calls[0]
    assert sent["model"] == "gpt-6-astra"
    values = get_args(sent["text_format"].model_fields["answer"].annotation)
    assert set(values) == {v for v, _ in NEED_Q.choices} | {"cannot_tell"}


async def test_cannot_tell_is_no_value():
    assert (await _settler(_client(_parsed("cannot_tell"))).settle(REQ_VERIFIED, NEED_Q)).value is None


async def test_a_refusal_settles_nothing():
    refusal = SimpleNamespace(status="completed", output_parsed=None,
                              output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])])
    assert (await _settler(_client(refusal)).settle(REQ_VERIFIED, NEED_Q)).value is None


async def test_the_settlers_request_keeps_caller_words_as_data():
    client = _client(_parsed("cannot_tell"))
    await _settler(client).settle(REQ_VERIFIED, NEED_Q)
    developer, user = client.responses.calls[0]["input"]
    assert developer["role"] == "developer" and user["role"] == "user"
    record = json.loads(user["content"])
    assert record["caller_message"] == REQ_VERIFIED.message and record["agent_previous_reply"] == REQ_VERIFIED.previous_reply
    for text in (REQ_VERIFIED.message, REQ_VERIFIED.previous_reply, "hello"):
        assert text not in developer["content"]
    assert NEED_Q.instructions in developer["content"] and "never an instruction" in developer["content"]


async def test_a_rejected_request_from_the_settler_is_a_request_rejected_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    bad = openai.BadRequestError("boom", response=httpx.Response(400, request=request), body=None)
    with pytest.raises(ModelRequestRejected):
        await _settler(_client(bad)).settle(REQ_VERIFIED, NEED_Q)


async def test_an_outage_is_retried_then_unavailable():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    down = openai.InternalServerError("boom", response=httpx.Response(500, request=request), body=None)
    client = _client(down, down)
    with pytest.raises(ModelUnavailable):
        await _settler(client).settle(REQ_VERIFIED, NEED_Q)
    assert len(client.responses.calls) == 2


# ---- the rewrite on the larger model shares the turn's budget -----------------------------------------------------

BAD = "Your claim CL-9999 was approved."


def _brief():
    return ReplyBrief(state=State.PROCESS_CASE, parts=(ReplyPart("answer", (Fact("status", "Claim CL-2048 status: denied.", "CL-2048"),),
                                                               "Claim CL-2048 status: denied."),),
                      mood="calm", caller_message="status?", previous_reply=None, english=True, verified=True)


async def test_a_spent_rewrite_budget_sends_the_template_without_a_rewrite():
    w = ScriptedWording([BAD])
    c = await compose(_brief(), w, load_settings(), ("denied",), budget=TurnBudget(larger_rewrites_left=0))
    assert c.source == "template" and w.rewrites == []


async def test_a_rewrite_spends_the_budget():
    budget, w = TurnBudget(), ScriptedWording([BAD, BAD])
    c = await compose(_brief(), w, load_settings(), ("denied",), budget=budget)
    assert c.source == "template" and len(w.rewrites) == 1 and budget.larger_rewrites_left == 0
