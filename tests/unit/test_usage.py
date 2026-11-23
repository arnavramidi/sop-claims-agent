"""Token counts of a turn's model calls: collected per call, totalled, never able to change a call or a reply."""

import types

import pytest

from claims_agent.http_view import for_http
from claims_agent.models import usage as usage_mod
from claims_agent.models.call import with_retry
from claims_agent.models.usage import Metered, collecting, turn_usage
from claims_agent.observability.turn_log import token_metrics, trace_event


def _usage(i=10, o=5, cached=2, reasoning=3):
    return types.SimpleNamespace(
        input_tokens=i, output_tokens=o, total_tokens=i + o,
        input_tokens_details=types.SimpleNamespace(cached_tokens=cached),
        output_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning))


class _Endpoint:
    def __init__(self, *responses):
        self.script = list(responses)

    async def create(self, **kw):
        return self.script.pop(0)

    async def parse(self, **kw):
        return self.script.pop(0)


class _Client:
    def __init__(self, *responses):
        self.responses = _Endpoint(*responses)
        self.decisions = _Endpoint(*responses)


def _ask(client, job, model="m-text"):
    async def call():
        return await client.responses.create(model=model)
    return with_retry(call, retries=0, timeout=5, model=model, job=job)


async def test_each_call_is_recorded_with_its_job_model_and_counts_and_totalled():
    client = Metered(_Client(types.SimpleNamespace(usage=_usage(10, 5, 2, 3)),
                             types.SimpleNamespace(usage=_usage(20, 1, 0, 0), model="m-larger")))
    with collecting():
        await _ask(client, "word")
        await _ask(client, "rewrite", model="m-big")
        got = turn_usage()
    for c in got["calls"]:
        c.pop("started"), c.pop("ended")
    assert got["calls"] == [
        {"job": "word", "model": "m-text", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
         "cached_tokens": 2, "reasoning_tokens": 3},
        {"job": "rewrite", "model": "m-larger", "input_tokens": 20, "output_tokens": 1, "total_tokens": 21,
         "cached_tokens": 0, "reasoning_tokens": 0}]
    assert got["totals"] == {"input_tokens": 30, "output_tokens": 6, "total_tokens": 36, "cached_tokens": 2,
                             "reasoning_tokens": 3}


async def test_a_call_made_without_a_job_is_still_counted_under_its_endpoint():
    client = Metered(_Client(types.SimpleNamespace(usage=_usage())))
    with collecting():
        await client.responses.parse(model="m")
        assert turn_usage()["calls"][0]["job"] == "responses.parse"


async def test_a_response_without_usage_adds_nothing_and_is_returned_unchanged():
    reply = types.SimpleNamespace(usage=None)
    with collecting():
        assert await _ask(Metered(_Client(reply)), "word") is reply
        assert turn_usage() == {"calls": [], "totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                                                       "cached_tokens": 0, "reasoning_tokens": 0}}


async def test_a_call_outside_a_turn_is_not_recorded_and_still_works():
    reply = types.SimpleNamespace(usage=_usage())
    assert await _ask(Metered(_Client(reply)), "word") is reply
    assert turn_usage() is None


async def test_a_bookkeeping_error_never_changes_the_call(monkeypatch):
    def broken(response):
        raise RuntimeError("bookkeeping")
    monkeypatch.setattr(usage_mod, "read_usage", broken)
    reply = types.SimpleNamespace(usage=_usage())
    with collecting():
        assert await _ask(Metered(_Client(reply)), "word") is reply
        assert turn_usage()["calls"] == []


async def test_a_failed_call_with_no_response_contributes_nothing():
    class Boom:
        async def create(self, **kw):
            raise TimeoutError

    client = Metered(types.SimpleNamespace(responses=Boom()))
    with collecting():
        with pytest.raises(Exception):
            await _ask(client, "word")
        assert turn_usage()["calls"] == []


def test_a_calls_counts_become_the_metric_names_braintrust_recognises():
    call = {"job": "word", "model": "m", "input_tokens": 30, "output_tokens": 6, "total_tokens": 36,
            "cached_tokens": 2, "reasoning_tokens": 0}
    assert token_metrics(call) == {"prompt_tokens": 30, "completion_tokens": 6, "tokens": 36, "prompt_cached_tokens": 2}
    assert token_metrics({"input_tokens": "x"}) == {}


def test_the_turn_span_carries_no_token_metrics_so_a_dashboard_does_not_count_twice():
    record = {"usage": {"calls": [], "totals": {"total_tokens": 36}}}
    assert "metrics" not in trace_event(message="hi", reply="hello", record=record)
    assert "usage" in trace_event(message="hi", reply="hello", record=record)["metadata"]["record"]


async def test_each_call_carries_its_start_and_end_time():
    with collecting():
        await _ask(Metered(_Client(types.SimpleNamespace(usage=_usage()))), "word")
        call = turn_usage()["calls"][0]
    assert 0 < call["started"] <= call["ended"]


def test_hosted_responses_carry_no_panel_so_no_token_counts():
    body = {"reply": "hi", "inspector": {"usage": {"calls": [], "totals": {}}}}
    assert "inspector" not in for_http(body, hosted=True)
    assert for_http(body, hosted=False)["inspector"]["usage"] == {"calls": [], "totals": {}}
