"""A turn's record lists the token counts of the model calls it made."""

import json
import types

from claims_agent.models.usage import Metered
from claims_agent.reply.wording import ModelWording
from claims_agent.settings import load_settings
from helpers import make_deps, make_form, new_conversation, say, turn_record


class _Responses:
    def __init__(self, usage):
        self.usage, self.calls = usage, 0

    async def create(self, **kw):
        self.calls += 1
        return types.SimpleNamespace(output_text=REPLY, status="completed", model=kw["model"],
                                     usage=self.usage)


# A reply that asks for identity details has to state the rule and list every option, or the check has it rewritten
# (a second model call); this wording passes, so the turn makes exactly one.
REPLY = ("Hello. To verify who you are, 3 of these details must match the policy: full name, date of birth, phone "
         "number, email address or the last four digits of your SSN or national ID.")


def _wording(usage):
    s = load_settings()
    client = Metered(types.SimpleNamespace(responses=_Responses(usage)))
    return ModelWording(client, s.models.text, s.models.larger, retries=0, timeout=5)


def _usage(i, o):
    return types.SimpleNamespace(input_tokens=i, output_tokens=o, total_tokens=i + o, input_tokens_details=None,
                                 output_tokens_details=types.SimpleNamespace(reasoning_tokens=1))


async def test_the_turn_record_carries_each_model_call_and_the_totals(load_dataset):
    deps = make_deps(await load_dataset("starter"), wording=_wording(_usage(40, 9)))
    conv = await new_conversation(deps)
    r = await say(deps, conv, "hello there", make_form())
    u = r.inspector["usage"]
    assert [c["job"] for c in u["calls"]] == ["word"]
    assert u["calls"][0]["input_tokens"] == 40 and u["totals"]["total_tokens"] == 49
    assert u["totals"]["reasoning_tokens"] == 1
    assert (await turn_record(deps, r.turn_id)) == json.loads(json.dumps(r.inspector))


async def test_a_turn_whose_calls_report_no_usage_has_empty_totals(load_dataset):
    deps = make_deps(await load_dataset("starter"), wording=_wording(None))
    conv = await new_conversation(deps)
    r = await say(deps, conv, "hello there", make_form())
    assert r.inspector["usage"]["calls"] == [] and r.inspector["usage"]["totals"]["total_tokens"] == 0


async def test_a_bookkeeping_failure_leaves_the_reply_as_it_was(load_dataset, monkeypatch):
    deps = make_deps(await load_dataset("starter"), wording=_wording(_usage(1, 1)))
    conv = await new_conversation(deps)
    plain = (await say(deps, conv, "hello there", make_form())).reply
    from claims_agent.models import usage as usage_mod
    monkeypatch.setattr(usage_mod.TurnUsage, "summary", lambda self: 1 / 0)
    deps = make_deps(await load_dataset("starter"), wording=_wording(_usage(1, 1)))
    conv = await new_conversation(deps)
    r = await say(deps, conv, "hello there", make_form())
    assert r.reply == plain and r.inspector["usage"] is None
