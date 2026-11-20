import pytest
from helpers import BRIEF_MSG, BRIEF_SAID, FORM, make_form, say, verified_party

from claims_agent.models.errors import ModelNotPermitted, ModelUnavailable
from claims_agent.states import State

pytestmark = pytest.mark.planted


async def test_a_refused_decisions_call_switches_to_the_second_filler_and_the_conversation_completes(deps_with_switch, conv):
    deps_with_switch.filler.primary.fail_with(ModelNotPermitted("gpt-6-luna"))
    r = await say(deps_with_switch, conv, BRIEF_MSG, FORM, BRIEF_SAID)
    assert await verified_party(deps_with_switch, conv) == "P9" and r.inspector["filler"] == "structured"
    await say(deps_with_switch, conv, "thanks", make_form(done=True))
    assert len(deps_with_switch.filler.primary.requests) == 1


async def test_a_timeout_does_not_switch_fillers(deps_with_switch, conv):
    deps_with_switch.filler.primary.fail_with(ModelUnavailable("gpt-6-luna"))
    assert (await say(deps_with_switch, conv, "hello there")).state == State.ESCALATED and deps_with_switch.filler.name == "decisions"
