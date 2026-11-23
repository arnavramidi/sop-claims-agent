"""How the caller's mood is read from the form, what the wording model is told to do with it, and when an upset caller
is offered a person."""

import json
import types

from claims_agent.form.fields import Pick
from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.wording import ModelWording
from claims_agent.rules.mood import caller_mood
from claims_agent.settings import load_thresholds
from claims_agent.states import State

T = load_thresholds()


def test_a_confident_pick_is_the_mood():
    assert caller_mood(Pick(value="angry", p=0.9, options={"angry": 0.9, "upset": 0.1}), T) == "angry"
    assert caller_mood(Pick(value="calm", p=0.97, options={"calm": 0.97}), T) == "calm"


def test_a_message_split_between_upset_and_angry_reads_as_the_likelier_of_them():
    # what a real form returned for "I already told you who I am. This is ridiculous. ..."
    split = Pick(value="upset", p=0.35, options={"calm": 0.0, "uneasy": 0.0, "upset": 0.51, "angry": 0.49})
    assert caller_mood(split, T) == "upset"
    assert caller_mood(Pick(value="upset", p=0.4, options={"upset": 0.3, "angry": 0.7}), T) == "angry"


def test_an_unsure_reading_with_little_weight_off_calm_is_not_known():
    assert caller_mood(Pick(value="calm", p=0.4, options={"calm": 0.55, "uneasy": 0.25, "upset": 0.2}), T) is None
    assert caller_mood(Pick(), T) is None


class _Responses:
    def __init__(self):
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(output_text="ok", status="completed")


def _brief(mood: str) -> ReplyBrief:
    return ReplyBrief(state=State.VERIFY_ID, parts=(ReplyPart("ask_identity", (Fact("verification_rule", "Rule."),)),),
                      mood=mood, caller_message="This is ridiculous.", previous_reply=None, english=True,
                      verified=False)


async def test_the_wording_model_is_told_what_to_do_with_the_callers_mood():
    client = types.SimpleNamespace(responses=_Responses())
    await ModelWording(client, "text-model", "larger-model", retries=1, timeout=20).word(_brief("upset"))
    developer, user = client.responses.calls[0]["input"]
    assert json.loads(user["content"])["caller_mood"] == "upset"
    rules = developer["content"]
    assert "caller_mood" in rules and "acknowledges how the caller feels" in rules
    assert "promises nothing" in rules and "says nothing about any claim" in rules
