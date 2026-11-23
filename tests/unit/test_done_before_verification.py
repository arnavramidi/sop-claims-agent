"""An unverified caller who says they are done: closed at once when calm; offered a person once first when upset or
angry; closed when done again after that offer. A verified caller who is done goes to the wrap-up."""

from helpers import make_form

from claims_agent.form.fields import Pick
from claims_agent.rules.every_message import PERSON_OFFER, global_outcome
from claims_agent.settings import load_thresholds

T = load_thresholds()
# How a real form reads an upset message: sure it is not calm, split between upset and angry.
SPLIT = Pick(value="upset", p=0.35, options={"calm": 0.0, "uneasy": 0.0, "upset": 0.51, "angry": 0.49})


def test_an_upset_caller_is_offered_a_person_before_closing():
    out = global_outcome(make_form(done=True, mood=SPLIT), verified=False, pending=None, t=T)
    assert out.terminal is None and [p.instruction for p in out.parts] == ["offer_person_before_closing"]
    assert out.parts[0].pending == PERSON_OFFER and out.parts[0].plain


def test_done_again_after_the_offer_closes():
    out = global_outcome(make_form(done=True, mood=SPLIT), verified=False, pending=PERSON_OFFER, t=T)
    assert out.terminal == "close"


def test_a_calm_or_uneasy_caller_is_closed_at_once():
    for mood in ("calm", "uneasy"):
        assert global_outcome(make_form(done=True, mood=mood), verified=False, pending=None, t=T).terminal == "close"


def test_a_verified_caller_who_is_done_goes_to_the_wrap_up_whatever_the_mood():
    assert global_outcome(make_form(done=True, mood=SPLIT), verified=True, pending=None, t=T).terminal == "wrap_up"
