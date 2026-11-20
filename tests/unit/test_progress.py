from helpers import with_held

from claims_agent.memory import Counters, Memory, Pending
from claims_agent.rules.progress import StallResult, apply_stall, made_progress
from claims_agent.states import State


def _step(c, progressed, a=State.VERIFY_ID, b=State.VERIFY_ID):
    r = apply_stall(c, progressed, a, b, 3)
    c.stalled_in_a_row = r.stalled_in_a_row
    return r


def test_b1_three_off_topic_turns_offer_a_person_on_the_third():
    c = Counters()
    assert [_step(c, False).offer_person for _ in range(3)] == [False, False, True]


def test_b2_alternating_useful_and_off_topic_never_reaches_the_limit():
    c = Counters()
    for progressed in [False, True] * 5:
        assert not _step(c, progressed).offer_person


def test_b14_the_offer_repeats_every_three_stalled_turns_and_never_forces():
    c = Counters()
    assert [_step(c, False, State.PROCESS_CASE, State.PROCESS_CASE).offer_person for _ in range(6)] == \
        [False, False, True, False, False, True]


def test_turns_spent_waiting_for_consent_are_not_counted():
    assert apply_stall(Counters(stalled_in_a_row=2), False, State.AWAIT_CONSENT, State.AWAIT_CONSENT, 3) == StallResult(2, False)


def test_leaving_consent_while_stalled_counts():
    assert apply_stall(Counters(stalled_in_a_row=2), False, State.AWAIT_CONSENT, State.RESOLVE_INTENT, 3) == StallResult(0, True)


def test_x9_an_identity_value_is_progress_even_in_an_off_topic_message():
    before, after = Memory.fresh(), with_held(Memory.fresh(), "dob", "1985-03-15")
    assert made_progress(before, after, State.VERIFY_ID, State.VERIFY_ID, False) != []


def test_b5_a_refusal_that_adds_nothing_is_stalled():
    assert made_progress(Memory.fresh(), Memory.fresh(), State.VERIFY_ID, State.VERIFY_ID, False) == []


def test_a_changed_held_value_hint_focus_need_and_answer_each_count():
    base = Memory.fresh()
    held = with_held(base, "dob", "1985-03-15")
    changed = with_held(held, "dob", "1990-01-01")
    assert made_progress(held, changed, State.VERIFY_ID, State.VERIFY_ID, False)
    for edit in (lambda m: setattr(m.hints, "need", "status"),
                 lambda m: setattr(m.focus, "claim_ids", ["CL-1"]),
                 lambda m: setattr(m.request, "need", "status"),
                 lambda m: setattr(m.focus, "need", "status"),
                 lambda m: m.request.answered.update({"CL-1": ["status"]})):
        after = base.model_copy(deep=True)
        edit(after)
        assert made_progress(base, after, State.PROCESS_CASE, State.PROCESS_CASE, False) != []


def test_state_move_and_grounded_answer_count():
    m = Memory.fresh()
    assert made_progress(m, m, State.VERIFY_ID, State.AWAIT_CONSENT, False)
    assert made_progress(m, m, State.PROCESS_CASE, State.PROCESS_CASE, True)


def test_an_answered_yes_no_counts_but_a_cleared_offer_does_not():
    before = Memory.fresh()
    before.pending = Pending(kind="offer_email", label="send?", yes_no=True)
    after = before.model_copy(deep=True)
    after.pending = None
    assert made_progress(before, after, State.POST_PROCESS, State.POST_PROCESS, False) == []
    for reply in ("yes", "no"):
        assert made_progress(before, after, State.POST_PROCESS, State.POST_PROCESS, False, pending_reply=reply)
    for reply in ("unclear", "not_answering", None):
        assert made_progress(before, after, State.POST_PROCESS, State.POST_PROCESS, False, pending_reply=reply) == []
    # nothing was pending, so a stray "yes" is not an answer
    assert made_progress(Memory.fresh(), Memory.fresh(), State.VERIFY_ID, State.VERIFY_ID, False, pending_reply="yes") == []


def test_f3_the_same_value_in_other_words_is_not_progress_but_a_different_value_is():
    from claims_agent.memory import HeldValue
    a = with_held(Memory.fresh(), "dob", "1985-03-15")
    b = a.model_copy(deep=True)
    b.identity.held["dob"] = HeldValue(as_stated="March 15 1985", key="1985-03-15", turn=2)
    assert made_progress(a, b, State.VERIFY_ID, State.VERIFY_ID, False) == []
    c = a.model_copy(deep=True)
    c.identity.held["dob"] = HeldValue(as_stated="1 Jan 1990", key="1990-01-01", turn=2)
    assert made_progress(a, c, State.VERIFY_ID, State.VERIFY_ID, False) != []


def test_the_same_claim_described_again_with_other_certainty_is_not_progress():
    before = Memory.fresh()
    before.hints.claim_filter = {"statuses": ["denied"], "sureness": {"statuses": 0.95}}
    again = before.model_copy(deep=True)
    again.hints.claim_filter = {"statuses": ["denied"], "sureness": {"statuses": 0.9}}
    assert made_progress(before, again, State.VERIFY_ID, State.VERIFY_ID, False) == []
    other = before.model_copy(deep=True)
    other.hints.claim_filter = {"statuses": ["denied"], "types": ["healthcare"],
                                "sureness": {"statuses": 0.95, "types": 0.9}}
    assert made_progress(before, other, State.VERIFY_ID, State.VERIFY_ID, False) == ["hint added or changed"]
