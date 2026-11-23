"""The progress rule: did this turn move things forward, and when to offer a person."""

from dataclasses import dataclass

from claims_agent.memory import Counters, Memory
from claims_agent.states import State


def made_progress(before: Memory, after: Memory, state_before: State, state_after: State,
                  grounded_answer: bool, *, pending_reply: str | None = None) -> list[str]:
    """The reasons this turn counts as progress; an empty list means stalled.

    `pending_reply` is the form's confident pick for the pending question ("yes", "no", ...); memory does not
    hold it, so the engine passes it. It counts only when a yes/no question was pending and it is yes or no.
    """
    reasons: list[str] = []
    held_b = {k: v.key for k, v in before.identity.held.items()}
    held_a = {k: v.key for k, v in after.identity.held.items()}
    if any(held_b.get(k) != v for k, v in held_a.items()):
        reasons.append("identity value added or changed")
    if any(_hint(before, f) != _hint(after, f)
           for f in ("claim_filter", "claim_number", "need", "wants_email")
           if getattr(after.hints, f) not in (None, False)):
        reasons.append("hint added or changed")
    if before.focus.claim_ids != after.focus.claim_ids:
        reasons.append("claims in focus changed")
    if before.focus.last_list != after.focus.last_list:
        reasons.append("claims listed changed")
    if after.request.need != before.request.need and after.request.need is not None \
            or after.focus.need != before.focus.need and after.focus.need is not None:
        reasons.append("need set or changed")
    if any(after.request.answered.get(k) != before.request.answered.get(k) and v
           for k, v in after.request.answered.items()):
        reasons.append("answer added")
    if before.pending is not None and before.pending.yes_no and pending_reply in ("yes", "no"):
        reasons.append("pending yes/no answered")
    if state_before != state_after:
        reasons.append("state moved")
    if grounded_answer:
        reasons.append("grounded answer given")
    return reasons


@dataclass(frozen=True)
class StallResult:
    stalled_in_a_row: int
    offer_person: bool


def apply_stall(counters: Counters, progressed: bool, state_before: State, state_after: State,
                limit: int) -> StallResult:
    if state_before == State.AWAIT_CONSENT and state_after == State.AWAIT_CONSENT:
        return StallResult(counters.stalled_in_a_row, False)
    if progressed:
        return StallResult(0, False)
    count = counters.stalled_in_a_row + 1
    if count >= limit:
        return StallResult(0, True)
    return StallResult(count, False)


def _hint(memory: Memory, name: str):
    """A hint as progress compares it: a claim description without the model's certainty about it, so describing the
    same claim again with other certainty is not progress."""
    value = getattr(memory.hints, name)
    if name == "claim_filter" and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "sureness"}
    return value


# A remembered hint (a need or a description said again, kept for later) moves nothing the caller can see: a turn
# whose only change is a hint asks them the same thing again. The one definition of a turn that moved, for the stall
# count and for the upset caller's count alike.
ONLY_REMEMBERED = {"hint added or changed"}


def moved(reasons: list[str]) -> bool:
    """Whether a turn moved: any reason `made_progress` gives other than a remembered hint."""
    return bool(set(reasons) - ONLY_REMEMBERED)
