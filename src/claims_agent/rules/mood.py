"""How the caller sounds, and when an upset caller is offered a person instead of being asked again.

The form's mood question has one calm option and three that are not. A real form often splits an upset message
between two of them ("upset" 0.51, "angry" 0.49), so no single pick is confident although the form is sure the caller
is not calm. A confident pick is taken as it is; otherwise, when the options that are not calm reach the pick
threshold together, the likeliest of them is taken. Reading an upset caller as calm costs the acknowledgement they
needed; reading a calm one as upset costs one kind sentence.
"""

from collections.abc import Sequence

from claims_agent.form.fields import Pick
from claims_agent.form.read import picked
from claims_agent.settings import Thresholds

NOT_CALM = ("uneasy", "upset", "angry")
FRUSTRATED = ("upset", "angry")


def caller_mood(f: Pick, t: Thresholds) -> str | None:
    """The caller's mood: the confident pick, else the likeliest mood that is not calm when those together are
    confident, else None (not known)."""
    mood = picked(f, t)
    if mood is not None:
        return mood
    wary = {k: p for k, p in f.options.items() if k in NOT_CALM}
    if wary and sum(wary.values()) >= t.pick_confident:
        return max(wary, key=lambda k: wary[k])
    return None


def frustrated_again(moods: Sequence[str], progressed: bool) -> bool:
    """The caller sounded upset or angry on this turn and on the one before, and this turn moved nothing forward:
    asking again is not working, so a person is offered as the other way on. The caller may still carry on here."""
    return not progressed and len(moods) >= 2 and moods[-1] in FRUSTRATED and moods[-2] in FRUSTRATED
