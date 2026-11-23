"""When a pick is unsure, the cheapest step first, and the caller asked only when nothing cheaper settles it.

1. A confident pick is taken at once.
2. When the two likeliest readings lead to the same first answer, the likeliest is acted on.
3. The readings the claims rule out are removed: one option left is taken; otherwise the fast model is asked the
   narrowed question once more, and a confident answer is taken.
4. The larger model is asked, at most once per turn (the turn's budget) and only in a state allowed to settle
   unsure picks. Its answer must be one of the options it was offered.
5. Otherwise the caller is asked between the two likeliest readings.

A refused or omitted field (no value) goes through the same steps. Only choice questions come here: an unsure yes or
no (a person wanted, the caller done) is never turned into a yes by this loop.

Model failures: the larger model being unavailable, or refusing our request as malformed, falls through to asking the
caller (the malformed request is logged as an error: it is our own bug). A rejected key, a model the key may not use
and an account with no credit are not caught here; the engine rolls the turn back and raises them.
"""

import logging
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from claims_agent.answers.facts import opening
from claims_agent.budget import TurnBudget
from claims_agent.form.fields import Pick
from claims_agent.form.filler import FormFiller
from claims_agent.form.questions import FormRequest, Question, build_questions
from claims_agent.form.read import picked
from claims_agent.form.settle import Settler
from claims_agent.models.errors import ModelRequestRejected, ModelUnavailable
from claims_agent.settings import Thresholds

if TYPE_CHECKING:
    from claims_agent.db.claims import Claim
    from claims_agent.engine import TurnContext

logger = logging.getLogger("claims_agent.rules.unsure")


@dataclass(frozen=True)
class Resolved:
    value: str | None
    ask: tuple[str, str] | None  # the two likeliest readings, when the caller must be asked
    steps: tuple[str, ...]  # steps tried, in order
    larger: str | None = None  # how the larger model's step ended when it settled nothing


def _ranked(pick: Pick, allowed: Collection[str]) -> list[str]:
    """The pick's readings, likeliest first, among the allowed values."""
    if pick.options:
        ranked = sorted(pick.options, key=lambda k: pick.options[k], reverse=True)
    else:
        ranked = [pick.value] if pick.value is not None else []
    return [v for v in ranked if v in allowed]


async def resolve_pick(field: Pick, question: Question, req: FormRequest, *,
                       same_first_answer: Callable[[str, str], bool], ruled_out: Collection[str],
                       filler: FormFiller, settler: Settler | None, t: Thresholds, budget: TurnBudget) -> Resolved:
    if picked(field, t) is not None:
        return Resolved(field.value, None, ())
    if question.type != "choice":
        raise ValueError("only a choice question goes through the unsure loop")
    options = [v for v, _ in question.choices]
    readings = _ranked(field, options)
    if len(readings) >= 2 and same_first_answer(readings[0], readings[1]):
        return Resolved(readings[0], None, ("same_answer",))

    steps: list[str] = []
    left = [v for v in options if v not in ruled_out]
    if left and len(left) < len(options):
        steps.append("narrowed")
        if len(left) == 1:
            return Resolved(left[0], None, tuple(steps))
        question = replace(question, choices=tuple(c for c in question.choices if c[0] not in ruled_out))
        again = await filler.fill_one(req, question)
        if picked(again, t) in left:
            return Resolved(again.value, None, tuple(steps))
        readings = list(dict.fromkeys([*_ranked(field, left), *_ranked(again, left)]))
    else:
        left = options

    larger = None
    if settler is not None and budget.larger_settles_left > 0:
        budget.larger_settles_left -= 1
        steps.append("larger")
        try:
            got = await settler.settle(req, question)
        except ModelRequestRejected:  # before ModelUnavailable: it is one, but it is our own malformed request
            logger.error("the larger model refused our request to settle %s as malformed; the caller is asked",
                         question.name, exc_info=True)
            larger = "request_rejected"
        except ModelUnavailable:
            logger.warning("the larger model was unavailable to settle %s; the caller is asked", question.name)
            larger = "unavailable"
        else:
            if got.value is not None and got.value in left:
                return Resolved(got.value, None, tuple(steps))
            larger = "cannot_tell" if got.value is None else "not_offered"

    steps.append("ask_caller")
    ask = (readings[0], readings[1]) if len(readings) >= 2 else None
    return Resolved(None, ask, tuple(steps), larger)


# ---- the two states that settle unsure picks ---------------------------------------------------------------------

def settler_for(ctx: "TurnContext") -> Settler | None:
    """The larger model, only in a state whose freedom level lets it settle unsure picks."""
    s = ctx.deps.settings
    setting = s.states.get(ctx.conversation.state.value)
    allowed = setting is not None and s.freedom[setting.freedom].settle_unsure
    return ctx.deps.settler if allowed else None


def fired(res: Resolved) -> list[dict]:
    """What the turn record keeps: the steps tried, and why the larger model settled nothing when it was asked."""
    out: list[dict] = [{"unsure_steps": list(res.steps)}] if res.steps else []
    if res.larger is not None:
        out.append({"rule": "larger_model", "outcome": res.larger})
    return out


async def _in_state(ctx: "TurnContext", name: str, field: Pick, *, same_first_answer: Callable[[str, str], bool],
                    ruled_out: Collection[str]) -> Resolved:
    req = ctx.request
    question = None if req is None else next((q for q in build_questions(req) if q.name == name), None)
    if question is None:  # no request for this turn to ask about: nothing but the caller can settle it
        readings = _ranked(field, field.options or ([field.value] if field.value else []))
        res = Resolved(None, (readings[0], readings[1]) if len(readings) >= 2 else None, ("ask_caller",))
    else:
        res = await resolve_pick(field, question, req, same_first_answer=same_first_answer, ruled_out=ruled_out,
                                 filler=ctx.deps.filler, settler=settler_for(ctx), t=ctx.deps.thresholds,
                                 budget=ctx.budget)
    record(ctx, name, field, res)
    return res


def record(ctx: "TurnContext", name: str, field: Pick, res: Resolved) -> None:
    """The events of an unsure pick: the field below its threshold and where it went, and each time it was passed on
    to the larger model or to the caller."""
    t = ctx.deps.thresholds
    if picked(field, t) is not None:
        return
    if ctx.settled is not None:  # the inspector shows, beside the question, how its answer was settled
        ctx.settled[name] = {"value": res.value, "steps": list(res.steps), "larger": res.larger}
    went_to = "caller" if res.value is None else \
        "larger_model" if res.steps and res.steps[-1] == "larger" else (res.steps[-1] if res.steps else "taken")
    ctx.record("below_threshold", field=name, options=sorted(field.options), probabilities=dict(field.options),
               threshold=t.pick_confident, went_to=went_to, transcript=None)
    if "larger" in res.steps:
        why = f"{name} was below its threshold and the cheaper steps did not settle it"
        ctx.record("escalation", why=why, went_to="larger_model", transcript=None)
    if res.value is None:
        ctx.record("escalation", why=f"{name} was still unsure after every cheaper step"
                   + (f" (the larger model: {res.larger})" if res.larger else ""), went_to="caller", transcript=None)


async def need_in_focus(ctx: "TurnContext", claims: Sequence["Claim"]) -> Resolved:
    """The caller's need, with claims in focus: two needs with the same opening for every claim in focus are the same
    answer, and a need none of the claims can have (no denial reason, no documents outstanding) is ruled out."""
    s = ctx.deps.settings

    def same_first_answer(a: str, b: str) -> bool:
        return bool(claims) and all(opening(a, c, s) == opening(b, c, s) for c in claims)

    ruled_out = set()
    if not any((c.denial_reason or "").strip() for c in claims):
        ruled_out.add("denial_question")
    if not any(c.documents for c in claims):
        ruled_out.add("document_submission")
    return await _in_state(ctx, "need", ctx.form.need, same_first_answer=same_first_answer, ruled_out=ruled_out)


async def answer_wanted(ctx: "TurnContext") -> Resolved:
    """The answer the caller wants: nothing is ruled out and two readings are never the same answer, so the larger
    model is asked, then the caller."""
    return await _in_state(ctx, "answer_wanted", ctx.form.answer_wanted, same_first_answer=lambda a, b: a == b,
                           ruled_out=())
