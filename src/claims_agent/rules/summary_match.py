"""Picking one of a few listed claims by what the caller said about it, read against each claim's one-line summary.

Used only for a verified caller, and only over the claims that were listed (two to five): they are read again through
the turn's connection, so the database's row rules decide what is visible and the model never sees another claim. The
summaries are text from a data file; they travel as the descriptions of the choices, never inside the instruction.
The model's answer is one of the listed claim numbers or "cannot tell"; anything else settles nothing.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claims_agent.db.claims import Claim, fetch_ids
from claims_agent.form.questions import Question
from claims_agent.form.read import picked
from claims_agent.models.errors import ModelRequestRejected, ModelUnavailable
from claims_agent.reply.brief import Fact

if TYPE_CHECKING:
    from claims_agent.engine import TurnContext

logger = logging.getLogger("claims_agent.rules.summary_match")

CANNOT_TELL = "cannot_tell"
INSTRUCTIONS = ("Which of the listed claims does the caller mean? Choose by the summary on file for each claim. "
                "Choose cannot_tell when the caller's words do not single out one claim.")
NOT_SEPARATED = Fact("cannot_tell", "The description does not tell these claims apart")


def summary_question(claims: Sequence[Claim]) -> Question:
    """One choice per listed claim, described by its summary, plus "cannot_tell"."""
    choices = tuple((c.case_id, c.summary or "no summary on file") for c in claims)
    return Question("claim_summary", "choice", INSTRUCTIONS,
                    (*choices, (CANNOT_TELL, "the description does not settle which claim")))


def what_differs(claims: Sequence[Claim]) -> list[str]:
    """Among the type, status, year and month, those that are not the same for every claim, in that order."""
    seen = {
        "case_type": {c.case_type for c in claims},
        "status": {c.status for c in claims},
        "year": {c.created_at.year if c.created_at else None for c in claims},
        "month": {c.created_at.month if c.created_at else None for c in claims},
    }
    return [name for name, values in seen.items() if len(values) > 1]


def found_claim_fact(claim: Claim) -> Fact:
    """The claim as it was found: its number, type, opened date and its summary as written, and nothing more."""
    details = ", ".join(x for x in (claim.case_type, f"opened {claim.created_at.isoformat()}"
                                    if claim.created_at else None) if x)
    text = f"Claim {claim.case_id}" + (f": {details}" if details else "") + "."
    if claim.summary:
        text += f' Summary on file: "{claim.summary}".'
    return Fact("found_claim", text, claim.case_id)


@dataclass(frozen=True)
class SummaryPick:
    case_id: str | None
    outcome: str  # picked | cannot_tell | unsure | not_listed | not_asked | unavailable | request_rejected


async def try_summary(ctx: "TurnContext", claims: Sequence[Claim]) -> SummaryPick:
    """Ask the fast model which of `claims` was meant. A key or account problem is not caught here (the engine rolls
    the turn back); an outage or a request the model refused as malformed leaves the caller to be asked."""
    if ctx.conversation.verified_party_id is None:
        raise ValueError("claim summaries are read only for a verified caller")
    if ctx.request is None or ctx.budget is None or ctx.budget.summary_picks_left <= 0:
        return SummaryPick(None, "not_asked")
    shown = await fetch_ids(ctx.conn, [c.case_id for c in claims])
    if len(shown) < 2 or not any(c.summary for c in shown):
        return SummaryPick(None, "not_asked")
    ctx.budget.summary_picks_left -= 1
    question = summary_question(shown)
    try:
        pick = await ctx.deps.filler.fill_one(ctx.request, question)
    except ModelRequestRejected:  # before ModelUnavailable: it is one, but it is our own malformed request
        logger.error("the model refused our request to pick a claim by summary as malformed; the caller is asked",
                     exc_info=True)
        return SummaryPick(None, "request_rejected")
    except ModelUnavailable:
        logger.warning("the model was unavailable to pick a claim by summary; the caller is asked")
        return SummaryPick(None, "unavailable")
    value = picked(pick, ctx.deps.thresholds)
    if value is None:
        return SummaryPick(None, "unsure")
    if value == CANNOT_TELL:
        return SummaryPick(None, "cannot_tell")
    if value not in {c.case_id for c in shown}:
        return SummaryPick(None, "not_listed")
    return SummaryPick(value, "picked")


async def pick_by_summary(ctx: "TurnContext", claims: Sequence[Claim]) -> str | None:
    """A case_id when the model is confident and the claim is among those listed, else None."""
    return (await try_summary(ctx, claims)).case_id
