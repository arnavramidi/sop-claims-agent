"""The rules side of arithmetic: when a caller asks for a total, a difference, a ranking, a count or a day count, code
works it out over the caller's own claims (the database's row rules decide which claims exist) and puts the result in
facts. RESOLVE_INTENT calls `over_filter` when no single claim is meant; PROCESS_CASE calls `over_claims` for the
claims in focus. Both are called before an answer pick is acted on."""

from collections.abc import Sequence

from claims_agent.answers.arithmetic import aggregate, calculation_facts, figures_needed
from claims_agent.db.claims import Claim, ClaimFilter, merge, filter_from_form
from claims_agent.engine import StepOutcome, TurnContext, guidance
from claims_agent.form.fields import YesNo
from claims_agent.form.read import picked, yes
from claims_agent.loader.read import MONEY_FIELDS
from claims_agent.memory import Pending
from claims_agent.reply.brief import ReplyPart
from claims_agent.states import State

# Needs no single claim: runs over a filter. The others (difference, days open, days left) need claims in focus.
OVER_A_FILTER = ("total", "highest", "lowest", "count")
NOT_ASKED = YesNo(p=0.0)
WHICH_FIGURE = Pending(kind="which_figure", label="which money figure?", yes_no=False)


def wanted(ctx: TurnContext) -> tuple[str, list[str]] | None:
    """The confident calculation this turn's message asks for (never "none"), and the money figures it names."""
    calc = picked(ctx.form.calculation, ctx.deps.thresholds)
    if calc in (None, "none"):
        return None
    if yes(ctx.form.no_figure_said, ctx.deps.thresholds):
        return calc, []
    return calc, [m for m in MONEY_FIELDS if yes(ctx.form.figures.get(m, NOT_ASKED), ctx.deps.thresholds)]


async def parts(ctx: TurnContext, calc: str, figures: list[str], f: ClaimFilter, claims: Sequence[Claim],
                case_ids: Sequence[str] | None) -> tuple[list[ReplyPart], bool]:
    """The reply parts for one calculation and whether the reply is grounded in the data."""
    # Imported late: both modules import this one (directly or through the state that calls it).
    from claims_agent.rules.process_case import _parts
    from claims_agent.rules.resolve_intent import sentences

    g, s = await guidance(ctx.deps, ctx.conn), ctx.deps.settings
    agg = None
    if calc in OVER_A_FILTER and len(figures) == figures_needed(calc):
        agg = await aggregate(ctx.conn, f, ctx.today, figures[0] if figures else MONEY_FIELDS[0],
                              order="lowest" if calc == "lowest" else "highest",
                              in_progress=s.claims.in_progress_statuses, case_ids=case_ids)
    result = calculation_facts(calc, figures, claims, agg, g, s, ctx.today)
    if not result.grounded:
        facts = result.facts
        return [ReplyPart("ask_which_figure", facts, sentences("Which of these figures do you mean?", facts),
                          pending=WHICH_FIGURE)], False
    return _parts([result])


async def over_filter(ctx: TurnContext, how_many: str, stored: ClaimFilter | None) -> StepOutcome | None:
    """A total, highest, lowest or count the caller asks for over claims in general (how many "all" or "none"): over
    the current filter, or over every claim when there is none. The state stays."""
    asked = wanted(ctx)
    if asked is None or asked[0] not in OVER_A_FILTER or how_many not in ("all", "none"):
        return None
    if (ctx.stated.claim_number or "").strip():
        return None
    t = ctx.deps.thresholds
    new = filter_from_form(ctx.form, t)
    relation = picked(ctx.form.search_relation, t) or "no_list_shown"
    f = merge(stored, new, relation) if stored is not None else new
    made, grounded = await parts(ctx, asked[0], asked[1], f, [], None)
    return StepOutcome(State.RESOLVE_INTENT, made, grounded=grounded,
                       fired=[{"rule": "calculation", "calculation": asked[0], "figures": asked[1]}])
