"""Arithmetic over the caller's own claims, worded as facts.

The database adds up and ranks (`db.claims.aggregate`); this module only turns the result, or the claims' own figures,
into facts in the canonical forms the reply check reads: amounts with two decimals, day counts as "N days", a count as
a digit. A model never does a sum. A figure that is missing is "unavailable", never 0.00, and a total says how many
claims it covers and which had no figure on file.
"""

from collections.abc import Sequence
from datetime import date

from claims_agent.db.claims import TOP_SIZE, Aggregate, Claim, aggregate
from claims_agent.db.guidance import Guidance
from claims_agent.form.questions import CALCULATIONS
from claims_agent.loader.read import MONEY_FIELDS
from claims_agent.reply.brief import Fact
from claims_agent.settings import Settings
from claims_agent.answers.facts import AnswerResult, EXPECTED_NOTE, _days, _shown, deadline_fact, figure_words

__all__ = ["CALCULATIONS", "Aggregate", "aggregate", "calculation_facts", "figures_needed"]

# How many money figures each calculation needs the caller to name.
FIGURES_NEEDED = {"total": 1, "highest": 1, "lowest": 1, "difference": 2}


def figures_needed(calc: str) -> int:
    return FIGURES_NEEDED.get(calc, 0)


def _ask_which_figure(g: Guidance) -> AnswerResult:
    """A5: the four figures in the glossary's own words, so the caller can say which."""
    facts = tuple(Fact("possible_figure", f"Possible figure: {figure_words(n)}" + (f". Meaning: {g.glossary[n]}"
                                                                                   if g.glossary.get(n) else ""))
                  for n in MONEY_FIELDS)
    return AnswerResult(facts, grounded=False)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _total(figure: str, agg: Aggregate) -> list[Fact]:
    label = figure_words(figure)
    covered = agg.count - len(agg.missing)
    if covered == 0:
        return [Fact("not_on_file", f"No {label} is on file for any of the {_plural(agg.count, 'claim')} asked about.")]
    note = f" {EXPECTED_NOTE}" if figure == "expected_reimbursement_amount" else ""
    expected = "expected " if figure == "expected_reimbursement_amount" else ""
    facts = [Fact("total", f"Total {expected}{label} across {_plural(covered, 'claim')}: {agg.total:.2f}.{note}")]
    if figure == "net_pay":
        facts += [Fact("total", f"Claim {c} net pay: not paid yet (counted as 0.00).", c)
                  for c in agg.not_paid_yet[:TOP_SIZE]]
        if len(agg.not_paid_yet) > TOP_SIZE:  # a long list is counted, never listed in full
            facts.append(Fact("total", f"Claims not paid yet (counted as 0.00), {len(agg.not_paid_yet)} in all; "
                                       f"the first {TOP_SIZE} are named above."))
    if agg.missing:
        facts.append(Fact("total", f"Claims with no {label} on file, left out of the total: {len(agg.missing)}."))
        facts += [Fact("total", f"Claim {c} {label}: unavailable (not on file).", c) for c in agg.missing[:TOP_SIZE]]
    return facts


def _ranked(calc: str, figure: str, agg: Aggregate) -> list[Fact]:
    label = figure_words(figure)
    if not agg.top:
        return [Fact("not_on_file", f"No {label} is on file for any of the {_plural(agg.count, 'claim')} asked about.")]
    best = agg.top[0][1]
    winners = [c for c, v in agg.top if v == best]
    who = " and ".join(winners)
    verb = f"claim {who} with {best:.2f}" if len(winners) == 1 else f"claims {who}, each with {best:.2f}"
    facts = [Fact("ranking", f"{calc.capitalize()} {label} among {_plural(agg.count - len(agg.missing), 'claim')}: {verb}.",
                  winners[0])]
    if len(agg.top) > 1:
        facts.append(Fact("ranking", f"Ranked by {label}, {calc} first: "
                                     + ", ".join(f"{c} {v:.2f}" for c, v in agg.top) + "."))
    if agg.missing:
        more = f", and {len(agg.missing) - TOP_SIZE} more" if len(agg.missing) > TOP_SIZE else ""
        facts.append(Fact("ranking", f"Claims with no {label} on file, left out: "
                                     + ", ".join(agg.missing[:TOP_SIZE]) + more + "."))
    return facts


def _difference(figures: Sequence[str], claims: Sequence[Claim], s: Settings) -> list[Fact]:
    first, second = figures
    facts: list[Fact] = []
    for c in claims:
        (a, va), (b, vb) = _shown(c, first, s), _shown(c, second, s)
        facts += [Fact("amount", f"Claim {c.case_id} {figure_words(first)}: {a}.", c.case_id),
                  Fact("amount", f"Claim {c.case_id} {figure_words(second)}: {b}.", c.case_id)]
        if va is not None and vb is not None:
            facts.append(Fact("difference", f"Difference between {figure_words(first)} and {figure_words(second)} for "
                                            f"claim {c.case_id}: {abs(va - vb):.2f}.", c.case_id))
        else:
            facts.append(Fact("difference", f"No difference can be given for claim {c.case_id}: a figure is "
                                            "unavailable or not paid yet.", c.case_id))
        facts.append(Fact("not_on_file", f"The reason for the difference is not on file for claim {c.case_id}.",
                          c.case_id))
    return facts


def _days_open(claims: Sequence[Claim], today: date) -> list[Fact]:
    facts = []
    for c in claims:
        if c.created_at is None:
            facts.append(Fact("not_on_file", f"The opened date of claim {c.case_id} is not on file.", c.case_id))
        elif (today - c.created_at).days < 0:
            facts.append(Fact("opened", f"Claim {c.case_id} is dated {c.created_at.isoformat()}, after today.", c.case_id))
        else:
            facts.append(Fact("opened", f"Claim {c.case_id} was opened on {c.created_at.isoformat()}, "
                                        f"{_days((today - c.created_at).days)} ago.", c.case_id))
    return facts


def calculation_facts(calc: str, figures: Sequence[str], claims: Sequence[Claim], agg: Aggregate | None,
                      g: Guidance, s: Settings, today: date) -> AnswerResult:
    """The facts for one calculation. `figures` are the money fields the caller named; `agg` is the database's result
    for total, highest, lowest and count (the caller runs `aggregate`); `claims` are the claims in focus (difference,
    days open, days left). Not grounded when the caller must still say which figure."""
    if calc not in CALCULATIONS or calc == "none":
        raise ValueError(f"not a calculation: {calc!r}")
    if any(f not in MONEY_FIELDS for f in figures):
        raise ValueError(f"not a money field: {[f for f in figures if f not in MONEY_FIELDS]}")
    if calc in FIGURES_NEEDED and len(figures) != FIGURES_NEEDED[calc]:
        return _ask_which_figure(g)
    if calc == "total" and agg is not None:
        return AnswerResult(tuple(_total(figures[0], agg)))
    if calc in ("highest", "lowest") and agg is not None:
        return AnswerResult(tuple(_ranked(calc, figures[0], agg)))
    if calc == "difference":
        return AnswerResult(tuple(_difference(figures, claims, s)), offer_person=False)
    if calc == "count":
        n = agg.count if agg is not None else len(claims)
        return AnswerResult((Fact("count", f"{n} claim matches." if n == 1 else f"{n} claims match."),))
    if calc == "days_open":
        return AnswerResult(tuple(_days_open(claims, today)))
    if calc == "days_left":
        results = [deadline_fact(c, today) for c in claims]
        return AnswerResult(tuple(f for f, _ in results), offer_person=any(p for _, p in results))
    raise ValueError(f"{calc} needs the database's result")
