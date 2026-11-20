"""One reply: worded, checked, rewritten once if it fails (when the turn's budget still allows a rewrite on the larger
model), and the plain template if the rewrite fails too or cannot be made."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from claims_agent.budget import TurnBudget
from claims_agent.reply.brief import ReplyBrief
from claims_agent.reply.check import CheckResult, check_reply
from claims_agent.reply.wording import Wording
from claims_agent.settings import Settings


@dataclass(frozen=True)
class Composed:
    text: str
    source: Literal["live", "rewrite", "template", "fixed", "last_resort"]
    checks: tuple[CheckResult, ...]


LAST_RESORT = "I'm not able to share that here. I can put you through to a person who can help."


async def compose(brief: ReplyBrief, wording: Wording, s: Settings, data_statuses: Sequence[str],
                  *, fixed: bool = False, budget: TurnBudget | None = None) -> Composed:
    """Every text that leaves is checked, the code-written ones too; `checks` ends with the check of the sent text."""
    rc = s.reply_check

    def check(text: str) -> CheckResult:
        return check_reply(text, brief, wary_status_words=rc.wary_status_words, data_statuses=data_statuses,
                           status_forms=rc.status_forms, max_length=rc.max_length)

    def plain(source: Literal["template", "fixed"], earlier: tuple[CheckResult, ...]) -> Composed:
        text = brief.plain_text()
        r = check(text)
        if r.passed:
            return Composed(text, source, (*earlier, r))
        return Composed(LAST_RESORT, "last_resort", (*earlier, r, check(LAST_RESORT)))

    if fixed:
        return plain("fixed", ())
    first = await wording.word(brief)
    r1 = check(first)
    if r1.passed:
        return Composed(first, "live", (r1,))
    if budget is not None:  # the rewrite runs on the larger model: at most once per turn
        if budget.larger_rewrites_left <= 0:
            return plain("template", (r1,))
        budget.larger_rewrites_left -= 1
    second = await wording.rewrite(brief, first, r1.failures)
    r2 = check(second)
    if r2.passed:
        return Composed(second, "rewrite", (r1, r2))
    return plain("template", (r1, r2))
