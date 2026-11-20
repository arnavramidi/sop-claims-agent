"""The data-driven follow-ups: which one comes next, and how one is filled for a claim.

A template comes from an untrusted data file, so it is filled only by plain replacement of the blanks it
names (the loader already refused templates with anything else); nothing in it is ever evaluated."""

from collections.abc import Collection, Mapping
from datetime import date

from claims_agent.answers.facts import AnswerResult, deadline_fact, exists_for, render, status_fact
from claims_agent.db.claims import Claim
from claims_agent.db.guidance import Guidance
from claims_agent.loader.read import FollowupRow
from claims_agent.reply.brief import Fact
from claims_agent.settings import Settings

__all__ = ["exists_for", "next_followup", "fill"]


def _after(s: Settings) -> dict[str, list[str]]:
    out = {}
    for topic, before in s.followups.comes_after.items():
        out[topic] = [before] if isinstance(before, str) else list(before)
    return out


def next_followup(need: str, claim: Claim, answered: Collection[str], g: Guidance, s: Settings) -> FollowupRow | None:
    """The next follow-up for the need: those linked to it that exist for the claim and are not answered, in
    the settings' order (names the data lacks are ignored) then the data's order, skipping one whose
    `comes_after` topic is still waiting to be answered. A cycle of waits is ignored rather than offering nothing."""
    left = followup_candidates(need, claim, answered, g, s)
    waiting = {f.topic for f in left}
    after = _after(s)
    for f in left:
        if not any(t in waiting for t in after.get(f.topic, ())):
            return f
    return left[0] if left else None


def followup_candidates(need: str, claim: Claim, answered: Collection[str], g: Guidance, s: Settings) -> list[FollowupRow]:
    """The follow-ups linked to the need that exist for the claim and are not answered, in the settings' order
    (names the data lacks are ignored) then the data's order."""
    by_topic = {f.topic: f for f in g.followups}
    ranked = [by_topic[t] for t in s.followups.order if t in by_topic]
    ranked += [f for f in g.followups if f not in ranked]
    return [f for f in ranked if need in f.needs and exists_for(f, claim, g, s) and f.topic not in answered]


def _not_available(claim: Claim) -> AnswerResult:
    fact = Fact("not_on_file", f"That detail is not on file for claim {claim.case_id}.", claim.case_id)
    return AnswerResult((fact,), grounded=False, offer_person=True)


def fill(f: FollowupRow, claim: Claim, g: Guidance, s: Settings, today: date, *,
         document: str | None = None) -> AnswerResult:
    if not exists_for(f, claim, g, s):
        return _not_available(claim)
    special = s.followups.special.get(f.topic)
    facts: list[Fact] = []
    offer = False
    deadline, passed = deadline_fact(claim, today)
    if special and special.drop_when_deadline_passed and passed:
        facts, offer = [deadline], True
    else:
        rendered = render(f, claim, g, s, document)
        if rendered is None:
            return _not_available(claim)
        body = rendered.body
        if claim.case_id not in body:
            body = f"For claim {claim.case_id}: {body}"
        facts.append(Fact("guidance", body, claim.case_id))
        if rendered.names_documents:
            facts.append(Fact("documents", f"Documents outstanding for claim {claim.case_id}: {len(claim.documents)}.",
                              claim.case_id))
        if rendered.left_out:
            facts.append(Fact("not_on_file", f"No specific file requirements are on file for: "
                                             f"{', '.join(rendered.left_out)}.", claim.case_id))
        if special and special.with_deadline and claim.appeal_deadline:
            facts.append(deadline)
            offer = passed
    # Follow-up texts are written for denied claims: without a denial reason a model words them live.
    live = not claim.denial_reason
    if live:
        facts.append(status_fact(claim))
    return AnswerResult(tuple(facts), reword_live=live, offer_person=offer)
