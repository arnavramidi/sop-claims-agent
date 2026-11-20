"""Judge a verification attempt from what the database says matched. Pure: no database, no model."""

from collections.abc import Sequence
from dataclasses import dataclass

from claims_agent.identity.types import KINDS, CustomerMatch

KINDS_NEEDED = 3


@dataclass(frozen=True)
class VerifyDecision:
    candidate: str | None  # party_id
    verified_party: str | None
    wrong_now: frozenset[str]  # kinds (and "policy") judged wrong on everything held so far
    all_kinds_no_match: bool  # all five held and nobody has three, or no candidate can be singled out


def _only_one(parties: list[CustomerMatch]) -> CustomerMatch | None:
    return parties[0] if len(parties) == 1 else None


def judge(held: frozenset[str], matches: Sequence[CustomerMatch]) -> VerifyDecision:
    if len({m.party_id for m in matches}) != len(matches):
        raise ValueError("a customer is listed twice")
    real = frozenset(KINDS) & held  # only the five kinds ever count
    held_kinds = {m.party_id: m.kinds & real for m in matches}
    count = {party: len(kinds) for party, kinds in held_kinds.items()}
    top = max(count.values(), default=0)

    candidate: CustomerMatch | None = None
    if top > 0:
        tied = [m for m in matches if count[m.party_id] == top]
        candidate = _only_one(tied) or _only_one([m for m in tied if m.policy_ok])
    else:
        candidate = _only_one([m for m in matches if m.policy_ok])

    policy_held = "policy" in held
    if candidate is not None:
        own = held_kinds[candidate.party_id]
        wrong = set(real - own)
        if policy_held and not candidate.policy_ok:
            wrong.add("policy")
        verified = (
            candidate.party_id
            if count[candidate.party_id] >= KINDS_NEEDED and (not policy_held or candidate.policy_ok)
            else None
        )
    else:
        seen = set().union(*held_kinds.values()) if held_kinds else set()
        wrong = set(real - seen)
        if policy_held and not any(m.policy_ok for m in matches):
            wrong.add("policy")
        verified = None

    all_held = len(real) == len(KINDS)
    return VerifyDecision(
        candidate=candidate.party_id if candidate else None,
        verified_party=verified,
        wrong_now=frozenset(wrong),
        all_kinds_no_match=all_held and (top < KINDS_NEEDED or candidate is None),
    )
