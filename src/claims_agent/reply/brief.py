"""What the wording model is handed: the facts it may use, the parts of the reply, and the caller's context."""

from dataclasses import dataclass

from claims_agent.memory import Pending
from claims_agent.states import State


@dataclass(frozen=True)
class Fact:
    kind: str
    text: str
    claim: str | None = None


@dataclass(frozen=True)
class ReplyPart:
    instruction: str
    facts: tuple[Fact, ...] = ()
    plain: str = ""
    pending: Pending | None = None
    prepared: str = ""  # a reply to this part worded ahead and checked (see claims_agent.prepared)


@dataclass(frozen=True)
class ReplyBrief:
    state: State
    parts: tuple[ReplyPart, ...]
    mood: str
    caller_message: str
    previous_reply: str | None
    english: bool
    verified: bool
    caller_claim_numbers: tuple[str, ...] = ()

    def facts(self) -> tuple[Fact, ...]:
        return tuple(f for p in self.parts for f in p.facts)

    def plain_text(self) -> str:
        """The code-written fallback reply: each part's plain text, separated by blank lines."""
        return "\n\n".join(p.plain for p in self.parts if p.plain)
