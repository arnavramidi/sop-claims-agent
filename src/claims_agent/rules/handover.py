"""The note a person reads when a conversation is handed to them. Written by code, never by a model."""

from typing import TYPE_CHECKING

from claims_agent.db.conversations import Conversation
from claims_agent.inspect import mask_ids
from claims_agent.memory import Memory

if TYPE_CHECKING:
    from claims_agent.engine import TurnContext


def note(conversation: Conversation, memory: Memory, message: str, reason: str) -> dict:
    """Verified or not, which customer, the claims in focus, what was asked and why it was handed over.
    "Which customer" is only ever the customer the database verified; a candidate is never named."""
    return {
        "verified": conversation.verified_party_id is not None,
        "customer": conversation.verified_party_id,
        "claims_in_focus": list(memory.focus.claim_ids),
        "asked": mask_ids(message),
        "reason": reason,
        "state": conversation.state.value,
        "turn": memory.counters.turn_no,
    }


def handover_note(ctx: "TurnContext", reason: str) -> dict:
    return note(ctx.conversation, ctx.memory, ctx.message, reason)
