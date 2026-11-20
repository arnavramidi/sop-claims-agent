"""Checks that run on every message, whatever the state. Pure: no database, no model.

A form field a model filled may be missing (None) or refused (probability None). Nothing that ends, resets
or hands over the conversation fires on less than a confident yes (at or above the top of the unsure band);
an unsure request for a person becomes an offer to confirm. At most one
part per turn carries an offer of a person.
"""

import unicodedata
from dataclasses import dataclass
from typing import Literal

from claims_agent.form.fields import Form
from claims_agent.form.read import confident_yes, picked, unsure, yes
from claims_agent.memory import Pending
from claims_agent.reply.brief import ReplyPart
from claims_agent.settings import Thresholds

Terminal = Literal["handover", "reset_identity", "wrap_up", "close"]

PERSON_OFFER = Pending(kind="offer_person", label="speak to a person?", yes_no=True)


@dataclass(frozen=True)
class GlobalOutcome:
    terminal: Terminal | None
    parts: tuple[ReplyPart, ...]


def _selector(ch: str) -> bool:
    return 0xFE00 <= ord(ch) <= 0xFE0F or 0xE0100 <= ord(ch) <= 0xE01EF


def usable(message: str, pending: Pending | None) -> bool:
    return _count(message) >= (1 if pending is not None else 2)


def _count(message: str) -> int:
    """Ruling 12: usable means two letters or digits, or one when any question is pending.

    A combining mark counts only straight after a letter or digit, never after another mark ("जी" is two); variation
    selectors and enclosing marks (emoji, keycaps) never count, so an emoji-only message is unreadable.
    """
    count, prev_base = 0, False
    for ch in message:
        cat = unicodedata.category(ch)
        if cat[0] in "LN":
            count += 1
            prev_base = True
            continue
        if prev_base and cat in ("Mn", "Mc") and not _selector(ch):
            count += 1
        prev_base = False
    return count


def global_outcome(form: Form, *, verified: bool, pending: Pending | None, t: Thresholds) -> GlobalOutcome:
    offer_pending = pending is not None and pending.kind == "offer_person"
    reply = picked(form.pending_reply, t) if offer_pending else None
    accepted = reply == "yes"  # a confident pick (pick_confident); below it is no answer and nothing is added
    if confident_yes(form.wants_person, t) or accepted:
        return GlobalOutcome("handover", ())
    if confident_yes(form.other_person_policy, t):
        return GlobalOutcome("reset_identity", ())
    if confident_yes(form.done, t):
        return GlobalOutcome("wrap_up" if verified else "close", ())

    about = picked(form.about, t)
    unsupported = about == "unsupported_insurance"  # the specific offer: it wins over a generic one
    parts: list[ReplyPart] = []
    if not unsupported and unsure(form.wants_person, t) and not offer_pending:
        parts.append(ReplyPart("offer_person", pending=PERSON_OFFER))
    if yes(form.skips_step, t) or yes(form.asks_why_step, t):
        parts.append(ReplyPart("explain_step"))
    if about == "off_topic":
        parts.append(ReplyPart("decline_off_topic"))
    elif unsupported:
        parts.append(ReplyPart("unsupported_offer_person", pending=PERSON_OFFER))
    if yes(form.asks_personal_detail, t):
        parts.append(ReplyPart("cannot_read_out_personal_detail"))
    if not yes(form.english, t):
        parts.append(ReplyPart("offer_person_in_language"))  # no pending: a bare "yes" must not hand over
    return GlobalOutcome(None, tuple(parts))
