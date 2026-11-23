"""Checks that run on every message, whatever the state. Pure: no database, no model.

A form field a model filled may be missing (None) or refused (probability None). Nothing that ends, resets
or hands over the conversation fires on less than a confident yes (at or above the top of the unsure band);
an unsure request for a person becomes an offer to confirm. At most one
part per turn carries an offer of a person.
"""

import unicodedata
from dataclasses import dataclass
from typing import Literal

from claims_agent.db.claims import filter_from_form
from claims_agent.form.fields import Form
from claims_agent.form.read import confident_yes, picked, unsure, yes
from claims_agent.memory import Pending
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.rules.mood import FRUSTRATED, caller_mood
from claims_agent.settings import Thresholds

Terminal = Literal["handover", "reset_identity", "wrap_up", "close", "confirm_policyholder"]

PERSON_OFFER = Pending(kind="offer_person", label="speak to a person?", yes_no=True)
# Asked of a representative when it is unclear whether the message is still about the policyholder they act for.
SAME_POLICYHOLDER = Pending(kind="same_policyholder", label="still the same policyholder?", yes_no=True)
OFFER_BEFORE_CLOSING = ReplyPart("offer_person_before_closing", plain="I understand if you'd like to stop here. Would "
                                "you like me to pass you to a person before we finish?", pending=PERSON_OFFER)
# Why identity is checked before anything else, for a caller who asks why or wants to skip it: the reason travels as a
# fact, so the reply explains it rather than only asking again.
WHY_VERIFY = Fact("why_verify", "Claim details are private to the policyholder. Checking who is calling protects the "
                                "caller's information: it keeps those details from being shared with anyone else, so "
                                "this step can't be skipped.")
EXPLAIN_STEP = ReplyPart("explain_step", (WHY_VERIFY,), "Claim details are private, and checking who is calling "
                                                       "protects your information, so this step can't be skipped.")
ASK_SAME_POLICYHOLDER = ReplyPart("ask_same_policyholder", plain="Is this still about the policyholder you're calling for?",
                                  pending=SAME_POLICYHOLDER)


@dataclass(frozen=True)
class GlobalOutcome:
    terminal: Terminal | None
    parts: tuple[ReplyPart, ...]


def _selector(ch: str) -> bool:
    return 0xFE00 <= ord(ch) <= 0xFE0F or 0xE0100 <= ord(ch) <= 0xE01EF


def usable(message: str, pending: Pending | None) -> bool:
    return _count(message) >= (1 if pending is not None else 2)


def _count(message: str) -> int:
    """Usable means two letters or digits, or one when any question is pending.

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


def global_outcome(form: Form, *, verified: bool, pending: Pending | None, t: Thresholds,
                   acting_for_policyholder: bool = False, names_someone_else: bool = False,
                   maybe_someone_else: bool = False) -> GlobalOutcome:
    """A different person resets the conversation. For a caller acting for the policyholder, "someone else's policy"
    is that policyholder's, so it takes a different policyholder named in the message (see rules/other_person.py).
    When that is unclear, the representative is asked whether it is still the same policyholder, and nothing else is
    done that turn (no claim is told about the wrong person); a clear "no" to that question resets."""
    offer_pending = pending is not None and pending.kind == "offer_person"
    reply = picked(form.pending_reply, t) if offer_pending else None
    accepted = reply == "yes"  # a confident pick (pick_confident); below it is no answer and nothing is added
    if confident_yes(form.wants_person, t) or accepted:
        return GlobalOutcome("handover", ())
    if confident_yes(form.other_person_policy, t) and (not acting_for_policyholder or names_someone_else):
        return GlobalOutcome("reset_identity", ())
    same_asked = pending is not None and pending.kind == SAME_POLICYHOLDER.kind
    if same_asked and picked(form.pending_reply, t) == "no":
        return GlobalOutcome("reset_identity", ())
    if acting_for_policyholder and maybe_someone_else and not same_asked:
        return GlobalOutcome("confirm_policyholder", (ASK_SAME_POLICYHOLDER,))
    if confident_yes(form.done, t):
        if verified:
            return GlobalOutcome("wrap_up", ())
        # An upset caller who gives up before being verified is offered a person once before the conversation
        # closes; the state's own step then says nothing more. Done again, or after the offer, it closes.
        if caller_mood(form.mood, t) in FRUSTRATED and not offer_pending:
            return GlobalOutcome(None, (OFFER_BEFORE_CLOSING,))
        return GlobalOutcome("close", ())

    about = picked(form.about, t)
    unsupported = about == "unsupported_insurance"  # the specific offer: it wins over a generic one
    parts: list[ReplyPart] = []
    if not unsupported and unsure(form.wants_person, t) and not offer_pending:
        parts.append(ReplyPart("offer_person", pending=PERSON_OFFER))
    if not verified and (yes(form.skips_step, t) or yes(form.asks_why_step, t)):
        # The explanation is why identity checks matter; once verified, no step is there to protect the caller's
        # identity, so a wish to skip ahead is simply answered by the state's own step.
        parts.append(EXPLAIN_STEP)
    if about == "off_topic":
        parts.append(ReplyPart("decline_off_topic"))
    elif unsupported:
        parts.append(ReplyPart("unsupported_offer_person", pending=PERSON_OFFER))
    if yes(form.asks_personal_detail, t):
        parts.append(ReplyPart("cannot_read_out_personal_detail"))
    if not yes(form.english, t):
        parts.append(ReplyPart("offer_person_in_language"))  # no pending: a bare "yes" must not hand over
    return GlobalOutcome(None, tuple(parts))


# The answers to the email offer that are about the offer itself.
EMAIL_REPLIES = ("send", "dont_send", "whats_in_it", "which_address", "different_address")


def asks_something(form: Form, t: Thresholds, *, claim_number: str | None = None,
                   pending: Pending | None = None) -> bool:
    """Whether the message asks the claims line for anything: a need, an answer, a calculation or a money figure, a
    claim (by number, by description, from a list, or all of them), help with a document, a yes or no to the yes/no
    question pending, or a reply to the email offer. Thanks, goodbyes, small talk and off-topic questions ask nothing.
    A request for the summary email is read separately (`wants_email_summary`). In a message that is off-topic or asks
    for the email, the vague readings (an answer that is "something else", a bare mention of a claim, and in an
    off-topic message a need) are that message itself, not a question about a claim."""
    answer = picked(form.answer_wanted, t)
    off_topic = picked(form.about, t) == "off_topic"
    itself = off_topic or yes(form.wants_email_summary, t)
    if answer not in (None, "none") and not (answer == "something_else" and itself):
        return True
    # In an off-topic message a need read into it ("a general claim question") is that message too, like "something
    # else". (A request for the email keeps its need: "email me why it was denied" asks for the reason.)
    if picked(form.need, t) not in (None, "none") and not off_topic:
        return True
    if picked(form.calculation, t) not in (None, "none"):
        return True
    if any(yes(v, t) for v in form.figures.values()) or yes(form.cant_get_document, t):
        return True
    # A request for the summary "of this" or "of my claim" names the claim it is about, and asks nothing of it: a
    # bare mention (the form's "mentions a claim", or one claim meant, or the same as before) counts only in a message
    # that is not itself the email request or off-topic. A claim number, a description, several or all claims, or
    # anything specific the message asks still counts.
    how_many = picked(form.how_many, t)
    if (yes(form.mentions_claim, t) or how_many in ("one", "same_as_before")) and not itself:
        return True
    if yes(form.quoted_claim_number, t) or (claim_number or "").strip():
        return True
    if filter_from_form(form, t).conditions() or how_many in ("several", "all"):
        return True
    if picked(form.listed_claim, t) not in (None, "none"):
        return True
    if pending is not None and pending.yes_no and picked(form.pending_reply, t) in ("yes", "no"):
        return True
    return form.email_offer_reply is not None and picked(form.email_offer_reply, t) in EMAIL_REPLIES
