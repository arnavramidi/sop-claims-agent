"""VERIFY_ID: collect identifying details across turns; only the database can mark the caller verified.

The comparison happens inside the database (match_identity) and so does the write (bind_policyholder,
request_consent, which re-check the three kinds themselves), and so does the count of wrong values. This step only
judges what came back, reads the database's count and decides the reply. The reply is the same whatever matched: its
facts are the verification rule and the kinds not yet given, never how many matched, how many are still needed, or
whether a person or policy exists.
"""

import logging
from datetime import date

from claims_agent.db.identity import (IdentityLocked, NameShape, bind_policyholder, match_identity, name_shape,
                                      request_consent, wrong_values)
from claims_agent.engine import STEPS, StepOutcome, TurnContext
from claims_agent.form.fields import StatedValues
from claims_agent.form.read import confident_yes, picked, yes
from claims_agent.identity.mask import mask_email
from claims_agent.identity.normalize import normalise
from claims_agent.identity.types import KINDS, FormatFailure, StatedIdentity
from claims_agent.identity.verify import KINDS_NEEDED, judge
from claims_agent.inspect import mask_ids
from claims_agent.memory import HeldValue, IdentityMemory, Pending
from claims_agent.records import flag_corrected
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.rules.every_message import PERSON_OFFER
from claims_agent.states import State

logger = logging.getLogger("claims_agent.rules.verify_id")

# How the reply names each kind. The ID is named by the kinds of ID the data holds (an SSN or a national ID), so a
# caller knows what to give, and never by the type one customer has.
ID_LABEL = "the last four digits of your SSN or national ID"
LABEL = {"name": "full name", "dob": "date of birth", "phone": "phone number", "email": "email address",
         "id": ID_LABEL}
AGAIN = {"name": "your full name", "dob": "your date of birth", "phone": "your phone number",
         "email": "your email address", "id": ID_LABEL}

RULE = Fact("verification_rule",
            f"To verify who you are, {KINDS_NEEDED} of these details must match the policy: "
            + ", ".join(LABEL[k] for k in KINDS)
            + f". A policy number helps find the policy but is not one of the {KINDS_NEEDED}.")


def _gave_a_value(v: StatedValues) -> bool:
    return any((x or "").strip() for x in (v.policyholder_name, v.phone, v.email, v.id_digits, v.policy_number,
                                           v.date_of_birth_as_stated)) or any(d.strip() for d in v.date_of_birth)


def _given(v: StatedValues) -> dict[str, str]:
    """Each identity value the extractor found in this message, as stated, by kind."""
    dob = (v.date_of_birth_as_stated or "").strip() or ", ".join(d.strip() for d in v.date_of_birth if d.strip())
    given = {"name": v.policyholder_name, "dob": dob, "phone": v.phone, "email": v.email, "id": v.id_digits,
             "policy": v.policy_number}
    return {kind: value.strip() for kind, value in given.items() if (value or "").strip()}


def _values(v: StatedValues, s: StatedIdentity, name: NameShape | None) -> dict[str, tuple[str, str]]:
    """Each well-formed value of this message: kind -> (as kept, normalised key). A name's key ignores word order (the
    database's reading of it) and a date's key lists its readings, so saying the same value again is the same value.
    An email and an ID number are kept masked: only their keys are ever compared."""
    out: dict[str, tuple[str, str]] = {}
    if name is not None:
        out["name"] = (v.policyholder_name.strip(), name.key)
    if s.dobs:
        readings = sorted(d.isoformat() for d in s.dobs)
        out["dob"] = ((v.date_of_birth_as_stated or "").strip() or ", ".join(readings), ",".join(readings))
    if s.phone:
        out["phone"] = (v.phone.strip(), s.phone)
    if s.email:
        out["email"] = (mask_email(s.email), s.email)
    if s.id_last4:
        out["id"] = (mask_ids(v.id_digits.strip()), s.id_last4)
    if s.policy:
        out["policy"] = (v.policy_number.strip(), s.policy)
    return out


def held_identity(held: dict[str, HeldValue]) -> StatedIdentity:
    """Everything held so far, latest per kind, ready for the database to compare. Only what is held is ever sent."""
    def key(kind: str) -> str | None:
        return held[kind].key if kind in held else None
    return StatedIdentity(
        name=held["name"].as_stated if "name" in held else None,
        dobs=tuple(date.fromisoformat(d) for d in held["dob"].key.split(",")) if "dob" in held else (),
        phone=key("phone"), email=key("email"), id_last4=key("id"), policy=key("policy"))


def _still_asked(ident: IdentityMemory) -> list[str]:
    """The kinds not given yet that may still be asked for: none the caller said they would rather not give."""
    return [k for k in KINDS if k not in ident.held and k not in ident.refused]


def _listed(kinds: list[str]) -> str:
    return ", ".join(LABEL[k] for k in kinds[:-1]) + (" or " if len(kinds) > 1 else "") + LABEL[kinds[-1]]


# How a reply may name each kind, and the number of details that must match: whenever details are asked for, the reply
# must say the rule and name every kind the caller can still give (checked by code, reply.check.required_mentions), so
# the caller can choose and give several at once rather than be asked for one at a time.
MENTION = {"name": ("full name", "your name"), "dob": ("date of birth", "birth date", "birthdate"),
           "phone": ("phone",), "email": ("email", "e-mail"), "id": ("last four", "last 4")}
THE_RULE = (str(KINDS_NEEDED), {2: "two", 3: "three", 4: "four"}.get(KINDS_NEEDED, str(KINDS_NEEDED)))
# A reply in a turn that kept no detail thanks the caller for none.
NO_THANKS = ("thank", "thanks")


def _options(kinds: list[str]) -> tuple[tuple[str, ...], ...]:
    return (THE_RULE, *(MENTION[k] for k in kinds))


def _kept_fact(kept: list[str]) -> tuple[Fact, ...]:
    """The kinds of detail kept from this message (the caller's own words; never whether they matched)."""
    return (Fact("detail_kept", "Kept from this message: " + ", ".join(LABEL[k] for k in kept)),) if kept else ()


def _ask(ident: IdentityMemory, kept: list[str] | None = None, *, lost: bool = False) -> ReplyPart:
    """The rule and every kind the caller can still give, as options. Thanks only when this message's detail was
    kept (`kept`: its kinds); when the message gave a detail and none of it was kept (`lost`), no thanks at all."""
    kept = kept or []
    missing = _still_asked(ident)
    facts = (RULE, Fact("kinds_not_given", f"Details not given yet ({len(missing)}): "
                        + (", ".join(LABEL[k] for k in missing) or "none")), *_kept_fact(kept))
    rules = dict(must_mention=_options(missing), must_not_mention=NO_THANKS if lost and not kept else ())
    thanks = "Thank you. " if kept else ""
    if not ident.held:
        if not missing:  # every kind refused: the rule is said, and a person is offered beside it
            return ReplyPart("ask_identity", facts, f"To verify who you are, {KINDS_NEEDED} details must match the "
                             "policy.", **rules)
        return ReplyPart("ask_identity", facts,
                         f"{thanks}To verify who you are, I need {KINDS_NEEDED} of these details to match the policy: "
                         f"{_listed(missing)}. Which can you give me? You can give several at once.", **rules)
    if not missing:
        return ReplyPart("ask_another_detail", facts,
                         f"{thanks}To verify who you are, {KINDS_NEEDED} details must match the policy. Could you check "
                         "the details you gave and say them again?", **rules)
    return ReplyPart("ask_another_detail", facts,
                     f"{thanks}To verify who you are, {KINDS_NEEDED} details must match the policy. Could you give any "
                     f"of these: {_listed(missing)}? You can give several at once.", **rules)


def _not_read(said: StatedValues) -> ReplyPart:
    """The fast model and the extractor disagree about whether the message gave a detail, so nothing from it is kept:
    the caller is told plainly that it could not be read and asked again, by kind when the extractor found one (never
    the value), and is not thanked for it."""
    kinds = [k for k in (*KINDS, "policy") if k in _given(said)]
    if kinds:
        return ReplyPart("restate_value", tuple(Fact("detail_to_restate", AGAIN.get(k, "your policy number"))
                                                for k in kinds),
                         f"Sorry, I couldn't read {_and(AGAIN.get(k, 'your policy number') for k in kinds)}. "
                         "Could you give it again?")
    return ReplyPart("unreadable_detail", plain="Sorry, I couldn't read the detail in that message. Could you give it "
                     "again?")


def _and(items) -> str:
    items = list(items)
    return ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]


def _reachable(ident: IdentityMemory) -> bool:
    """Whether the caller can still give enough kinds of detail without the ones they would rather not give. Read from
    what the caller said, never from what matched."""
    return len(set(ident.held) & set(KINDS)) + len(_still_asked(ident)) >= KINDS_NEEDED


def _refusal(ident: IdentityMemory, refused_now: list[str]) -> list[ReplyPart]:
    """A caller who would rather not give a detail: thanked, not asked for it again, told which other details would
    do and, plainly, that nothing about a claim can be shared without three that match. A person is offered as a
    yes-or-no question once three can no longer be reached; until then the reply says a person can help instead."""
    others = _still_asked(ident)
    facts = [RULE, Fact("without_three", f"Without {KINDS_NEEDED} details that match the policy, nothing about a "
                                         "claim can be shared.")]
    plain = ["Thank you for telling me."]
    if refused_now:
        facts.insert(0, Fact("not_asked_again", "The caller would rather not give, and will not be asked again for: "
                             + ", ".join(LABEL[k] for k in refused_now)))
        plain.append("I won't ask for that again.")
    if others:
        facts.append(Fact("other_details", "Any of these would do instead: " + ", ".join(LABEL[k] for k in others)))
        plain.append(f"Any of these would do instead: {_listed(others)}.")
    plain.append(f"Without {KINDS_NEEDED} details that match the policy, I can't share anything about a claim.")
    reachable = _reachable(ident)
    if reachable:
        facts.append(Fact("person_instead", f"If the caller cannot or would rather not give {KINDS_NEEDED}, a person "
                                            "can help instead."))
        plain.append(f"If you can't or would rather not give {KINDS_NEEDED}, I can pass you to a person instead.")
    out = [ReplyPart("acknowledge_refusal", tuple(facts), " ".join(plain), must_mention=_options(others))]
    if not reachable:
        out.append(ReplyPart("offer_person", plain="Would you like me to pass you to a person?", pending=PERSON_OFFER))
    return out


# Asked of a caller acting for someone who has not given their own name: the representative list is matched on it.
CALLER_NAME = Pending(kind="caller_name", label="your own full name", yes_no=False)
ASK_CALLER_NAME = ReplyPart(
    "ask_caller_name", (Fact("why_own_name", "The people who may act for a policyholder are listed by their own "
                                             "names, so the caller's own full name is needed."),),
    "The people who can act for a policyholder are listed by name. Could you tell me your own full name?",
    pending=CALLER_NAME)


async def step(ctx: TurnContext) -> StepOutcome:
    t, limits = ctx.deps.thresholds, ctx.deps.settings.limits
    form, said, conn, cid = ctx.form, ctx.stated, ctx.conn, ctx.conversation.id
    ident = ctx.memory.section("identity", State.VERIFY_ID)
    parts: list[ReplyPart] = []
    flags: list[str] = []

    # A caller who is done is not asked for anything more: the every-message rules either closed the conversation or,
    # for an upset caller, offered a person first, and that offer is the whole reply.
    if form.done is not None and confident_yes(form.done, t):
        return StepOutcome(State.VERIFY_ID, fired=[{"rule": "verify_id", "caller_done": True}])

    role = picked(form.caller_role, t) if form.caller_role is not None else None
    # A caller told they are not listed for a policyholder who now says they are that policyholder goes to a person,
    # as a representative waiting for consent does: who is calling is then a question for a person, not for details.
    if role == "policyholder" and ident.refused_as_representative:
        return StepOutcome(State.VERIFY_ID, handover=("rule", "role_switch"),
                           fired=[{"rule": "verify_id", "role_switch": True}])

    # The answer to "your own full name" is the caller's own name, wherever the extractor put it (unless the caller now
    # says they are the policyholder, whose own name is the policyholder's).
    if (ctx.memory.pending is not None and ctx.memory.pending.kind == CALLER_NAME.kind and role != "policyholder"
            and not (said.caller_name or "").strip() and (said.policyholder_name or "").strip()):
        said = said.model_copy(update={"caller_name": said.policyholder_name, "policyholder_name": None})

    # 1. The cross-check: the fast model's "gave a detail" and the extractor's values must agree.
    says_detail = form.gave_identity_detail is not None and yes(form.gave_identity_detail, t)
    counted = says_detail == _gave_a_value(said)
    if not counted:
        flags.append("cross_check")

    # 2 and 3. Format checks, then merge, latest value per kind. A value that fails its format check is asked for
    # again and is never held, so it can never count as a wrong value. A name's shape is the database's reading.
    failed_kinds: set[str] = set()
    partial_kinds: set[str] = set()  # kinds treated as only part of a detail (the inspector shows them)
    kept: list[str] = []
    if counted:
        stated_identity, failures = normalise(said)
        failed_kinds = {f.kind for f in failures}
        partial = form.partial_detail is not None and confident_yes(form.partial_detail, t)
        name_said = (said.policyholder_name or "").strip()
        name = await name_shape(conn, name_said) if name_said else None
        # A one-word name given again, after the same word was asked for in full, is the caller's whole name (a
        # customer whose name on file is one word): held, compared, and counted when wrong like any other value.
        insisted = (name is not None and name.shape == "single" and ident.part_of_name is not None
                    and ident.part_of_name.key == name.key)
        name_counts = name is not None and (name.shape == "whole" or insisted)
        if name is not None and not name_counts:
            failed_kinds.add("name")
            # A part of a detail is told by its shape (some digits, a year, one word of a name), whatever the model
            # read; part of a name after a whole name is held is not a new detail at all.
            if not ("name" in ident.held and ident.held["name"].shape == "whole"):
                failures.append(FormatFailure("name", name_said, "only part of a name", partial=True))
                ident.part_of_name = HeldValue(as_stated=name_said, key=name.key, turn=ctx.memory.counters.turn_no)
        in_part = [f for f in failures if f.partial or partial]
        partial_kinds = {f.kind for f in in_part}
        if in_part:
            parts.append(ReplyPart("ask_full_value",
                                   tuple(Fact("detail_to_complete", AGAIN[f.kind]) for f in in_part),
                                   "It sounds like that was only part of a detail. Could you give the whole of it?"))
        parts += [ReplyPart("restate_value", (Fact("detail_to_restate", AGAIN[f.kind]),),
                            f"Sorry, I couldn't use that. Could you give {AGAIN[f.kind]} again?")
                  for f in failures if f not in in_part]
        # Every value that passed its format check is kept, whatever the model thought of the message as a whole:
        # "SSN last four is 4472" is a complete detail even though it reads like part of one.
        correcting = form.correcting_detail is not None and yes(form.correcting_detail, t)
        values = _values(said, stated_identity, name if name_counts else None)
        kept = [k for k in KINDS if k in values]
        for kind, (as_kept, key) in values.items():
            if correcting and kind in ident.held and ident.held[kind].turn < ctx.memory.counters.turn_no:
                await flag_corrected(conn, cid, ident.held[kind].turn, kind)  # the earlier reading was wrong
            ident.held[kind] = HeldValue(as_stated=as_kept, key=key, turn=ctx.memory.counters.turn_no,
                                         shape=name.shape if kind == "name" and name is not None else None)
        if name_counts:  # a name held this turn ends the wait for the whole name
            ident.part_of_name = None
        if (said.caller_name or "").strip():
            ident.caller_name = said.caller_name.strip()
    else:
        parts.append(_not_read(said))
    for kind, value in _given(said).items():
        flags_here = [name for name, on in (("format_failed", kind in failed_kinds), ("cross_check_failed", not counted),
                                                ("partial", kind in partial_kinds))
                      if on]
        ctx.record("extraction", kind=kind, value=value, message=ctx.message, flags=flags_here)
    if role in ("policyholder", "third_party"):
        ident.caller_role = role
    # How the caller says they are related is kept as a note; it never decides whether they are listed (the
    # relationship rule in db/004_functions.sql, request_consent).
    relationship = picked(form.relationship, t) if form.relationship is not None else None
    if relationship not in (None, "not_stated"):
        ident.relationship = relationship
    # Refusals: kinds only, never asked for again in this conversation. A kind already refused is not refused anew:
    # the refusal was acknowledged when it was made, and a later message (or a reading carried over from it) does not
    # turn the reply back into that acknowledgement.
    refused_now = [k for k in KINDS if k in form.refuses and yes(form.refuses[k], t) and k not in ident.refused]
    ident.refused += [k for k in refused_now if k not in ident.refused]
    refusing = bool(refused_now) or yes(form.declines, t)

    # 4. Ask the database what matches, and judge it. The database counts each distinct wrong value once, itself.
    # THE INVARIANT: every value is either asked for again and never sent to the database (a part of a detail, a
    # malformed one: never held), or held, sent, and counted by the database when it does not match
    # (db/012_identity_lockout.sql). No value is compared without being counted, so no detail can be guessed without
    # limit, and a conversation at its limit is refused by the database whatever this step remembers.
    asked = held_identity(ident.held)
    try:
        matches = await match_identity(conn, cid, asked)
    except IdentityLocked:
        return StepOutcome(State.VERIFY_ID, handover=("rule", "wrong_values"),
                           fired=[{"rule": "verify_id", "refused_at_limit": True}], flags=flags)
    decision = judge(frozenset(ident.held), matches)
    ident.tally = {m.party_id: sorted(m.kinds) for m in matches}
    ident.wrong_values = await wrong_values(conn, cid)  # the database's count; memory keeps a copy to show
    fired = [{"rule": "verify_id", "kinds_held": sorted(ident.held), "wrong_values": len(ident.wrong_values)}]

    # A caller acting for someone is matched against the policyholder's list by their own name, so it is asked for
    # before anything is checked against that list; the reply is the same whatever the details matched.
    third_party = ident.caller_role == "third_party"
    own_name = await name_shape(conn, ident.caller_name) if third_party and ident.caller_name else None
    needs_own_name = third_party and (own_name is None or own_name.shape != "whole")

    # 5 and 6. Three kinds for one customer: the database decides, again, before anything is opened.
    if decision.verified_party is not None and not needs_own_name:
        party = decision.verified_party
        if third_party:
            result = await request_consent(conn, cid, party, asked, ident.caller_name)
            fired.append({"rule": "request_consent", "result": result})
            if result == "requested":
                return StepOutcome(State.AWAIT_CONSENT, parts, hop=True, fired=fired, flags=flags)
            if result == "not_listed":
                ident.refused_as_representative = True
                parts += [ReplyPart("not_listed", plain="The name you gave isn't listed as someone who can act for "
                                                        "this policyholder, so I can't go further here."),
                          ReplyPart("offer_person", plain="Would you like me to pass you to a person?",
                                    pending=PERSON_OFFER)]
                return StepOutcome(State.VERIFY_ID, parts, fired=fired, flags=flags)
        elif await bind_policyholder(conn, cid, party, asked):
            fired.append({"rule": "bind_policyholder"})
            parts.append(ReplyPart("verified", plain="Thank you, you're verified."))  # RESOLVE_INTENT asks
            return StepOutcome(State.RESOLVE_INTENT, parts, hop=True, fired=fired, flags=flags)
        else:
            # The judgment here and the database disagree: a bug or a tampered input. The database wins; record it.
            fired.append({"rule": "bind_policyholder", "result": False})
            logger.warning("the database refused a verification the judgment allowed (turn %s)",
                           ctx.memory.counters.turn_no)

    # 7. Not verified.
    if decision.all_kinds_no_match:
        return StepOutcome(State.VERIFY_ID, parts, handover=("rule", "all_kinds_no_match"), fired=fired, flags=flags)
    if len(ident.wrong_values) >= limits.wrong_values:
        return StepOutcome(State.VERIFY_ID, parts, handover=("rule", "wrong_values"), fired=fired, flags=flags)
    if needs_own_name:
        parts.append(ASK_CALLER_NAME)
        if len(set(ident.held) & set(KINDS)) >= KINDS_NEEDED:
            return StepOutcome(State.VERIFY_ID, parts, fired=fired, flags=flags)
    if refusing:
        fired.append({"rule": "refusal", "kinds": refused_now})
        parts += _refusal(ident, refused_now)
        return StepOutcome(State.VERIFY_ID, parts, fired=fired, flags=flags)
    parts.append(_ask(ident, kept, lost=says_detail or _gave_a_value(said)))
    if ident.refused and not _reachable(ident):
        parts.append(ReplyPart("offer_person", plain="Would you like me to pass you to a person?", pending=PERSON_OFFER))
    return StepOutcome(State.VERIFY_ID, parts, fired=fired, flags=flags)


STEPS[State.VERIFY_ID] = step


def details_asked_for(ident: IdentityMemory, pending: Pending | None) -> tuple[str, ...]:
    """The details the last reply asked for, by label: the caller's own full name when that was the question, and the
    kinds still asked for (every reply that asks lists them all); when every kind has been given, the ones the caller
    was asked to say again. The extractor reads an unlabelled value as the one of these it fits."""
    own = ("the caller's own full name",) if pending is not None and pending.kind == CALLER_NAME.kind else ()
    kinds = _still_asked(ident) or [k for k in KINDS if k not in ident.refused]
    return (*own, *(LABEL[k] for k in kinds))
