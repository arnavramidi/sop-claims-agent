"""VERIFY_ID: collect identifying details across turns; only the database can mark the caller verified.

The comparison happens inside the database (match_identity) and so does the write (bind_policyholder,
request_consent, which re-check the three kinds themselves). This step only judges what came back, counts wrong values
and decides the reply. The reply is the same whatever matched: its facts are the verification rule and the kinds not
yet given, never how many matched, how many are still needed, or whether a person or policy exists.
"""

import hashlib
import logging
from datetime import date

from claims_agent.db.identity import bind_policyholder, match_identity, request_consent
from claims_agent.engine import STEPS, StepOutcome, TurnContext
from claims_agent.form.fields import StatedValues
from claims_agent.form.read import confident_yes, picked, yes
from claims_agent.identity.normalize import name_candidates, normalise
from claims_agent.identity.types import KINDS, StatedIdentity
from claims_agent.identity.verify import KINDS_NEEDED, judge
from claims_agent.memory import HeldValue, IdentityMemory
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.rules.every_message import PERSON_OFFER
from claims_agent.states import State

logger = logging.getLogger("claims_agent.rules.verify_id")

# How the reply names each kind. The ID is always "the ID on file", never its type.
LABEL = {"name": "full name", "dob": "date of birth", "phone": "phone number", "email": "email address",
         "id": "the ID on file"}
AGAIN = {"name": "your full name", "dob": "your date of birth", "phone": "your phone number",
         "email": "your email address", "id": "the digits of the ID on file"}

RULE = Fact("verification_rule",
            f"To verify who you are, {KINDS_NEEDED} of these details must match the policy: "
            + ", ".join(LABEL[k] for k in KINDS)
            + f". A policy number helps find the policy but is not one of the {KINDS_NEEDED}.")


def fingerprint(conversation_id, kind: str, key: str) -> str:
    """A counted wrong value as memory keeps it: the kind and a digest of the normalised value salted with the
    conversation id, so the count works while memory never holds what was typed."""
    digest = hashlib.sha256(f"{conversation_id}|{kind}|{key}".encode()).hexdigest()[:16]
    return f"{kind}:{digest}"


def _gave_a_value(v: StatedValues) -> bool:
    return any((x or "").strip() for x in (v.policyholder_name, v.phone, v.email, v.id_digits, v.policy_number,
                                           v.date_of_birth_as_stated)) or any(d.strip() for d in v.date_of_birth)


def _values(v: StatedValues, s: StatedIdentity) -> dict[str, tuple[str, str]]:
    """Each well-formed value of this message: kind -> (as stated, normalised key). A name's key ignores word order
    and a date's key lists its readings, so saying the same value again is the same value."""
    out: dict[str, tuple[str, str]] = {}
    if s.names:
        out["name"] = (v.policyholder_name.strip(), min(s.names))
    if s.dobs:
        readings = sorted(d.isoformat() for d in s.dobs)
        out["dob"] = ((v.date_of_birth_as_stated or "").strip() or ", ".join(readings), ",".join(readings))
    if s.phone:
        out["phone"] = (v.phone.strip(), s.phone)
    if s.email:
        out["email"] = (v.email.strip(), s.email)
    if s.id_last4:
        out["id"] = (v.id_digits.strip(), s.id_last4)
    if s.policy:
        out["policy"] = (v.policy_number.strip(), s.policy)
    return out


def held_identity(held: dict[str, HeldValue]) -> StatedIdentity:
    """Everything held so far, latest per kind, ready for the database to compare."""
    def key(kind: str) -> str | None:
        return held[kind].key if kind in held else None
    return StatedIdentity(
        names=name_candidates(held["name"].as_stated) if "name" in held else (),
        dobs=tuple(date.fromisoformat(d) for d in held["dob"].key.split(",")) if "dob" in held else (),
        phone=key("phone"), email=key("email"), id_last4=key("id"), policy=key("policy"))


async def _contradicting(conn) -> str:
    """A relationship that equals none the data holds: what "other" means to the database."""
    cur = await conn.execute("SELECT value FROM vocabulary WHERE kind = 'relationship'")
    stored = {v.strip().lower() for (v,) in await cur.fetchall()}
    value, n = "other", 1
    while value in stored:
        n += 1
        value = f"other {n}"
    return value


def _ask(ident: IdentityMemory) -> ReplyPart:
    missing = [k for k in KINDS if k not in ident.held]
    facts = (RULE, Fact("kinds_not_given", f"Details not given yet ({len(missing)}): "
                        + (", ".join(LABEL[k] for k in missing) or "none")))
    if not ident.held:
        return ReplyPart("ask_identity", facts,
                         f"To verify who you are, I need {KINDS_NEEDED} of these details to match the policy: "
                         + ", ".join(LABEL[k] for k in KINDS[:-1]) + f" or {LABEL[KINDS[-1]]}. What can you give me?")
    if not missing:
        return ReplyPart("ask_another_detail", facts,
                         "Thank you. Could you check the details you gave and say them again?")
    listed = ", ".join(LABEL[k] for k in missing[:-1]) + (" or " if len(missing) > 1 else "") + LABEL[missing[-1]]
    return ReplyPart("ask_another_detail", facts,
                     f"Thank you. To verify who you are, {KINDS_NEEDED} details must match the policy. "
                     f"Could you give one more of these: {listed}?")


async def step(ctx: TurnContext) -> StepOutcome:
    t, limits = ctx.deps.thresholds, ctx.deps.settings.limits
    form, said, conn, cid = ctx.form, ctx.stated, ctx.conn, ctx.conversation.id
    ident = ctx.memory.section("identity", State.VERIFY_ID)
    parts: list[ReplyPart] = []
    flags: list[str] = []

    # 1. The cross-check: the fast model's "gave a detail" and the extractor's values must agree.
    says_detail = form.gave_identity_detail is not None and yes(form.gave_identity_detail, t)
    counted = says_detail == _gave_a_value(said)
    if not counted:
        flags.append("cross_check")

    # 2 and 3. Format checks, then merge, latest value per kind. A value that fails its format check is asked for
    # again and is never held, so it can never count as a wrong value.
    if counted:
        stated_identity, failures = normalise(said)
        partial = form.partial_detail is not None and confident_yes(form.partial_detail, t)
        if partial and failures:
            parts.append(ReplyPart("ask_full_value",
                                   tuple(Fact("detail_to_complete", AGAIN[f.kind]) for f in failures),
                                   "It sounds like that was only part of a detail. Could you give the whole of it?"))
        elif failures:
            parts += [ReplyPart("restate_value", (Fact("detail_to_restate", AGAIN[f.kind]),),
                                f"Sorry, I couldn't use that. Could you give {AGAIN[f.kind]} again?")
                      for f in failures]
        # Every value that passed its format check is kept, whatever the model thought of the message as a whole:
        # "SSN last four is 4472" is a complete detail even though it reads like part of one.
        for kind, (as_stated, key) in _values(said, stated_identity).items():
            ident.held[kind] = HeldValue(as_stated=as_stated, key=key, turn=ctx.memory.counters.turn_no)
        if (said.caller_name or "").strip():
            ident.caller_name = said.caller_name.strip()
    role = picked(form.caller_role, t) if form.caller_role is not None else None
    if role in ("policyholder", "third_party"):
        ident.caller_role = role
    relationship = picked(form.relationship, t) if form.relationship is not None else None
    if relationship not in (None, "not_stated"):
        ident.relationship = relationship

    # 4. Ask the database what matches, judge it, count each distinct wrong value once.
    asked = held_identity(ident.held)
    matches = await match_identity(conn, asked)
    decision = judge(frozenset(ident.held), matches)
    ident.tally = {m.party_id: sorted(m.kinds) for m in matches}
    for kind in sorted(decision.wrong_now):
        mark = fingerprint(cid, kind, ident.held[kind].key)
        if mark not in ident.wrong_values:
            ident.wrong_values.append(mark)
    fired = [{"rule": "verify_id", "kinds_held": sorted(ident.held), "wrong_values": len(ident.wrong_values)}]

    # 5 and 6. Three kinds for one customer: the database decides, again, before anything is opened.
    if decision.verified_party is not None:
        party = decision.verified_party
        if ident.caller_role == "third_party":
            rel = None if ident.relationship in (None, "not_stated") else \
                await _contradicting(conn) if ident.relationship == "other" else ident.relationship
            result = await request_consent(conn, cid, party, asked, name_candidates(ident.caller_name or ""), rel)
            fired.append({"rule": "request_consent", "result": result})
            if result == "requested":
                return StepOutcome(State.AWAIT_CONSENT, parts, hop=True, fired=fired, flags=flags)
            if result == "not_listed":
                parts += [ReplyPart("not_listed", plain="The person named isn't listed as someone who can act for "
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
    parts.append(_ask(ident))
    return StepOutcome(State.VERIFY_ID, parts, fired=fired, flags=flags)


STEPS[State.VERIFY_ID] = step
