"""One whole turn, in one database transaction: read the message, fill the form, run the rules, word and check the
reply, record it.

The order is pinned (plan Task 11). The engine owns what no state owns: the turn number, the identity reset, the
pending question, the stall count and every hand-over. Each state's step runs between a memory snapshot and an
ownership check, so a step can only change its own section of memory.

Failures have four outcomes. A rejected key, a model the key may not use, or an account with no credit is for the
operator: everything is rolled back and the error is raised. A request the API refused as malformed, any other model
failure, and any other error roll the turn back to where it started and hand over, each with its own reason; nothing
reaches the caller raw. If even that hand-over cannot be recorded, everything is rolled back and one `TurnFailed` is
raised, carrying no caller text and no data.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from uuid import UUID

from psycopg import AsyncConnection

from claims_agent.answers.facts import ANSWERS, COMPARE_CLAIMS, available
from claims_agent.db import conversations as store
from claims_agent.db.claims import (Claim, as_listed, customer_vocabulary, fetch_ids, filter_from_form,
                                    filter_to_dict)
from claims_agent.db.guidance import Guidance, load_guidance
from claims_agent.db.conversations import Conversation
from claims_agent.db.identity import clear_verified
from claims_agent.db.pool import AppDb
from claims_agent.form.fields import Form, ListedClaim, StatedValues
from claims_agent.form.filler import Extractor, FormFiller
from claims_agent.form.questions import RECENT_WINDOW, FormRequest
from claims_agent.form.read import picked, yes
from claims_agent.inspect import inspector_payload, mask_ids
from claims_agent.loader.read import FALLBACK_TOPIC
from claims_agent.memory import Memory
from claims_agent.models.errors import (ModelAccountProblem, ModelError, ModelKeyRejected, ModelNotPermitted,
                                        ModelRequestRejected, ModelUnavailable)
from claims_agent.reply.brief import ReplyBrief, ReplyPart
from claims_agent.reply.compose import compose
from claims_agent.reply.wording import Wording
from claims_agent.rules.every_message import PERSON_OFFER, global_outcome, usable
from claims_agent.rules.handover import handover_note, note
from claims_agent.rules.progress import apply_stall, made_progress
from claims_agent.settings import Settings, Thresholds
from claims_agent.states import State

__all__ = ["Deps", "TurnContext", "StepOutcome", "TurnResult", "TurnFailed", "STEPS", "run_turn", "handover_note",
           "mask_ids", "inspector_payload"]

logger = logging.getLogger("claims_agent.engine")

# Failures the operator must fix (the key, the models it may use, the account's credit): they repeat on every turn,
# so nothing changes, nobody is handed over, and the page says what is wrong.
RERAISE: tuple[type[Exception], ...] = (ModelKeyRejected, ModelNotPermitted, ModelAccountProblem)

MAX_HOPS = 4
FINAL = (State.ESCALATED, State.CLOSED)
# The fired entry of a reply whose claim facts were kept from the model (Ruling 7); later turns read it back.
KEPT_FROM_MODEL = {"rule": "freedom", "kept_from_model": "claim_facts"}

# The answer kinds a caller can ask for by name, offered to the form beside the data's follow-up topics: the answers
# module's own list, less "not_on_file" (an outcome, never something asked for).
ANSWER_KINDS = tuple(a for a in ANSWERS if a not in ("not_on_file", COMPARE_CLAIMS))

# Fixed wording for parts that carry none of their own (the every-message rules build parts without text).
# Ruling: plain text is built only from facts and fixed wording, and holds no number or date.
PLAIN = {
    "ask_again": "Sorry, I couldn't read that. Could you say it again?",
    "offer_person": "Would you like me to pass you to a person?",
    "explain_step": "Each step is there to protect your information, so it can't be skipped.",
    "decline_off_topic": "I can only help with claims questions here. Is there a claim I can help you with?",
    "unsupported_offer_person": "I can't help with that request here. Would you like me to pass you to a person?",
    "cannot_read_out_personal_detail": "I can't read out personal details held on file, but I'm happy to help another way.",
    "offer_person_in_language": "You can ask for a person at any time.",
    "anything_else": "Is there anything else I can help you with?",
    "ask_need": "What can I help you with today?",
    "handover": "I'm passing you to a person, who will help you from here.",
    "closed": "This conversation has ended. Thank you for getting in touch.",
    "model_trouble": "Sorry, something went wrong on our side. I'm passing you to a person, who will help you from here.",
}
FINAL_NOTICE = {
    State.ESCALATED: "This conversation has been passed to a person, who will pick it up from here. "
                     "Please start a new conversation if you need anything else.",
    State.CLOSED: "This conversation has ended. Please start a new conversation if you need anything else.",
}


@dataclass
class Deps:
    settings: Settings
    thresholds: Thresholds
    db: AppDb
    guidance: Guidance | None
    filler: FormFiller
    extractor: Extractor
    wording: Wording


@dataclass
class TurnContext:
    conn: AsyncConnection
    conversation: Conversation
    memory: Memory
    message: str  # as the models see it (cut to limits.message_chars)
    form: Form
    stated: StatedValues
    deps: Deps
    today: date
    entered_by_hop: bool


@dataclass
class StepOutcome:
    next_state: State
    parts: list[ReplyPart] = field(default_factory=list)
    hop: bool = False
    grounded: bool = False
    handover: tuple[str, str] | None = None  # (kind, reason): hand over; kind is "caller", "rule" or "failure"
    fired: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)  # shown in the inspector ("cross_check")


class TurnFailed(Exception):
    """A turn failed and even its hand-over could not be recorded; nothing of the turn was saved. Carries no caller
    text and no data: the cause is in the log."""

    def __init__(self) -> None:
        super().__init__("the turn could not be completed and nothing was saved")


@dataclass(frozen=True)
class TurnResult:
    turn_id: int
    reply: str
    state: State
    inspector: dict


Step = Callable[[TurnContext], Awaitable[StepOutcome]]


async def _not_built(ctx: TurnContext) -> StepOutcome:
    """A state whose rules are a later task: ask whether there is anything else, and stay."""
    return StepOutcome(ctx.conversation.state, [ReplyPart("anything_else", plain=PLAIN["anything_else"])])


# Each state's rules; later tasks register theirs here (rules/<state>.py sets STEPS[State.X] = step on import).
STEPS: dict[State, Step] = {s: _not_built for s in (State.VERIFY_ID, State.AWAIT_CONSENT, State.RESOLVE_INTENT,
                                                     State.PROCESS_CASE, State.POST_PROCESS)}


# ---- the turn ---------------------------------------------------------------------------------------------------

async def run_turn(deps: Deps, conversation_id: UUID, message: str) -> TurnResult:
    async with deps.db.conversation(conversation_id) as conn:
        conv, memory = await store.load_for_turn(conn, conversation_id)  # the row stays locked until commit
        memory.counters.turn_no += 1
        start_conv, start_memory = replace(conv), memory.model_copy(deep=True)
        try:
            async with conn.transaction():  # a savepoint: a failure below undoes everything after step 1
                return await _turn(deps, conn, conv, memory, message)
        except RERAISE:
            raise
        except ModelRequestRejected:  # before ModelUnavailable: it is one, but it will repeat until the code is fixed
            logger.error("turn handed over: the model API refused our request as malformed", exc_info=True)
            reason = "request_rejected"
        except ModelUnavailable:
            logger.warning("turn handed over: a model call failed after its retry")
            reason = "model_failure"
        except Exception:
            logger.exception("turn handed over: unexpected error inside the turn")
            reason = "internal_error"
        try:
            return await _failed(deps, conn, start_conv, start_memory, message, reason)
        except Exception:
            logger.exception("the turn's hand-over could not be recorded; the whole turn is rolled back")
        # Raised outside the handler, so it carries no context: the transaction rolls back as it leaves this block.
        raise TurnFailed()


class _Turn:
    """What one turn has gathered so far."""

    def __init__(self, deps: Deps, conn: AsyncConnection, conv: Conversation, memory: Memory, message: str):
        self.deps, self.conn, self.conv, self.memory = deps, conn, conv, memory
        self.whole = message
        cap = deps.settings.limits.message_chars
        self.message = message[:cap]
        self.flags: list[str] = ["message_cut"] if len(message) > cap else []
        self.fired: list[dict] = []
        self.parts: list[ReplyPart] = []
        self.form: Form | None = None
        self.stated = StatedValues()
        self.grounded = False
        self.note: dict | None = None
        self.freedom_state: State | None = conv.state if conv.state.value in deps.settings.states else None
        self.today = conv.today or deps.settings.today.date


async def _turn(deps: Deps, conn: AsyncConnection, conv: Conversation, memory: Memory, message: str) -> TurnResult:
    t, s = deps.thresholds, deps.settings
    turn = _Turn(deps, conn, conv, memory, message)
    state_before = conv.state
    before = memory.model_copy(deep=True)
    pending_at_start = memory.pending

    # 2. Finished conversations get one fixed line and no model call.
    if conv.state in FINAL or (conv.state == State.DONE and conv.verified_party_id is None):
        notice = FINAL_NOTICE.get(conv.state, FINAL_NOTICE[State.CLOSED])
        turn.parts = [ReplyPart("final_notice", plain=notice)]
        return await _finish(turn, state_before, fixed=True)
    if conv.state == State.DONE:  # W8: a verified caller who comes back after wrapping up
        conv.state = State.RESOLVE_INTENT
        memory.wrapup.offers_made = 0
        memory.wrapup.unclear_replies = 0
        turn.fired.append({"rule": "reopened"})

    # 3. Unreadable: ask again, a stalled turn. Asking again re-asks a pending question that is not yes/no, so it
    # stays for the next reply (a yes/no offer stays by its own rule).
    if not usable(message, pending_at_start):
        again = pending_at_start if pending_at_start is not None and not pending_at_start.yes_no else None
        turn.parts = [ReplyPart("ask_again", plain=PLAIN["ask_again"], pending=again)]
    else:
        await _read_and_act(turn)

    # 9. Progress and the stall count.
    pick = picked(turn.form.pending_reply, t) if turn.form is not None else None
    if conv.state not in FINAL:
        reasons = made_progress(before, memory, state_before, conv.state, turn.grounded, pending_reply=pick)
        stall = apply_stall(memory.counters, bool(reasons), state_before, conv.state, s.limits.stalled_turns)
        memory.counters.stalled_in_a_row = stall.stalled_in_a_row
        if stall.offer_person and not any(_offers_person(p) for p in turn.parts):
            turn.parts.append(ReplyPart("offer_person", plain=PLAIN["offer_person"], pending=PERSON_OFFER))
            turn.fired.append({"rule": "stall_limit"})
    _settle_pending(turn, pick)
    return await _finish(turn, state_before)


async def _read_and_act(turn: _Turn) -> None:
    """Steps 4 to 8: the form, the hints, the every-message rules, then the state's own steps."""
    deps, conn, conv, memory = turn.deps, turn.conn, turn.conv, turn.memory
    t = deps.thresholds
    verified = conv.verified_party_id is not None

    req = await _form_request(turn, verified)
    turn.form, turn.stated = await _fill(deps, req, turn.message, verified)
    form = turn.form
    _save_hints(memory, form, turn.stated, t)
    mood = picked(form.mood, t)
    memory.counters.mood_by_turn.append(mood or "unclear")

    outcome = global_outcome(form, verified=verified, pending=memory.pending, t=t)
    language = [p for p in outcome.parts if p.instruction == "offer_person_in_language"]
    turn.parts = [p for p in outcome.parts if p.instruction != "offer_person_in_language"]
    hop = False
    if outcome.terminal is not None:
        turn.fired.append({"rule": "every_message", "outcome": outcome.terminal})
    if outcome.terminal == "handover":
        await _hand_over(turn, ("caller", "asked_for_person"))
        return
    if outcome.terminal == "close":
        conv.state = State.CLOSED
        turn.parts = [ReplyPart("closed", plain=PLAIN["closed"])]
        return
    if outcome.terminal == "reset_identity":  # Ruling 4: the database and memory forget together, in this transaction
        await clear_verified(conn, conv.id)
        memory.reset_for_new_identity()
        conv.state = State.VERIFY_ID
        await _refresh_parties(turn)
    elif outcome.terminal == "wrap_up":
        conv.state = State.POST_PROCESS
        hop = True

    await _run_steps(turn, hop)
    if conv.state not in FINAL:
        if not turn.parts and conv.state != State.DONE:
            # Backstop: no open turn ends unanswered. The open need question is RESOLVE_INTENT's to ask; this only
            # catches a state that said nothing (a turn that ended in DONE has said its last word).
            turn.parts.append(ReplyPart("ask_need", plain=PLAIN["ask_need"]))
            turn.fired.append({"rule": "nothing_said"})
        turn.parts += language


async def _run_steps(turn: _Turn, entered_by_hop: bool) -> None:
    conv, memory = turn.conv, turn.memory
    hops = 0
    while True:
        state = conv.state
        ctx = _context(turn, entered_by_hop)
        before = memory.snapshot()
        out = await STEPS[state](ctx)
        memory.assert_only_owned_changed(before, state)
        turn.freedom_state = state
        await _refresh_parties(turn)
        # A step that ends the conversation owns the whole reply: nothing else may be said, or offered, after it.
        turn.parts = list(out.parts) if out.next_state in FINAL else turn.parts + out.parts
        turn.fired += out.fired
        turn.flags += out.flags
        turn.grounded = turn.grounded or out.grounded
        if out.handover is not None:
            await _hand_over(turn, out.handover)
            return
        if out.next_state == State.ESCALATED:
            raise ValueError(f"{state.value} moved to ESCALATED without a hand-over reason")
        conv.state = out.next_state
        if not out.hop or out.next_state not in STEPS or hops == MAX_HOPS:
            return
        hops += 1
        entered_by_hop = True


def offered_documents(claims: Sequence[Claim], s: Settings) -> tuple[str, ...]:
    """The outstanding documents offered to the form as choices: each claim's own order, repeats once, at most
    `limits.documents_offered` (the API caps a choice's options). A document past the cap can still be named."""
    return tuple(dict.fromkeys(d for c in claims for d in c.documents))[: s.limits.documents_offered]


async def guidance(deps: Deps, conn: AsyncConnection) -> Guidance:
    """The guidance data: the copy loaded when the app started, else read through this turn's connection."""
    return deps.guidance if deps.guidance is not None else await load_guidance(conn)


def _context(turn: _Turn, entered_by_hop: bool) -> TurnContext:
    return TurnContext(turn.conn, turn.conv, turn.memory, turn.message, turn.form, turn.stated, turn.deps, turn.today,
                       entered_by_hop)


async def _hand_over(turn: _Turn, handover: tuple[str, str]) -> None:
    kind, reason = handover
    turn.note = handover_note(_context(turn, False), reason)
    await store.add_escalation(turn.conn, turn.conv.id, turn.memory.counters.turn_no, kind, reason, turn.note)
    turn.conv.state = State.ESCALATED
    turn.parts = [ReplyPart("handover", plain=PLAIN["handover"])]  # nothing else is said
    turn.fired.append({"rule": "handover", "reason": reason})


async def _refresh_parties(turn: _Turn) -> None:
    """Who the conversation is verified as is the database's answer, read back after anything that may change it."""
    row = await (await turn.conn.execute(
        "SELECT verified_party_id, pending_party_id FROM conversations WHERE id = %s", [turn.conv.id])).fetchone()
    turn.conv.verified_party_id, turn.conv.pending_party_id = row


def _offers_person(p: ReplyPart) -> bool:
    return p.pending is not None and p.pending.kind == PERSON_OFFER.kind


def _settle_pending(turn: _Turn, pick: str | None) -> None:
    """One yes/no offer per reply. An offer of a person wins over any other yes/no offer, which is dropped. Otherwise
    the last part that carries a question becomes pending. With no new question: a yes/no question answered yes or no
    is cleared and one not answered stays; a question that is not yes/no lived for this one reply and is cleared."""
    memory = turn.memory
    if turn.conv.state in FINAL:
        memory.pending = None
        return
    person = next((p for p in turn.parts if _offers_person(p)), None)
    if person is not None:
        kept = [p for p in turn.parts
                if p is person or p.pending is None or (not p.pending.yes_no and not _offers_person(p))]
        turn.fired += [{"rule": "offer_dropped", "kind": p.pending.kind} for p in turn.parts
                       if not any(p is k for k in kept)]
        turn.parts = kept
        memory.pending = person.pending
        return
    carrying = [p for p in turn.parts if p.pending is not None]
    if carrying:
        memory.pending = carrying[-1].pending
    elif memory.pending is not None and (not memory.pending.yes_no or pick in ("yes", "no")):
        memory.pending = None


async def _finish(turn: _Turn, state_before: State, *, fixed: bool = False) -> TurnResult:
    """Step 10: the reply brief, the checked reply, the turn record, the save."""
    deps, conn, conv, memory = turn.deps, turn.conn, turn.conv, turn.memory
    s, t = deps.settings, deps.thresholds
    turn.parts = [p if p.plain else replace(p, plain=PLAIN.get(p.instruction, "")) for p in turn.parts]
    freedom_name = s.states[turn.freedom_state.value].freedom if turn.freedom_state is not None else None
    freedom = s.freedom[freedom_name] if freedom_name is not None else None
    if freedom is not None and not freedom.claim_facts and any(f.claim for p in turn.parts for f in p.facts):
        fixed = True  # Ruling 7: in this state no claim fact may enter a model call
        turn.fired.append(dict(KEPT_FROM_MODEL))

    form = turn.form
    rows = await _turns_since(conn, conv.id, memory.counters.context_floor)
    brief = ReplyBrief(
        state=conv.state, parts=tuple(turn.parts),
        mood=(picked(form.mood, t) if form is not None else None) or "calm",
        caller_message=turn.message, previous_reply=_previous_reply(s, turn.freedom_state, rows),
        english=yes(form.english, t) if form is not None else True,
        verified=conv.verified_party_id is not None,
        caller_claim_numbers=(turn.stated.claim_number.strip(),) if (turn.stated.claim_number or "").strip() else ())
    composed = await compose(brief, deps.wording, s, await _vocabulary(conn, "status"), fixed=fixed)
    if composed.source == "last_resort" and conv.state not in FINAL:
        memory.pending = PERSON_OFFER  # the last-resort line offers a person: a "yes" must hand over
        memory.counters.stalled_in_a_row = 0  # Ruling 20: offering a person resets the stall count

    payload = inspector_payload(
        state_before=state_before.value, state_after=conv.state.value,
        freedom=None if freedom is None else {"level": freedom_name, **freedom.model_dump()},
        filler=deps.filler.name, form=None if form is None else form.model_dump(mode="json"),
        stated=turn.stated.model_dump(mode="json"), flags=turn.flags, fired=turn.fired, parts=turn.parts,
        composed=composed, memory=memory.model_dump(mode="json"),
        counts={"stalled": memory.counters.stalled_in_a_row, "wrong_values": len(memory.identity.wrong_values),
                "consent_checks": memory.consent.checks_used},
        claims_visible=await _claims_visible(conn), outbox=await _outbox(conn, conv.id), handover_note=turn.note)
    turn_id = await store.append_turn(conn, conv.id, memory.counters.turn_no, turn.whole, composed.text,
                                      state_before.value, conv.state.value, payload)
    await store.save(conn, conv, memory)
    return TurnResult(turn_id, composed.text, conv.state, payload)


async def _failed(deps: Deps, conn: AsyncConnection, conv: Conversation, memory: Memory, message: str,
                  reason: str) -> TurnResult:
    """The turn was rolled back to where it started (after step 1); hand over with a fixed line, no model call."""
    turn = _Turn(deps, conn, conv, memory, message)
    state_before = conv.state
    turn.note = note(conv, memory, turn.message, reason)
    await store.add_escalation(conn, conv.id, memory.counters.turn_no, "failure", reason, turn.note)
    conv.state = State.ESCALATED
    memory.pending = None
    turn.parts = [ReplyPart("model_trouble", plain=PLAIN["model_trouble"])]
    turn.fired.append({"rule": "handover", "reason": reason})
    return await _finish(turn, state_before, fixed=True)


# ---- step 4 to 6: the form, the extractor, the hints ----------------------------------------------------------

async def _turns_since(conn: AsyncConnection, conversation_id: UUID, floor: int) -> list[tuple[int, str, str, bool]]:
    """Earlier turns a model may see again: none from before a different-person reset (numbered below the floor).
    Each row: turn number, caller message, reply, and whether that reply was kept from the model (Ruling 7)."""
    cur = await conn.execute(
        "SELECT turn_no, message, reply, coalesce(record->'fired' @> %s::jsonb, false) FROM turns "
        "WHERE conversation_id = %s AND turn_no >= %s ORDER BY turn_no, id",
        [json.dumps([KEPT_FROM_MODEL]), conversation_id, floor])
    return [(n, m or "", r or "", kept) for n, m, r, kept in await cur.fetchall()]


def _claim_facts_allowed(s: Settings, state: State | None) -> bool:
    setting = s.states.get(state.value) if state is not None else None
    return setting is None or s.freedom[setting.freedom].claim_facts


def _previous_reply(s: Settings, state: State | None, rows: list[tuple[int, str, str, bool]]) -> str | None:
    """The agent's previous reply, for a model request: left out when this state may not send claim facts to a model,
    or when that reply was itself kept from the model (it may hold claim facts as plain text)."""
    if not rows or rows[-1][3] or not _claim_facts_allowed(s, state):
        return None
    return rows[-1][2]


async def _vocabulary(conn: AsyncConnection, kind: str) -> tuple[str, ...]:
    """The data-wide words of one kind (claim types, statuses, relationships, needs), as loaded."""
    cur = await conn.execute("SELECT DISTINCT value FROM vocabulary WHERE kind = %s ORDER BY value", [kind])
    return tuple(v for (v,) in await cur.fetchall())


async def _followups(conn: AsyncConnection) -> list[tuple[str, tuple[str, ...]]]:
    """The enabled follow-up topics in their order, with their example phrases."""
    cur = await conn.execute("SELECT key, body FROM guidance WHERE kind = 'followup'")
    rows = [(k, b) for k, b in await cur.fetchall() if k and k != FALLBACK_TOPIC and b.get("enabled")]
    rows.sort(key=lambda kb: kb[1].get("position", 0))
    return [(k, tuple(b.get("examples") or ())) for k, b in rows]


async def _form_request(turn: _Turn, verified: bool) -> FormRequest:
    """Before verification only vocabulary goes in: the data-wide types, statuses, relationships and needs, the
    answer kinds and follow-up topics, and the last five calendar years. After it, types, statuses and years are the
    caller's own claims', and a list of claims the caller was just shown goes in as the listed claims (in the order
    shown, read back under the database's row rules, so a claim no longer visible is left out). With claims in
    focus, the answers offered are those that exist for them and the documents are their outstanding ones, read the
    same way (never before verification, and only in a state that may send claim facts to a model)."""
    conn, memory, s = turn.conn, turn.memory, turn.deps.settings
    cap = s.limits.message_chars
    rows = await _turns_since(conn, turn.conv.id, memory.counters.context_floor)
    listed: tuple[ListedClaim, ...] = ()
    followups = await _followups(conn)
    topics = [topic for topic, _ in followups]
    names: tuple[str, ...] = (*ANSWER_KINDS, *topics)
    documents: tuple[str, ...] = ()
    if verified:
        types, statuses, years = await customer_vocabulary(conn)
        if memory.focus.last_list:
            listed = tuple(as_listed(c) for c in await fetch_ids(conn, memory.focus.last_list))
        in_focus = await fetch_ids(conn, memory.focus.claim_ids) \
            if memory.focus.claim_ids and _claim_facts_allowed(s, turn.conv.state) else []
        if in_focus:
            g = await guidance(turn.deps, conn)
            names = tuple(dict.fromkeys(a for c in in_focus for a in available(c, g, len(in_focus), s)
                                        if a in names or a == COMPARE_CLAIMS))
            documents = offered_documents(in_focus, s)
    else:
        types, statuses = await _vocabulary(conn, "case_type"), await _vocabulary(conn, "status")
        years = tuple(range(turn.today.year - 4, turn.today.year + 1))
    return FormRequest(
        message=turn.message,
        recent_caller_messages=tuple(m[:cap] for _, m, _, _ in rows[-RECENT_WINDOW:]),
        previous_reply=_previous_reply(turn.deps.settings, turn.conv.state, rows),
        pending=memory.pending.label if memory.pending is not None else None,
        state=turn.conv.state, verified=verified,
        claim_types=types, claim_statuses=statuses, years=years, listed_claims=listed,
        needs=await _vocabulary(conn, "need"), relationships=await _vocabulary(conn, "relationship"),
        answers=tuple((a, a.replace("_", " ")) for a in names), documents=documents,
        examples={f"answer_wanted:{topic}": ex for topic, ex in followups if ex},
    )


def _rank(e: BaseException) -> int:
    """Which failure decides the turn when the filler and the extractor both fail: an operator problem first, then our
    own bug (it must be recorded as internal_error, never as a model outage), then a rejected request, then an outage."""
    if isinstance(e, RERAISE):
        return 0
    if isinstance(e, ModelRequestRejected):
        return 2
    if isinstance(e, ModelUnavailable):
        return 3
    return 1


async def _fill(deps: Deps, req: FormRequest, message: str, verified: bool) -> tuple[Form, StatedValues]:
    """The form, and the stated values: before verification the extractor runs alongside the filler; after it, only
    when the caller quoted a claim number."""
    if verified:
        form = await deps.filler.fill(req)
        if yes(form.quoted_claim_number, deps.thresholds):
            return form, await deps.extractor.extract(message, want_identity=False, want_claim_number=True)
        return form, StatedValues()
    results = await asyncio.gather(deps.filler.fill(req),
                                   deps.extractor.extract(message, want_identity=True, want_claim_number=True),
                                   return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        chosen = min(errors, key=_rank)
        for e in errors:  # nothing is dropped unseen: our own bug keeps its traceback
            if e is chosen:
                continue
            if isinstance(e, ModelError):
                logger.warning("a second model failure in this turn was not the one acted on: %s", type(e).__name__)
            else:
                logger.error("an error in this turn was not the one acted on", exc_info=e)
        raise chosen
    return results[0], results[1]


def _save_hints(memory: Memory, form: Form, stated: StatedValues, t: Thresholds) -> None:
    """Step 6, in every state: the claim description, the need, and a wish for an email summary. The description is
    read by the claim filter's own reader and kept in its own shape, so a hint and the filter that later reads it can
    never disagree about what the caller said."""
    hints = memory.hints
    described = filter_from_form(form, t)
    if described.conditions():
        hints.claim_filter = filter_to_dict(described)
    number = (stated.claim_number or "").strip()
    if number:
        hints.claim_number = number
    need = picked(form.need, t)
    if need not in (None, "none"):
        hints.need = need
    if yes(form.wants_email_summary, t):
        hints.wants_email = True


# ---- the inspector's database reads ---------------------------------------------------------------------------

async def _claims_visible(conn: AsyncConnection) -> int:
    return (await (await conn.execute("SELECT count(*) FROM claims")).fetchone())[0]


async def _outbox(conn: AsyncConnection, conversation_id: UUID) -> list[dict]:
    cur = await conn.execute("SELECT id, subject, body, created_at FROM outbox WHERE conversation_id = %s ORDER BY id",
                             [conversation_id])
    return [{"id": i, "recipient": "the policyholder's email on file", "subject": sub, "body": body,
             "created_at": at.isoformat()} for i, sub, body, at in await cur.fetchall()]


import claims_agent.rules.verify_id  # noqa: E402,F401  registers VERIFY_ID's step in STEPS
import claims_agent.rules.await_consent  # noqa: E402,F401  registers AWAIT_CONSENT's step in STEPS
import claims_agent.rules.resolve_intent  # noqa: E402,F401  registers RESOLVE_INTENT's step in STEPS
import claims_agent.rules.process_case  # noqa: E402,F401  registers PROCESS_CASE's step in STEPS
import claims_agent.rules.post_process  # noqa: E402,F401  registers POST_PROCESS's step in STEPS
