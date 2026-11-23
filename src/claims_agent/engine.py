"""One whole turn, in one database transaction: read the message, fill the form, run the rules, word and check the
reply, record it.

The order of the steps is fixed. The engine owns what no state owns: the turn number, the identity reset, the
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

from claims_agent import prepared
from claims_agent.answers.facts import ANSWER_MEANS, ANSWERS, COMPARE_CLAIMS, available
from claims_agent.budget import TurnBudget
from claims_agent.db import conversations as store
from claims_agent.db.claims import (Claim, as_listed, customer_vocabulary, fetch_ids, filter_from_form,
                                    filter_to_dict)
from claims_agent.db.guidance import Guidance, load_guidance
from claims_agent.db.conversations import Conversation
from claims_agent.db.identity import clear_verified
from claims_agent.db.pool import AppDb
from claims_agent.db.actions import turn_db_actions
from claims_agent.examples import with_approved
from claims_agent.form.fields import Form, ListedClaim, StatedValues
from claims_agent.form.filler import Extractor, FormFiller
from claims_agent.form.questions import RECENT_WINDOW, FormRequest
from claims_agent.form.read import confident_yes, picked, yes
from claims_agent.form.settle import Settler
from claims_agent.form.asked import Reading, form_asked
from claims_agent.inspect import inspector_payload, mask_ids
from claims_agent.loader.read import FALLBACK_TOPIC
from claims_agent.memory import Memory
from claims_agent.models.usage import collecting, turn_usage
from claims_agent.observability import NoLog, TurnLog, needs_review, safe_finish, safe_start
from claims_agent.models.errors import (ModelAccountProblem, ModelError, ModelKeyRejected, ModelNotPermitted,
                                        ModelRequestRejected, ModelUnavailable)
from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.compose import compose
from claims_agent.records import Recorder
from claims_agent.reply.wording import Wording
from claims_agent.rules import other_person
from claims_agent.rules.every_message import PERSON_OFFER, GlobalOutcome, asks_something, global_outcome, usable
from claims_agent.rules.handover import handover_note, note
from claims_agent.rules.mood import caller_mood, upset_unmoved
from claims_agent.rules.progress import apply_stall, made_progress, moved
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
# The fired entry of a reply whose claim facts were kept from the model; later turns read it back.
KEPT_FROM_MODEL = {"rule": "freedom", "kept_from_model": "claim_facts"}

# The answer kinds a caller can ask for by name, offered to the form beside the data's follow-up topics: the answers
# module's own list, less "not_on_file" (an outcome, never something asked for).
ANSWER_KINDS = tuple(a for a in ANSWERS if a not in ("not_on_file", COMPARE_CLAIMS))

# Fixed wording for parts that carry none of their own (the every-message rules build parts without text).
# Plain text is built only from facts and fixed wording, and holds no number or date.
PLAIN = {
    "ask_again": "Sorry, I couldn't read that. Could you say it again?",
    "offer_person": "Would you like me to pass you to a person?",
    "decline_off_topic": "I can only help with claims questions here.",
    "unsupported_offer_person": "I can't help with that request here. Would you like me to pass you to a person?",
    "cannot_read_out_personal_detail": "I can't read out personal details held on file, but I'm happy to help another way.",
    "offer_person_in_language": "You can ask for a person at any time.",
    "anything_else": "Is there anything else I can help you with?",
    "ask_need": "What can I help you with today?",
    "handover": "I'm passing you to a person, who will help you from here.",
    "closed": "This conversation has ended. Thank you for getting in touch.",
    "goodbye": "Thank you for getting in touch. Goodbye.",
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
    turn_log: TurnLog = field(default_factory=NoLog)
    settler: Settler | None = None  # the larger model, for unsure picks
    examples: dict[str, tuple[str, ...]] = field(default_factory=dict)  # reviewed example phrasings, read at start


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
    request: FormRequest | None = None  # this turn's form request, as the fast model was asked it
    budget: TurnBudget = field(default_factory=TurnBudget)  # the turn's own: every step shares it
    recorder: Recorder | None = None  # the turn's event records
    settled: dict | None = None  # how this reading's unsure answers were settled, by question name (the inspector's)

    def record(self, kind: str, /, **body) -> None:
        """Keep one event for this turn (nothing happens in a context built without a recorder)."""
        if self.recorder is not None:
            self.recorder.add(kind, **body)


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
    span_id: str | None = None


Step = Callable[[TurnContext], Awaitable[StepOutcome]]


# Each state's rules: rules/<state>.py sets STEPS[State.X] = step on import (the imports at the end of this module).
STEPS: dict[State, Step] = {}


# ---- the turn ---------------------------------------------------------------------------------------------------

async def run_turn(deps: Deps, conversation_id: UUID, message: str) -> TurnResult:
    """One turn. Its trace is sent only after the turn's transaction has committed, and never changes the outcome."""
    result = await _run_turn(deps, conversation_id, message)
    safe_finish(deps.turn_log, result.span_id, message=message, reply=result.reply, record=result.inspector,
                tags=(("needs_review",) if needs_review(result.inspector, deps.thresholds) else ()))
    return result


async def _run_turn(deps: Deps, conversation_id: UUID, message: str) -> TurnResult:
    with collecting():  # the model calls of this turn (and of its hand-over, if it fails) report their token usage
        return await _run_turn_counted(deps, conversation_id, message)


async def _run_turn_counted(deps: Deps, conversation_id: UUID, message: str) -> TurnResult:
    async with deps.db.conversation(conversation_id) as conn:
        conv, memory = await store.load_for_turn(conn, conversation_id)  # the row stays locked until commit
        memory.counters.turn_no += 1
        start_conv, start_memory = replace(conv), memory.model_copy(deep=True)
        span_id = safe_start(deps.turn_log, conversation_id, memory.counters.turn_no)
        try:
            async with conn.transaction():  # a savepoint: a failure below undoes everything after step 1
                return await _turn(deps, conn, conv, memory, message, span_id)
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
            return await _failed(deps, conn, start_conv, start_memory, message, reason, span_id)
        except Exception:
            logger.exception("the turn's hand-over could not be recorded; the whole turn is rolled back")
        # Raised outside the handler, so it carries no context: the transaction rolls back as it leaves this block.
        raise TurnFailed()


class _Turn:
    """What one turn has gathered so far."""

    def __init__(self, deps: Deps, conn: AsyncConnection, conv: Conversation, memory: Memory, message: str,
                 span_id: str | None = None):
        self.deps, self.conn, self.conv, self.memory = deps, conn, conv, memory
        self.span_id = span_id
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
        self.request: FormRequest | None = None
        self.budget = TurnBudget()
        self.part_progress = False  # a part of a message split into questions moved things forward
        lim = deps.settings.limits
        self.recorder = Recorder(conv.id, memory.counters.turn_no, body_chars=lim.event_body_chars,
                                 max_events=lim.events_per_turn, floor=memory.counters.context_floor)
        self.mood_change: tuple[str, str] | None = None  # (before, after) when the caller's clear mood changed
        self.focus_at_start = list(memory.focus.claim_ids)  # prepared replies are discarded when this changes
        self.asked: list[Reading] = []  # every form filled this turn, with its request, for the inspector
        self.path: list[str] = []  # the state of each step that ran, in order


async def _turn(deps: Deps, conn: AsyncConnection, conv: Conversation, memory: Memory, message: str,
                span_id: str | None = None) -> TurnResult:
    t, s = deps.thresholds, deps.settings
    turn = _Turn(deps, conn, conv, memory, message, span_id)
    state_before = conv.state
    before = memory.model_copy(deep=True)
    pending_at_start = memory.pending

    # 2. Finished conversations get one fixed line and no model call.
    if conv.state in FINAL or (conv.state == State.DONE and conv.verified_party_id is None):
        notice = FINAL_NOTICE.get(conv.state, FINAL_NOTICE[State.CLOSED])
        turn.parts = [ReplyPart("final_notice", plain=notice)]
        return await _finish(turn, state_before, fixed=True)
    # A verified caller writing after the wrap-up: the message is read as in a reopened conversation, and only a real
    # new question reopens it (see _finished_again).
    came_back = conv.state == State.DONE
    if came_back:
        conv.state = State.RESOLVE_INTENT

    # 3. Unreadable: ask again, a stalled turn. Asking again re-asks a pending question that is not yes/no, so it
    # stays for the next reply (a yes/no offer stays by its own rule). After the end, nothing is asked: goodbye.
    if not usable(message, pending_at_start):
        if came_back:
            _goodbye_again(turn, [])
        else:
            again = pending_at_start if pending_at_start is not None and not pending_at_start.yes_no else None
            turn.parts = [ReplyPart("ask_again", plain=PLAIN["ask_again"], pending=again)]
    else:
        await _read_and_act(turn, came_back)

    # 9. Progress and the stall count (not for a conversation that has ended and stays ended: thanking is not
    # stalling, and nothing is offered after a goodbye).
    pick = picked(turn.form.pending_reply, t) if turn.form is not None else None
    if conv.state not in FINAL and not (came_back and conv.state == State.DONE):
        reasons = made_progress(before, memory, state_before, conv.state, turn.grounded, pending_reply=pick)
        stall = apply_stall(memory.counters, moved(reasons) or turn.part_progress, state_before, conv.state,
                            s.limits.stalled_turns)
        memory.counters.stalled_in_a_row = stall.stalled_in_a_row
        # A turn whose only change is a remembered hint did not move (rules.progress.moved), for the stall count and
        # for an upset caller, who is offered a person without waiting for the stall limit; the offer resets the
        # stall count like any other.
        upset = False
        if turn.form is not None:
            memory.counters.upset_unmoved, upset = upset_unmoved(
                memory.counters.upset_unmoved, memory.counters.mood_by_turn[-1], moved(reasons) or turn.part_progress,
                s.limits.upset_turns_before_offer)
        if (stall.offer_person or upset) and not any(_offers_person(p) for p in turn.parts):
            turn.parts.append(ReplyPart("offer_person", plain=PLAIN["offer_person"], pending=PERSON_OFFER))
            turn.fired.append({"rule": "stall_limit" if stall.offer_person else "frustrated_again"})
            memory.counters.stalled_in_a_row = 0
            memory.counters.upset_unmoved = 0
    _settle_pending(turn, pick)
    return await _finish(turn, state_before)


async def _read_and_act(turn: _Turn, came_back: bool = False) -> None:
    """Steps 4 to 8: the form, the hints, the every-message rules, then the state's own steps (for a message that asks
    several things, part by part)."""
    deps, conv, memory = turn.deps, turn.conv, turn.memory
    t = deps.thresholds
    verified = conv.verified_party_id is not None

    req = await _form_request(turn, verified)
    turn.request = req
    turn.form, turn.stated = await _fill(deps, req, turn.message, verified,
                                         acting_for=memory.identity.caller_role == "third_party",
                                         asked_for=_details_asked_for(conv, memory))
    turn.asked.append(Reading(req, turn.form))
    form = turn.form
    _save_hints(memory, form, turn.stated, t)
    mood = caller_mood(form.mood, t)
    last_clear = next((m for m in reversed(memory.counters.mood_by_turn) if m != "unclear"), None)
    if mood is not None and last_clear is not None and mood != last_clear:
        turn.mood_change = (last_clear, mood)
    memory.counters.mood_by_turn.append(mood or "unclear")

    outcome = global_outcome(form, verified=verified, pending=memory.pending, t=t,
                             **await other_person.switch_reading(turn.conn, memory, form, turn.stated, t))
    language = [p for p in outcome.parts if p.instruction == LANGUAGE]
    hop = False
    if came_back:
        if _finished_again(turn, outcome):
            turn.parts += language
            return
        memory.wrapup.offers_made = 0
        memory.wrapup.unclear_replies = 0
        turn.fired.append({"rule": "reopened"})
        if outcome.terminal is None and _left_the_script(turn) == EMAIL_ONLY:
            _promise_email(turn)  # asked for after the end: the wrap-up decides whether it is sent (once)
            conv.state, hop = State.POST_PROCESS, True

    if outcome.terminal is not None or not await _answer_each_part(turn):
        await _act(turn, outcome, hop=hop)
        if outcome.terminal in OWNS_REPLY:
            return
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
        turn.path.append(state.value)
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


def answer_names(claims: Sequence[Claim], g: Guidance, s: Settings) -> tuple[str, ...]:
    """The answers the form offers with these claims in focus: the claim-record answers and the enabled follow-up topics
    that exist for them, and a comparison when there are several."""
    names = (*ANSWER_KINDS, *(f.topic for f in g.followups if f.enabled and f.topic != FALLBACK_TOPIC))
    return tuple(dict.fromkeys(a for c in claims for a in available(c, g, len(claims), s)
                               if a in names or a == COMPARE_CLAIMS))


def offered_documents(claims: Sequence[Claim], s: Settings) -> tuple[str, ...]:
    """The outstanding documents offered to the form as choices: each claim's own order, repeats once, at most
    `limits.documents_offered` (the API caps a choice's options). A document past the cap can still be named."""
    return tuple(dict.fromkeys(d for c in claims for d in c.documents))[: s.limits.documents_offered]


async def guidance(deps: Deps, conn: AsyncConnection) -> Guidance:
    """The guidance data: the copy loaded when the app started, else read through this turn's connection."""
    return deps.guidance if deps.guidance is not None else await load_guidance(conn)


def _context(turn: _Turn, entered_by_hop: bool) -> TurnContext:
    return TurnContext(turn.conn, turn.conv, turn.memory, turn.message, turn.form, turn.stated, turn.deps, turn.today,
                       entered_by_hop, request=turn.request, budget=turn.budget, recorder=turn.recorder,
                       settled=turn.asked[-1].settled if turn.asked else None)


async def _hand_over(turn: _Turn, handover: tuple[str, str]) -> None:
    kind, reason = handover
    turn.note = handover_note(_context(turn, False), reason)
    await store.add_escalation(turn.conn, turn.conv.id, turn.memory.counters.turn_no, kind, reason, turn.note)
    turn.recorder.add("escalation", why=reason, went_to="person", transcript=None)
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
    is cleared and one not answered stays; a question that is not yes/no lived for this one reply and is cleared. A
    conversation that has ended with nothing asked has nothing pending."""
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
    elif turn.conv.state == State.DONE:
        memory.pending = None
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
        fixed = True  # In this state no claim fact may enter a model call
        turn.fired.append(dict(KEPT_FROM_MODEL))
    if memory.focus.claim_ids != turn.focus_at_start:
        await prepared.discard(conn, conv.id)
    turn.parts = prepared.kept(turn.parts, turn.form, t, freedom, s.prepared_replies.enabled)

    form = turn.form
    rows = await _turns_since(conn, conv.id, memory.counters.context_floor)
    brief = ReplyBrief(
        state=conv.state, parts=tuple(turn.parts),
        mood=(caller_mood(form.mood, t) if form is not None else None) or "calm",
        caller_message=turn.message, previous_reply=_previous_reply(s, turn.freedom_state, rows),
        english=yes(form.english, t) if form is not None else True,
        verified=conv.verified_party_id is not None,
        caller_claim_numbers=(turn.stated.claim_number.strip(),) if (turn.stated.claim_number or "").strip() else ())
    composed = await compose(brief, deps.wording, s, await _vocabulary(conn, "status"), fixed=fixed,
                           budget=turn.budget)
    if composed.source == "last_resort" and conv.state not in FINAL:
        memory.pending = PERSON_OFFER  # the last-resort line offers a person: a "yes" must hand over
        memory.counters.stalled_in_a_row = 0  # Offering a person resets the stall count

    if turn.mood_change is not None:
        turn.recorder.add("mood_change", before=turn.mood_change[0], after=turn.mood_change[1], message=turn.message,
                          reply=composed.text)
    turn.recorder.current = (turn.message, composed.text)
    await turn.recorder.flush(conn)  # in the turn's own transaction: a turn rolled back leaves none of these

    payload = inspector_payload(
        state_before=state_before.value, state_after=conv.state.value,
        freedom=None if freedom is None else {"level": freedom_name, **freedom.model_dump()},
        filler=deps.filler.name, form=None if form is None else form.model_dump(mode="json"),
        stated=turn.stated.model_dump(mode="json"), flags=turn.flags, fired=turn.fired, parts=turn.parts,
        composed=composed, memory=memory.model_dump(mode="json"),
        counts={"stalled": memory.counters.stalled_in_a_row, "wrong_values": len(memory.identity.wrong_values),
                "consent_checks": memory.consent.checks_used},
        claims_visible=await _claims_visible(conn), outbox=await _outbox(conn, conv.id), handover_note=turn.note,
        events=turn.recorder.shown(), prepared=await prepared.on_hand(conn, conv.id), usage=turn_usage(),
        form_asked=form_asked(turn.asked, t, cap=s.limits.form_asked_chars), db_actions=turn_db_actions(),
        models=s.models.model_dump(include={"fast", "text", "larger"}), path=turn.path)
    turn_id = await store.append_turn(conn, conv.id, memory.counters.turn_no, turn.whole, composed.text,
                                      state_before.value, conv.state.value, payload, turn.span_id)
    await store.save(conn, conv, memory)
    return TurnResult(turn_id, composed.text, conv.state, payload, turn.span_id)


async def _failed(deps: Deps, conn: AsyncConnection, conv: Conversation, memory: Memory, message: str,
                  reason: str, span_id: str | None = None) -> TurnResult:
    """The turn was rolled back to where it started (after step 1); hand over with a fixed line, no model call."""
    turn = _Turn(deps, conn, conv, memory, message, span_id)
    state_before = conv.state
    turn.note = note(conv, memory, turn.message, reason)
    await store.add_escalation(conn, conv.id, memory.counters.turn_no, "failure", reason, turn.note)
    turn.recorder.add("escalation", why=reason, went_to="person", transcript=None)
    conv.state = State.ESCALATED
    memory.pending = None
    turn.parts = [ReplyPart("model_trouble", plain=PLAIN["model_trouble"])]
    turn.fired.append({"rule": "handover", "reason": reason})
    return await _finish(turn, state_before, fixed=True)


# ---- step 4 to 6: the form, the extractor, the hints ----------------------------------------------------------

async def _turns_since(conn: AsyncConnection, conversation_id: UUID,
                       floor: int) -> list[tuple[int, str, str, bool, bool]]:
    """Earlier turns a model may see again: none from before a different-person reset (numbered below the floor).
    Each row: turn number, caller message, reply, whether that reply was kept from the model, and whether it was made
    in a state that may send claim facts to a model (unknown counts as may)."""
    cur = await conn.execute(
        "SELECT turn_no, message, reply, coalesce(record->'fired' @> %s::jsonb, false), "
        "coalesce((record->'freedom'->>'claim_facts')::boolean, true) FROM turns "
        "WHERE conversation_id = %s AND turn_no >= %s ORDER BY turn_no, id",
        [json.dumps([KEPT_FROM_MODEL]), conversation_id, floor])
    return [(n, m or "", r or "", kept, facts) for n, m, r, kept, facts in await cur.fetchall()]


def _claim_facts_allowed(s: Settings, state: State | None) -> bool:
    setting = s.states.get(state.value) if state is not None else None
    return setting is None or s.freedom[setting.freedom].claim_facts


def _previous_reply(s: Settings, state: State | None, rows: list[tuple[int, str, str, bool, bool]]) -> str | None:
    """The agent's previous reply, for a model request. Left out when that reply was itself kept from the model (it
    may hold claim facts as plain text), and, in a state that may not send claim facts to a model, when that reply
    was made in a state that may. A reply made where claim facts may not go holds none (one that would have was kept
    from the model), so it is given: before verification the models then read "why do you need that?" against what
    was just asked, and the wording does not repeat itself."""
    if not rows or rows[-1][3]:
        return None
    if not _claim_facts_allowed(s, state) and rows[-1][4]:
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
    shown, read back under the database's row rules, so a claim no longer visible is left out); with no list, several
    claims in focus are that list, in the order they were told, so "the first one" can pick one of them. With claims in
    focus, the answers offered are those that exist for them and the documents are their outstanding ones, read the
    same way (never before verification, and only in a state that may send claim facts to a model)."""
    conn, memory, s = turn.conn, turn.memory, turn.deps.settings
    cap = s.limits.message_chars
    rows = await _turns_since(conn, turn.conv.id, memory.counters.context_floor)
    turn.recorder.recent_turns = [n for n, *_ in rows[-RECENT_WINDOW:]]  # what the form's per-message questions mean
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
        if not listed and len(in_focus) > 1:  # several claims being discussed, in the order told: "the first one"
            listed = tuple(as_listed(c) for c in in_focus)
        if in_focus:
            names = answer_names(in_focus, await guidance(turn.deps, conn), s)
            documents = offered_documents(in_focus, s)
    else:
        types, statuses = await _vocabulary(conn, "case_type"), await _vocabulary(conn, "status")
        years = tuple(range(turn.today.year - 4, turn.today.year + 1))
    return FormRequest(
        message=turn.message,
        recent_caller_messages=tuple(m[:cap] for _, m, *_ in rows[-RECENT_WINDOW:]),
        previous_reply=_previous_reply(turn.deps.settings, turn.conv.state, rows),
        pending=memory.pending.label if memory.pending is not None else None,
        state=turn.conv.state, verified=verified,
        claim_types=types, claim_statuses=statuses, years=years, listed_claims=listed,
        needs=await _vocabulary(conn, "need"), relationships=await _vocabulary(conn, "relationship"),
        answers=tuple((a, ANSWER_MEANS.get(a, a.replace("_", " "))) for a in names), documents=documents,
        examples=with_approved({f"answer_wanted:{topic}": ex for topic, ex in followups if ex}, turn.deps.examples),
        acting_for_policyholder=memory.identity.caller_role == "third_party",
        figure_meanings=dict((await guidance(turn.deps, conn)).glossary) if verified else {},
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


def _details_asked_for(conv: Conversation, memory: Memory) -> tuple[str, ...]:
    """What the agent's last reply asked the caller to identify themselves with, as the extractor is told it: only while
    details are being collected (VERIFY_ID's own reading of its memory)."""
    if conv.state != State.VERIFY_ID or conv.verified_party_id is not None:
        return ()
    from claims_agent.rules.verify_id import details_asked_for  # late: that module imports this one
    return details_asked_for(memory.identity, memory.pending)


async def _fill(deps: Deps, req: FormRequest, message: str, verified: bool, *,
                acting_for: bool = False, asked_for: tuple[str, ...] = ()) -> tuple[Form, StatedValues]:
    """The form, and the stated values: before verification the extractor runs alongside the filler; after it, only
    when `_stated_after_verification` says so."""
    if verified:
        form = await deps.filler.fill(req)
        return form, await _stated_after_verification(deps, form, message, acting_for=acting_for, split=True,
                                                      req=req)
    results = await asyncio.gather(deps.filler.fill(req),
                                   deps.extractor.extract(message, want_identity=True, want_claim_number=True,
                                                          previous_reply=req.previous_reply, just_asked=req.pending,
                                                          asked_for=asked_for),
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


async def _stated_after_verification(deps: Deps, form: Form, message: str, *, acting_for: bool,
                                     split: bool, req: FormRequest | None = None) -> StatedValues:
    """After verification the extractor runs only when it is needed: the caller quoted a claim number; a caller acting
    for the policyholder is read as asking about someone else's policy (whether they name a different policyholder
    decides whether that is a different person); or, when `split` (a whole message, never one of its parts), the
    message asks more than one thing, and the same call splits it into its questions. The one rule for both."""
    t = deps.thresholds
    identity = acting_for and confident_yes(form.other_person_policy, t)
    number = yes(form.quoted_claim_number, t)
    parts = split and yes(form.multi_question, t)
    if not (number or identity or parts):
        return StatedValues()
    # The same context as the form's request (the same previous reply, filtered the same way), as data only.
    return await deps.extractor.extract(message, want_identity=identity, want_claim_number=number, want_parts=parts,
                                        previous_reply=req.previous_reply if req is not None else None,
                                        just_asked=req.pending if req is not None else None)


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
    # An off-topic message asks nothing of a claim: a need read into it is not remembered (it would count as progress
    # and later decide the need for a claim the caller never asked about).
    if need not in (None, "none") and picked(form.about, t) != "off_topic":
        if hints.need != need:
            hints.need_turn = memory.counters.turn_no
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


# ---- acting on one reading: the ending outcomes, leaving the script, the state's steps ---------------------------
# A whole message and each part of a message that asks several things are acted on by the same functions, so the two
# can never handle an ending, an off-topic question or a request for the email differently.

LANGUAGE = "offer_person_in_language"  # the whole message's language line is added once, after everything else
# Ending outcomes after which nothing else is said or done this turn.
OWNS_REPLY = ("handover", "confirm_policyholder", "close")
AFTER_VERIFICATION = (State.RESOLVE_INTENT, State.PROCESS_CASE, State.POST_PROCESS)
OFF_TOPIC_ONLY, EMAIL_ONLY = "off_topic_only", "email_only"
EMAIL_PARTS = ("email_contents", "email_at_wrap_up", "offer_email", "email_sent", "email_skipped",
               "email_already_sent")


async def _apply_ending(turn: _Turn, outcome: GlobalOutcome) -> bool:
    """The every-message rules' ending outcome, applied the one way. For an outcome in OWNS_REPLY the reply is set
    here and nothing else may run. A different person resets: the database and memory forget together, in this
    transaction, and nothing said before it is kept. Returns whether the state's steps are entered by a hop (a
    wrap-up)."""
    conv, memory = turn.conv, turn.memory
    if outcome.terminal == "handover":
        await _hand_over(turn, ("caller", "asked_for_person"))
    elif outcome.terminal == "confirm_policyholder":  # asked before anything is told about anyone's claims
        turn.parts = list(outcome.parts)
    elif outcome.terminal == "close":
        conv.state = State.CLOSED
        turn.parts = [ReplyPart("closed", plain=PLAIN["closed"])]
    elif outcome.terminal == "reset_identity":
        await prepared.discard(turn.conn, conv.id)  # while the rows are still visible to this conversation
        await clear_verified(turn.conn, conv.id)
        memory.reset_for_new_identity()
        conv.state = State.VERIFY_ID
        await _refresh_parties(turn)
        turn.parts, turn.grounded = [], False
    elif outcome.terminal == "wrap_up":
        # Already wrapping up: a caller who is done is answering the email offer ("no thanks, that's all"), so the
        # step reads that answer; entering the wrap-up again would only make the offer again.
        if conv.state == State.POST_PROCESS:
            return False
        conv.state = State.POST_PROCESS
        return True
    return False


async def _act(turn: _Turn, outcome: GlobalOutcome, *, part: int | None = None, hop: bool = False) -> bool:
    """One reading (a whole message, or one part of one) acted on: its ending outcome; then its every-message parts;
    then either the state's steps or, when the message left the script after verification, the every-message reply
    alone and a way back to where the conversation was. True when nothing more may run this turn."""
    promised = _promise_email(turn)
    if outcome.terminal is not None:
        turn.fired.append({"rule": "every_message", "outcome": outcome.terminal, **({"part": part} if part else {})})
    hop = await _apply_ending(turn, outcome) or hop
    if outcome.terminal in OWNS_REPLY:
        return True
    turn.parts += [p for p in outcome.parts if p.instruction != LANGUAGE and p not in turn.parts]
    strayed = _left_the_script(turn) if outcome.terminal is None else None
    if strayed is not None:
        turn.fired.append({"rule": strayed})
        if strayed == EMAIL_ONLY:
            turn.parts.append(_email_later_part(turn, promised))
        turn.parts += await _back_to_pending(turn)
    else:
        await _run_steps(turn, hop)
        _acknowledge_email(turn, promised)
    conv = turn.conv
    return outcome.terminal is not None or conv.state in FINAL or conv.state == State.DONE


def _left_the_script(turn: _Turn) -> str | None:
    """After verification, a message that asks the claims line for nothing (see `asks_something`) is not given to the
    state's step, which would answer a question nobody asked about the claim ("not on file") or ask a second question:
    a request for the summary email (outside the wrap-up, and not the answer to the overview's own email question)
    is acknowledged, and an off-topic message gets the every-message decline; either way the caller is brought back
    to where the conversation was. The stall rule offers a person when such messages repeat."""
    conv, memory, form, t = turn.conv, turn.memory, turn.form, turn.deps.thresholds
    if conv.verified_party_id is None or conv.state not in AFTER_VERIFICATION:
        return None
    if asks_something(form, t, claim_number=turn.stated.claim_number, pending=memory.pending):
        return None
    from claims_agent.rules.resolve_intent import OVERVIEW  # late: that module imports this one
    overview = memory.pending is not None and memory.pending.kind == OVERVIEW.kind
    if yes(form.wants_email_summary, t) and conv.state != State.POST_PROCESS and not overview:
        return EMAIL_ONLY
    return OFF_TOPIC_ONLY if picked(form.about, t) == "off_topic" else None


def _promise_email(turn: _Turn) -> bool:
    """A verified caller asks for the summary email before the wrap-up, read at least as surely as a "send" must be
    (send_email_min): they are told it will be sent at the end, and the wrap-up sends it without asking again. A less
    sure reading is only remembered (the wrap-up offer says it was asked for). Returns whether it is promised."""
    conv, hints, t = turn.conv, turn.memory.hints, turn.deps.thresholds
    f = turn.form.wants_email_summary
    if conv.verified_party_id is not None and conv.state in (State.RESOLVE_INTENT, State.PROCESS_CASE) \
            and f.p is not None and f.p >= t.send_email_min:
        hints.wants_email = hints.email_promised = True
    return hints.email_promised


def _email_later_part(turn: _Turn, promised: bool) -> ReplyPart:
    from claims_agent.rules.resolve_intent import destination_fact  # late: that module imports this one
    where = destination_fact(turn.memory)
    if promised:
        said = f"I'll send a summary of what we discussed to {where.text} when we finish."
        fact = Fact("email_later", f"A summary of the conversation will be sent to {where.text} when it ends")
    else:
        said = "I'll offer to send you a summary when we finish."
        fact = Fact("email_later", "The caller will be offered a summary email when the conversation ends")
    return ReplyPart("email_at_wrap_up", (fact, where), said)


def _acknowledge_email(turn: _Turn, promised: bool) -> None:
    """A message that asked for the summary email and something else: once the state's step has answered, the
    request is acknowledged (unless a part already speaks of the email, or the turn reached the wrap-up), before the
    reply's closing question."""
    conv, form, t = turn.conv, turn.form, turn.deps.thresholds
    if conv.verified_party_id is None or conv.state not in (State.RESOLVE_INTENT, State.PROCESS_CASE) \
            or not yes(form.wants_email_summary, t) or any(p.instruction in EMAIL_PARTS for p in turn.parts):
        return
    at = next((i for i in range(len(turn.parts) - 1, -1, -1) if turn.parts[i].pending is not None), len(turn.parts))
    turn.parts.insert(at, _email_later_part(turn, promised))


async def _back_to_pending(turn: _Turn) -> list[ReplyPart]:
    """Back to where the conversation was: the question pending, asked again exactly as it was last asked (its own
    part, facts and all); with none, whether there is anything else. Nothing when the reply already asks something."""
    if any(p.pending is not None for p in turn.parts):
        return []
    pending = turn.memory.pending
    if pending is not None:
        row = await (await turn.conn.execute(
            "SELECT p FROM turns t, jsonb_array_elements(t.record->'parts') AS p WHERE t.conversation_id = %s "
            "AND t.turn_no >= %s AND p->'pending'->>'kind' = %s ORDER BY t.turn_no DESC, t.id DESC LIMIT 1",
            [turn.conv.id, turn.memory.counters.context_floor, pending.kind])).fetchone()
        if row is not None:
            p = row[0]
            return [ReplyPart(p["instruction"], tuple(Fact(**f) for f in p["facts"]), p["plain"], pending)]
    from claims_agent.rules.process_case import ANYTHING_ELSE  # late: that module imports this one
    return [ReplyPart("anything_else", plain=PLAIN["anything_else"], pending=ANYTHING_ELSE)]


def _finished_again(turn: _Turn, outcome: GlobalOutcome) -> bool:
    """A verified caller writes to a conversation that has ended. A message that asks nothing new (thanks, goodbye,
    "no, that's all", small talk, an off-topic question) leaves it ended: a short goodbye, nothing offered again. A
    request for a person or about a different person is acted on as in any state, and a real new question or a
    request for the summary reopens it."""
    form, t, memory = turn.form, turn.deps.thresholds, turn.memory
    if outcome.terminal not in (None, "wrap_up") or yes(form.wants_email_summary, t):
        return False
    if outcome.terminal is None and asks_something(form, t, claim_number=turn.stated.claim_number,
                                                   pending=memory.pending):
        return False
    _goodbye_again(turn, [p for p in outcome.parts if p.instruction != LANGUAGE])
    return True


def _goodbye_again(turn: _Turn, said: list[ReplyPart]) -> None:
    """The conversation stays ended: what the every-message rules said, then goodbye (unless they offer something, a
    person for instance, which is then the reply's question)."""
    turn.conv.state = State.DONE
    offers = any(p.pending is not None for p in said)
    turn.parts = said if offers else [*said, ReplyPart("goodbye", plain=PLAIN["goodbye"])]
    turn.fired.append({"rule": "after_the_end"})


# ---- a message that asks several things --------------------------------------------------------------------------
# Only after verification (before it, a mixed message is handled whole, so nothing is split out of an unverified
# message), a message the form says asks more than one question is split by the extractor into self-contained
# questions in the caller's own words. Up to limits.questions_per_message of them are each read with a form of their
# own and acted on like a whole message, in order, against the same memory, so a later part sees what an earlier one
# changed. All their reply parts make one reply: one wording call, one reply check. The questions beyond the limit are
# not processed; the reply asks the caller to send them next.
#
# What holds for a whole message holds for every part: the one transaction bound to the conversation; the
# memory-ownership check around every step and hop; at most one question in the reply (the last one asked, with an
# offer of a person outranking it, as the engine settles it); a model failure in any part rolls the whole turn back.
# The stall and unreadable counts move once per caller message: the engine does both, once, around this loop. An
# ending outcome in a part or a step that ends the conversation stops the parts left, and a different-person reset
# drops everything said before it.

MORE_LATER = ReplyPart("more_questions_later",
                       plain="I've answered your first questions here. Please send me the rest next, and I'll answer "
                             "them then.")


async def _answer_each_part(turn: _Turn) -> bool:
    """Run a split message part by part. False, with nothing done, when the message is handled whole: before
    verification, when the form does not say it asks more than one thing, or when fewer than two readable parts came
    back. (A whole message with an ending outcome is never split: the caller checks that first.)"""
    conv, t = turn.conv, turn.deps.thresholds
    if conv.verified_party_id is None or not yes(turn.form.multi_question, t):
        return False
    pieces = [q.strip() for q in turn.stated.question_parts if usable(q.strip(), None)]
    if len(pieces) < 2:
        return False

    most = turn.deps.settings.limits.questions_per_message
    whole = (turn.form, turn.stated, turn.message, turn.request)
    taken = pieces[:most]
    turn.fired.append({"rule": "question_parts", "parts": len(pieces), "taken": len(taken)})
    turn.parts = []  # each part's own reading replaces the whole message's every-message parts
    stopped = False
    for number, piece in enumerate(taken, 1):
        stopped = await _one_part(turn, piece, number)
        if stopped:
            break
    turn.form, turn.stated, turn.message, turn.request = whole
    if conv.state in FINAL:
        return True  # the step or hand-over that ended the conversation owns the whole reply
    turn.parts = _one_question(turn)
    if len(pieces) > most and not stopped:
        at = len(turn.parts) - 1 if turn.parts and turn.parts[-1].pending is not None else len(turn.parts)
        turn.parts.insert(at, MORE_LATER)  # before the reply's closing question, so that question stays last
        turn.fired.append({"rule": "more_questions_later", "left": len(pieces) - most})
    return True


async def _one_part(turn: _Turn, piece: str, number: int) -> bool:
    """One part: its own form (the same context, the part as the message), then acted on as a whole message is.
    True when the parts left must not run."""
    deps, conv, memory, t = turn.deps, turn.conv, turn.memory, turn.deps.thresholds
    before, state_before, grounded_before = memory.model_copy(deep=True), conv.state, turn.grounded
    turn.message = piece
    turn.request = await _form_request(turn, True)  # an unsure pick in this part is settled on this part's words
    form = await deps.filler.fill(turn.request)
    turn.asked.append(Reading(turn.request, form, part=number))
    stated = await _stated_after_verification(deps, form, piece, split=False,
                                              acting_for=memory.identity.caller_role == "third_party",
                                              req=turn.request)
    turn.form, turn.stated, turn.grounded = form, stated, False

    outcome = global_outcome(form, verified=True, pending=memory.pending, t=t,
                             **await other_person.switch_reading(turn.conn, memory, form, stated, t))
    stopped = await _act(turn, outcome, part=number)
    if outcome.terminal in OWNS_REPLY:
        return True
    if outcome.terminal == "reset_identity":
        grounded_before = False  # nothing said before a different-person reset counts

    pick = picked(form.pending_reply, t)
    if made_progress(before, memory, state_before, conv.state, turn.grounded, pending_reply=pick):
        turn.part_progress = True
    turn.grounded = grounded_before or turn.grounded
    if outcome.terminal is None and memory.pending is not None and memory.pending.yes_no and pick in ("yes", "no"):
        memory.pending = None  # answered by this part: no later part may answer it again
    return stopped


def _one_question(turn: _Turn) -> list[ReplyPart]:
    """At most one question besides an offer of a person: the last one asked. (The engine then lets an offer of a
    person outrank it, as for any reply.)"""
    asking = [p for p in turn.parts if p.pending is not None and p.pending.kind != PERSON_OFFER.kind]
    if len(asking) < 2:
        return turn.parts
    last = asking[-1]
    turn.fired += [{"rule": "question_dropped", "kind": p.pending.kind} for p in asking[:-1]]
    return [p for p in turn.parts if p.pending is None or p.pending.kind == PERSON_OFFER.kind or p is last]


import claims_agent.rules.verify_id  # noqa: E402,F401  registers VERIFY_ID's step in STEPS
import claims_agent.rules.await_consent  # noqa: E402,F401  registers AWAIT_CONSENT's step in STEPS
import claims_agent.rules.resolve_intent  # noqa: E402,F401  registers RESOLVE_INTENT's step in STEPS
import claims_agent.rules.process_case  # noqa: E402,F401  registers PROCESS_CASE's step in STEPS
import claims_agent.rules.post_process  # noqa: E402,F401  registers POST_PROCESS's step in STEPS
