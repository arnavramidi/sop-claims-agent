"""Replies to the follow-ups a caller is most likely to ask next, worded ahead in the background, so that a common
follow-up is answered at once.

A prepared reply is exactly as trustworthy as a live one. Its facts are built by the same code from the claim's own
row and the guidance data; its wording passes the same reply check before it is stored; and when it is about to be
used, the facts are built again from the claim as it is now and the stored reply is used only if the claim's
last-changed stamp, the claims in focus and every fact are still the same. Anything else and the row is deleted and the
answer is worded live. A prepared reply is never used for an answer that must be worded live, for a caller who is not
calm, for a message that asks more than one thing or is not in English, in a state whose freedom level switches
prepared replies off, or when the form's pick is below `prepared_reply_min`.

The job runs after the response is sent, outside any turn: it reads the conversation without locking it, words the
replies with no database connection held, and writes its rows in a short transaction of its own, bound to the
conversation. The table's row rule refuses a row for a customer who is no longer the verified one, and the job drops
its rows when the focus or the state moved on meanwhile. A failure in the job is logged and dropped; it never reaches
a caller.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from uuid import UUID

import psycopg
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from claims_agent.answers.facts import AnswerResult
from claims_agent.answers.followups import fill, followup_candidates
from claims_agent.db.claims import Claim, fetch_ids
from claims_agent.db.guidance import load_guidance
from claims_agent.form.fields import Form
from claims_agent.form.read import picked, yes
from claims_agent.memory import Memory
from claims_agent.models.errors import ModelAccountProblem, ModelError, ModelKeyRejected, ModelNotPermitted
from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.rules.mood import caller_mood
from claims_agent.settings import Freedom, Settings, Thresholds
from claims_agent.states import State

if TYPE_CHECKING:
    from claims_agent.engine import Deps, StepOutcome, TurnContext

logger = logging.getLogger("claims_agent.prepared")

SOURCE = "prepared"
# Picks that name no answer: a "yes" to an offer is then the pick that chose the answer.
_NOT_AN_ANSWER = ("none", "something_else", "not_on_file")
# Problems the operator must fix (the key, the models it may use, the account's credit): logged once, never retried.
SETUP_ERRORS = (ModelKeyRejected, ModelNotPermitted, ModelAccountProblem)
_setup_logged: set[type] = set()


def focus_key(claim_ids: Sequence[str]) -> str:
    """The claims in focus, in order, as one text a prepared reply is stored with."""
    return json.dumps(list(claim_ids))


def _freedom(s: Settings) -> Freedom | None:
    setting = s.states.get(State.PROCESS_CASE.value)
    return s.freedom.get(setting.freedom) if setting is not None else None


def may_use_prepared(form: Form, t: Thresholds, freedom: Freedom | None, enabled: bool) -> bool:
    """Whether this message may be answered with a prepared reply: the switch is on, the state's freedom level allows
    it, the caller is calm, the message asks one thing in English, and the pick that chose the answer (the answer
    asked for, else the reply to the offer) is at least `prepared_reply_min`."""
    if not enabled or freedom is None or not freedom.prepared_replies:
        return False
    if caller_mood(form.mood, t) != "calm" or yes(form.multi_question, t) \
            or form.english is None or not yes(form.english, t):
        return False
    pick = form.answer_wanted if form.answer_wanted.value not in (None, *_NOT_AN_ANSWER) else form.pending_reply
    return picked(pick, t) is not None and pick.p is not None and pick.p >= t.prepared_reply_min


def _facts_json(facts: Sequence[Fact]) -> list[dict]:
    return [{"kind": f.kind, "text": f.text, "claim": f.claim} for f in facts]


def _single_answer(result: AnswerResult) -> ReplyPart | None:
    """The one part a prepared reply words: a follow-up whose facts are all told (nothing not on file, no person
    offered) and which need not be worded live. None for anything else."""
    from claims_agent.rules.process_case import _parts  # the step's own builder, so both build the same part
    if result.reword_live or result.offer_person:
        return None
    parts, _ = _parts([result])
    return parts[0] if len(parts) == 1 and parts[0].instruction == "answer" else None


def _brief(part: ReplyPart) -> ReplyBrief:
    return ReplyBrief(state=State.PROCESS_CASE, parts=(part,), mood="calm", caller_message="", previous_reply=None,
                      english=True, verified=True)


async def _statuses(conn: AsyncConnection) -> tuple[str, ...]:
    cur = await conn.execute("SELECT DISTINCT value FROM vocabulary WHERE kind = 'status' ORDER BY value")
    return tuple(v for (v,) in await cur.fetchall())


@dataclass(frozen=True)
class _Item:
    claim: Claim
    answer: str
    part: ReplyPart


@dataclass(frozen=True)
class _Plan:
    party_id: str
    key: str
    items: tuple[_Item, ...]
    statuses: tuple[str, ...]


async def _plan(deps: Deps, conversation_id: UUID) -> _Plan | None:
    """What to prepare, read in one short transaction without locking the conversation: up to `claims_prepared`
    claims in focus (in the focus's order), and for each the follow-ups the offer rule would offer next (linked to the
    need, existing for the claim, neither answered nor declined), taken one per claim in turn, at most
    `replies_prepared` in all. Nothing before verification or outside PROCESS_CASE."""
    from claims_agent.rules.resolve_intent import GENERAL
    s = deps.settings
    async with deps.db.conversation(conversation_id) as conn:
        row = await (await conn.execute(
            "SELECT state, verified_party_id, today, memory FROM conversations WHERE id = %s",
            [conversation_id])).fetchone()
        if row is None or row[1] is None or row[0] != State.PROCESS_CASE.value:
            return None
        party_id, today, memory = row[1], row[2] or s.today.date, Memory.model_validate(row[3])
        claims = await fetch_ids(conn, memory.focus.claim_ids[: s.limits.claims_prepared])
        g = deps.guidance if deps.guidance is not None else await load_guidance(conn)
        statuses = await _statuses(conn)
    need = memory.request.need or memory.focus.need or GENERAL
    queues: list[list[_Item]] = []
    for c in claims:
        done = [*memory.request.answered.get(c.case_id, []), *memory.request.declined.get(c.case_id, [])]
        items = []
        for f in followup_candidates(need, c, done, g, s):
            part = _single_answer(fill(f, c, g, s, today))
            if part is not None:
                items.append(_Item(c, f.topic, part))
        queues.append(items)
    chosen: list[_Item] = []
    while any(queues) and len(chosen) < s.limits.replies_prepared:
        for q in queues:
            if q and len(chosen) < s.limits.replies_prepared:
                chosen.append(q.pop(0))
    return _Plan(party_id, focus_key(memory.focus.claim_ids), tuple(chosen), statuses)


async def prepare_ahead(deps: Deps, conversation_id: UUID) -> int:
    """Word ahead the replies to the follow-ups the caller is most likely to ask next; the number of rows stored.
    Nothing unless the switch is on, the conversation is verified and in PROCESS_CASE, and that state's freedom level
    allows prepared replies. Never writes the conversation's memory."""
    s = deps.settings
    freedom = _freedom(s)
    if not s.prepared_replies.enabled or freedom is None or not freedom.prepared_replies:
        return 0
    plan = await _plan(deps, conversation_id)
    if plan is None or not plan.items:
        return 0
    rc = s.reply_check
    worded: list[tuple[_Item, str]] = []
    for item in plan.items:  # no database connection is held while a model words a reply
        brief = _brief(item.part)
        text = await deps.wording.word(brief)
        result = check_reply(text, brief, wary_status_words=rc.wary_status_words, data_statuses=plan.statuses,
                             status_forms=rc.status_forms, max_length=rc.max_length)
        if result.passed:
            worded.append((item, text))
    if not worded:
        return 0
    return await _store(deps, conversation_id, plan, worded)


async def _store(deps: Deps, conversation_id: UUID, plan: _Plan, worded: list[tuple[_Item, str]]) -> int:
    """The rows, in a short transaction of their own: dropped when the conversation moved on while they were worded
    (another state, another focus, another verified customer); the row rule refuses a customer who is no longer the
    verified one even if that changes after this check."""
    try:
        async with deps.db.conversation(conversation_id) as conn:
            row = await (await conn.execute(
                "SELECT state, verified_party_id, memory->'focus'->'claim_ids' FROM conversations WHERE id = %s",
                [conversation_id])).fetchone()
            if row is None or row[0] != State.PROCESS_CASE.value or row[1] != plan.party_id \
                    or focus_key(row[2] or []) != plan.key:
                logger.info("prepared replies dropped: the conversation moved on while they were worded")
                return 0
            for item, text in worded:
                await conn.execute(
                    "DELETE FROM prepared_replies WHERE conversation_id = %s AND case_id = %s AND answer = %s",
                    [conversation_id, item.claim.case_id, item.answer])
                await conn.execute(
                    "INSERT INTO prepared_replies (conversation_id, case_id, answer, text, claim_updated_at, focus_key, "
                    "party_id, facts) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [conversation_id, item.claim.case_id, item.answer, text, item.claim.updated_at, plan.key,
                     plan.party_id, Jsonb(_facts_json(item.part.facts))])
    except (psycopg.errors.InsufficientPrivilege, psycopg.errors.UniqueViolation):
        logger.info("prepared replies dropped: the database refused them (the conversation moved on)")
        return 0
    return len(worded)


async def take(conn: AsyncConnection, conversation_id: UUID, claim: Claim, answer: str, key: str, *,
               facts: Sequence[Fact]) -> str | None:
    """The prepared reply for this claim and answer, only when it was made from the claim as it is now (the same
    last-changed stamp), for the same claims in focus (`key`) and from exactly these freshly built facts. A row that
    does not match is deleted and None is returned."""
    row = await (await conn.execute(
        "SELECT text, claim_updated_at, focus_key, facts FROM prepared_replies "
        "WHERE conversation_id = %s AND case_id = %s AND answer = %s",
        [conversation_id, claim.case_id, answer])).fetchone()
    if row is None:
        return None
    text, stamp, stored_key, stored_facts = row
    if stamp == claim.updated_at and stored_key == key and stored_facts == _facts_json(facts):
        return text
    await conn.execute("DELETE FROM prepared_replies WHERE conversation_id = %s AND case_id = %s AND answer = %s",
                       [conversation_id, claim.case_id, answer])
    return None


async def discard(conn: AsyncConnection, conversation_id: UUID) -> None:
    """Every prepared reply of the conversation, gone (the claims in focus changed, or the caller is someone else)."""
    await conn.execute("DELETE FROM prepared_replies WHERE conversation_id = %s", [conversation_id])


async def on_hand(conn: AsyncConnection, conversation_id: UUID) -> list[dict]:
    """The prepared replies the conversation holds, as the inspector lists them (which claim and answer; no text)."""
    cur = await conn.execute("SELECT case_id, answer FROM prepared_replies WHERE conversation_id = %s "
                             "ORDER BY case_id, answer", [conversation_id])
    return [{"claim": c, "answer": a} for c, a in await cur.fetchall()]


# ---- using one in PROCESS_CASE --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    """One follow-up answered for one claim this turn: the claim, the answer, its result and the parts built for it
    (before anything else was added to the reply)."""
    claim: Claim
    answer: str
    result: AnswerResult
    parts: tuple[ReplyPart, ...]


async def serve(ctx: TurnContext, slot: Slot | None, out: StepOutcome) -> StepOutcome:
    """The step's outcome with the answer part carrying its prepared reply, when one may be used and still matches
    the facts just built; otherwise the outcome unchanged (the answer is worded live)."""
    if slot is None or len(slot.parts) != 1 or slot.parts[0].instruction != "answer" or slot.result.reword_live \
            or slot.result.offer_person:
        return out
    part = slot.parts[0]
    if not any(p is part for p in out.parts):
        return out
    s, t = ctx.deps.settings, ctx.deps.thresholds
    if not may_use_prepared(ctx.form, t, _freedom(s), s.prepared_replies.enabled):
        return out
    text = await take(ctx.conn, ctx.conversation.id, slot.claim, slot.answer, focus_key(ctx.memory.focus.claim_ids),
                      facts=part.facts)
    out.fired.append({"rule": "prepared_reply", "claim": slot.claim.case_id, "answer": slot.answer,
                      "used": text is not None})
    if text is None:
        return out
    return replace(out, parts=[replace(p, prepared=text) if p is part else p for p in out.parts])


def kept(parts: Sequence[ReplyPart], form: Form | None, t: Thresholds, freedom: Freedom | None,
         enabled: bool) -> list[ReplyPart]:
    """The reply's parts, with any prepared reply dropped unless the whole message may use one (a message split into
    several questions is read part by part, but it is not a single question)."""
    if form is not None and may_use_prepared(form, t, freedom, enabled) \
            and sum(1 for p in parts if p.prepared) <= 1:
        return list(parts)
    return [replace(p, prepared="") if p.prepared else p for p in parts]


# ---- the background job ---------------------------------------------------------------------------------------

async def run_in_background(deps: Deps, conversation_id: UUID) -> None:
    """`prepare_ahead`, after a response has been sent. Whatever goes wrong is logged and dropped: a problem the
    operator must fix is logged once, and nothing is retried."""
    try:
        await prepare_ahead(deps, conversation_id)
    except SETUP_ERRORS as e:
        if type(e) not in _setup_logged:
            _setup_logged.add(type(e))
            logger.error("replies were not prepared ahead: %s", type(e).__name__)
    except ModelError as e:
        logger.warning("replies were not prepared ahead: a model call failed (%s)", type(e).__name__)
    except Exception:
        logger.exception("replies were not prepared ahead: unexpected error")


def schedule(background, deps: Deps, conversation_id: UUID, state: State) -> None:
    """After a reply, prepare the next ones in the background (only with the switch on, in PROCESS_CASE)."""
    if deps.settings.prepared_replies.enabled and state == State.PROCESS_CASE:
        background.add_task(run_in_background, deps, conversation_id)
