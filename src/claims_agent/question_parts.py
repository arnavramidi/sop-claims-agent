"""Messages that ask more than one thing.

Only after verification (before it, a mixed message is handled whole, so nothing is split out of an unverified
message), a message the form says asks more than one question is split by the extractor into
self-contained questions in the caller's own words. Up to three of them are each read with a form of their own and run
through the every-message rules and the state's step, in order, against the same memory, so a later part sees what an
earlier one changed. All their reply parts make one reply: one wording call, one reply check. A fourth question and
beyond are not processed; the reply asks the caller to send them next.

What holds for a whole message holds for every part: the one transaction bound to the conversation; the
memory-ownership check around every step and hop (the engine's own step runner); at most one question in the reply
(the last one asked, with an offer of a person outranking it, as the engine settles it); a model failure in any part
rolls the whole turn back. The stall and unreadable counts move once per caller message: the engine does both, once,
around this loop. A terminal outcome in a part (a person, a different person, a wrap-up) or a step that ends the
conversation stops the parts left, and a different-person reset drops everything said before it.
"""

from claims_agent.db.identity import clear_verified
from claims_agent.engine import FINAL, _form_request, _hand_over, _refresh_parties, _run_steps, _Turn
from claims_agent.form.fields import StatedValues
from claims_agent.form.read import picked, yes
from claims_agent.reply.brief import ReplyPart
from claims_agent.rules.every_message import PERSON_OFFER, GlobalOutcome, global_outcome, usable
from claims_agent.rules.progress import made_progress
from claims_agent.states import State

MAX_PARTS = 3  # questions answered from one message; the rest wait for the next one
MORE_LATER = ReplyPart("more_questions_later",
                       plain="I've answered your first questions here. Please send me the rest next, and I'll answer "
                             "them then.")
LANGUAGE = "offer_person_in_language"  # the whole message's language line is the engine's; a part adds none


async def answer_each_part(turn: _Turn, outcome: GlobalOutcome) -> bool:
    """Run a split message part by part. False, with nothing done, when the message is handled whole: before
    verification, when the whole message already had a terminal outcome, when the form does not say it
    asks more than one thing, or when fewer than two readable parts came back."""
    conv, t = turn.conv, turn.deps.thresholds
    if outcome.terminal is not None or conv.verified_party_id is None or not yes(turn.form.multi_question, t):
        return False
    pieces = [q.strip() for q in turn.stated.question_parts if usable(q.strip(), None)]
    if len(pieces) < 2:
        return False

    whole = (turn.form, turn.stated, turn.message)
    taken = pieces[:MAX_PARTS]
    turn.fired.append({"rule": "question_parts", "parts": len(pieces), "taken": len(taken)})
    turn.parts = []  # each part's own reading replaces the whole message's every-message parts
    stopped = False
    for number, piece in enumerate(taken, 1):
        stopped = await _one_part(turn, piece, number)
        if stopped:
            break
    turn.form, turn.stated, turn.message = whole
    if conv.state in FINAL:
        return True  # the step or hand-over that ended the conversation owns the whole reply
    turn.parts = _one_question(turn)
    if len(pieces) > MAX_PARTS and not stopped:
        at = len(turn.parts) - 1 if turn.parts and turn.parts[-1].pending is not None else len(turn.parts)
        turn.parts.insert(at, MORE_LATER)  # before the reply's closing question, so that question stays last
        turn.fired.append({"rule": "more_questions_later", "left": len(pieces) - MAX_PARTS})
    return True


async def _one_part(turn: _Turn, piece: str, number: int) -> bool:
    """One part: its own form (the same context, the part as the message), the every-message rules, then the
    state's steps. True when the parts left must not run."""
    deps, conv, memory, t = turn.deps, turn.conv, turn.memory, turn.deps.thresholds
    before, state_before, grounded_before = memory.model_copy(deep=True), conv.state, turn.grounded
    turn.message = piece
    form = await deps.filler.fill(await _form_request(turn, True))
    stated = await deps.extractor.extract(piece, want_identity=False, want_claim_number=True) \
        if yes(form.quoted_claim_number, t) else StatedValues()
    turn.form, turn.stated, turn.grounded = form, stated, False

    outcome = global_outcome(form, verified=True, pending=memory.pending, t=t)
    hop = False
    if outcome.terminal is not None:
        turn.fired.append({"rule": "every_message", "outcome": outcome.terminal, "part": number})
    if outcome.terminal == "handover":
        await _hand_over(turn, ("caller", "asked_for_person"))
        return True
    if outcome.terminal == "reset_identity":  # as for a whole message; nothing said before it is used
        await clear_verified(turn.conn, conv.id)
        memory.reset_for_new_identity()
        conv.state = State.VERIFY_ID
        await _refresh_parties(turn)
        turn.parts, grounded_before = [], False
    elif outcome.terminal == "wrap_up":
        conv.state = State.POST_PROCESS
        hop = True
    # ("close" is only ever the outcome of an unverified message, and parts exist only after verification.)
    turn.parts += [p for p in outcome.parts if p.instruction != LANGUAGE and p not in turn.parts]
    await _run_steps(turn, hop)

    pick = picked(form.pending_reply, t)
    if made_progress(before, memory, state_before, conv.state, turn.grounded, pending_reply=pick):
        turn.part_progress = True
    turn.grounded = grounded_before or turn.grounded
    if outcome.terminal is None and memory.pending is not None and memory.pending.yes_no and pick in ("yes", "no"):
        memory.pending = None  # answered by this part: no later part may answer it again
    return outcome.terminal is not None or conv.state in FINAL or conv.state == State.DONE


def _one_question(turn: _Turn) -> list[ReplyPart]:
    """At most one question besides an offer of a person: the last one asked. (The engine then lets an offer of a
    person outrank it, as for any reply.)"""
    asking = [p for p in turn.parts if p.pending is not None and p.pending.kind != PERSON_OFFER.kind]
    if len(asking) < 2:
        return turn.parts
    last = asking[-1]
    turn.fired += [{"rule": "question_dropped", "kind": p.pending.kind} for p in asking[:-1]]
    return [p for p in turn.parts if p.pending is None or p.pending.kind == PERSON_OFFER.kind or p is last]
