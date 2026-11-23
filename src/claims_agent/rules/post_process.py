"""POST_PROCESS: ask whether there is anything else, offer the summary email, send it on a clear yes, and end.

The email's address is never shown or sent to a model: the database stores the summary for the policyholder's own
email on file and the reply names the place only by the phrase the destination fact carries. The summary's text comes
from facts built by code for the claims the caller discussed, read fresh through the turn's connection (so the
database's row rules decide what is visible).

The `wrapup` section is this state's. An offer counts as made only once it stands: the engine drops a yes/no offer
when a person is offered in the same reply, so the count goes up on a later turn in which the offer is still the
pending question, never on the turn that makes it.

The email is decided once: sent, or declined (said no, or stayed unclear for limits.unclear_email_replies replies).
A later wrap-up does not offer it again; only a new request for it does, and a summary already sent is sent again only
when something new was answered since. A caller who asked for the summary earlier and was told it would be sent at
the end (the `email_promised` hint) gets it without being asked again. Every way out of the step ends in one of: the
summary sent, a goodbye with nothing sent, "already sent", the offer (asked again), what the email holds, why the
address cannot be read out or changed, an answer followed by the offer, or a hop back to choose a claim or a need.
"""

from claims_agent.answers.facts import available
from claims_agent.answers.summary_email import write_summary
from claims_agent.db.claims import fetch, fetch_ids, filter_from_dict
from claims_agent.db.identity import send_summary
from claims_agent.engine import PLAIN, STEPS, StepOutcome, TurnContext, guidance
from claims_agent.form.read import picked, yes
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.rules.every_message import PERSON_OFFER
from claims_agent.rules.process_case import EMAIL_OFFER, NOT_AN_ANSWER, NOT_ON_FILE, SWITCHING, answer_parts
from claims_agent.rules.resolve_intent import destination_fact
from claims_agent.states import State

FULL_LIST_LIMIT = 500
REPLIES = ("send", "dont_send", "whats_in_it", "which_address", "different_address")
SETTLED = ("sent", "skipped")  # the email decisions that stand
GOODBYE = ReplyPart("goodbye", plain=PLAIN["goodbye"])


class _Wrapup:
    def __init__(self, ctx: TurnContext):
        self.ctx, self.t = ctx, ctx.deps.thresholds
        self.memory, self.form = ctx.memory, ctx.form
        self.wrap = ctx.memory.section("wrapup", State.POST_PROCESS)
        self.request = ctx.memory.request
        self.focus = ctx.memory.focus
        self.where = destination_fact(ctx.memory)
        self.hints = ctx.memory.hints
        self.unclear_limit = ctx.deps.settings.limits.unclear_email_replies
        self.fired: list[dict] = [{"rule": "post_process"}]

    def discussed(self) -> list[str]:
        return [case_id for case_id, answers in self.request.answered.items() if answers]

    def stay(self, *parts: ReplyPart, grounded: bool = False) -> StepOutcome:
        return StepOutcome(State.POST_PROCESS, list(parts), grounded=grounded, fired=self.fired)

    def done(self, *parts: ReplyPart) -> StepOutcome:
        return StepOutcome(State.DONE, list(parts), fired=self.fired)

    def offer(self, lead: str = "") -> ReplyPart:
        """The email offer: one yes/no question. It says where the summary would go only by the destination phrase."""
        facts = [self.where]
        text = f"Would you like a summary of what we discussed sent to {self.where.text}?"
        if self.memory.hints.wants_email:
            facts.append(Fact("requested_earlier", "The caller asked earlier for a summary by email."))
            text = f"You asked earlier for a summary by email. {text}"
        return ReplyPart("offer_email", tuple(facts), f"{lead} {text}".strip(), pending=EMAIL_OFFER)

    # ---- entry ---------------------------------------------------------------------------------------------------

    def settled(self) -> bool:
        """The email was sent or declined at an earlier wrap-up."""
        return self.wrap.outcome in SETTLED

    def new_answers(self) -> bool:
        """Something was answered since the email was last sent or declined."""
        def given(answered: dict[str, list[str]]) -> dict[str, set[str]]:
            return {k: set(v) for k, v in answered.items() if v}
        return given(self.request.answered) != given(self.wrap.answered_when_settled)

    def settle(self, outcome: str) -> None:
        """The email is decided: what was answered by now is kept, and the request for it is used up."""
        self.wrap.outcome = outcome
        self.wrap.answered_when_settled = {k: list(v) for k, v in self.request.answered.items()}
        self.hints.wants_email = self.hints.email_promised = False

    async def entry(self) -> StepOutcome:
        self.wrap.unclear_replies = 0
        self.wrap.claims_discussed = self.discussed()
        if not self.discussed() and self.focus.email_list_filter is None:
            if not self.settled():
                self.wrap.outcome = "nothing_to_send"
            self.fired.append({"rule": "nothing_discussed"})
            return self.done(GOODBYE)
        asked = self.hints.wants_email
        if self.settled() and not asked:
            self.fired.append({"rule": "email_already_settled", "outcome": self.wrap.outcome})
            return self.done(GOODBYE)
        if self.settled() and self.wrap.outcome == "sent" and not self.new_answers():
            return self.already_sent()
        if self.hints.email_promised:
            self.fired.append({"rule": "email_promised"})
            return await self.send()
        self.fired.append({"rule": "offer_email", "wants_email": asked})
        return self.stay(self.offer())

    def already_sent(self) -> StepOutcome:
        self.fired.append({"rule": "email_already_sent"})
        self.hints.wants_email = self.hints.email_promised = False
        return self.done(ReplyPart("email_already_sent", (self.where,),
                                   f"I've already sent the summary to {self.where.text}, and nothing new has come up "
                                   "since. Thank you for getting in touch."))

    # ---- the reply to the offer ----------------------------------------------------------------------------------

    async def send(self) -> StepOutcome:
        ctx = self.ctx
        g = await guidance(ctx.deps, ctx.conn)
        claims = await fetch_ids(ctx.conn, self.discussed())
        full = []
        if self.focus.email_list_filter is not None:
            full = await fetch(ctx.conn, filter_from_dict(self.focus.email_list_filter), ctx.today, FULL_LIST_LIMIT)
        subject, body = write_summary(claims, self.request, g, ctx.deps.settings, ctx.today, full_list=full)
        outbox_id = await send_summary(ctx.conn, ctx.conversation.id, subject, body)
        self.settle("sent")
        self.fired.append({"rule": "email_sent", "outbox_id": outbox_id, "claims": [c.case_id for c in claims],
                           "full_list": len(full)})
        return self.done(ReplyPart("email_sent", (self.where,),
                                   f"I've sent the summary to {self.where.text}. Thank you for getting in touch."))

    def skip(self, rule: str) -> StepOutcome:
        """Nothing is sent, and the conversation ends: the reply is a goodbye and asks nothing (a caller with something
        more to ask simply asks it, and the conversation reopens)."""
        self.settle("skipped")
        self.fired.append({"rule": rule})
        return self.done(ReplyPart("email_skipped", (self.where,),
                                   "Okay, I won't send an email. Thank you for getting in touch."))

    def contents(self) -> StepOutcome:
        said = "The email lists what we discussed, the status of each claim, and any follow-up items and deadlines."
        facts = [Fact("email_contents", said)]
        if self.focus.email_list_filter is not None:
            more = "It also lists the claims you asked for, one line each."
            facts.append(Fact("email_contents", more))
            said = f"{said} {more}"
        self.fired.append({"rule": "email_contents"})
        return self.stay(ReplyPart("email_contents", (*facts, self.where),
                                   f"{said} It would go to {self.where.text}. Would you like me to send it?",
                                   pending=EMAIL_OFFER))

    def address_not_read_out(self) -> StepOutcome:
        self.fired.append({"rule": "address_not_read_out"})
        return self.stay(ReplyPart("email_address_not_read_out", (self.where,),
                                   f"I can't read the address out, but the summary would reach {self.where.text}. "
                                   "Would you like me to send it?", pending=EMAIL_OFFER))

    def different_address(self) -> StepOutcome:
        self.fired.append({"rule": "different_address"})
        return self.stay(ReplyPart("different_address_not_possible", (self.where,),
                                   f"I can only send the summary to {self.where.text}."),
                         ReplyPart("offer_person", pending=PERSON_OFFER))

    def ask_again(self, rule: str) -> StepOutcome:
        self.fired.append({"rule": rule})
        return self.stay(self.offer("Sorry, I want to be sure."))

    def unclear(self, rule: str) -> StepOutcome:
        """A reply that does not settle the offer (unreadable, or a "send" read below the bar a send needs): asked
        again until limits.unclear_email_replies such replies, then nothing is sent and the conversation ends."""
        self.wrap.unclear_replies += 1
        if self.wrap.unclear_replies >= self.unclear_limit:
            return self.skip("unclear_limit")
        return self.ask_again(rule)

    # ---- something else ------------------------------------------------------------------------------------------

    async def something_else(self, unclear: bool) -> StepOutcome:
        form, t = self.form, self.t
        mentions = (yes(form.mentions_claim, t) or yes(form.quoted_claim_number, t)) \
            and picked(form.search_relation, t) in SWITCHING
        if mentions:
            self.fired.append({"rule": "another_claim"})
            return StepOutcome(State.RESOLVE_INTENT, hop=True, fired=self.fired)
        need = picked(form.need, t)
        if need not in (None, "none") and need != self.request.need:
            self.fired.append({"rule": "new_need", "need": need})
            return StepOutcome(State.RESOLVE_INTENT, hop=True, fired=self.fired)
        answer = picked(form.answer_wanted, t)
        if answer is not None and answer != "none":
            return await self.answer(answer)
        if unclear:
            return self.unclear("offer_again")
        return self.ask_again("offer_again")

    async def answer(self, answer: str) -> StepOutcome:
        """A question about a claim in focus: answered the way PROCESS_CASE answers it, then the offer again."""
        ctx = self.ctx
        claims = await fetch_ids(ctx.conn, self.focus.claim_ids)
        g = await guidance(ctx.deps, ctx.conn)
        s = ctx.deps.settings
        where = [c for c in claims if answer not in NOT_AN_ANSWER and answer in available(c, g, len(claims), s)]
        parts, grounded = await answer_parts(ctx, answer if where else NOT_ON_FILE, where or claims)
        self.fired.append({"rule": "answer" if where else "not_on_file", "answer": answer})
        if not where or any(p.pending is not None and p.pending.kind == PERSON_OFFER.kind for p in parts):
            return self.stay(*parts, grounded=grounded)
        return self.stay(*parts, self.offer(), grounded=grounded)


async def _reply(ctx: TurnContext) -> StepOutcome:
    # Defence in depth: nothing is sent for a conversation the database has not verified.
    if ctx.conversation.verified_party_id is None:
        return StepOutcome(State.POST_PROCESS, handover=("rule", "not_verified"),
                           fired=[{"rule": "post_process", "verified": False}])
    w = _Wrapup(ctx)
    if ctx.entered_by_hop:
        return await w.entry()
    if ctx.memory.pending is not None and ctx.memory.pending.kind == EMAIL_OFFER.kind:
        w.wrap.offers_made += 1  # the offer stood (it was not dropped for an offer of a person): now it counts
    f = ctx.form.email_offer_reply
    value = f.value if f is not None else None
    if value == "send":
        if f.p >= w.t.send_email_min:
            return await w.send()
        return w.unclear("send_not_sure")
    reply = picked(f, w.t) if f is not None else None
    if reply == "dont_send":
        return w.skip("email_declined")
    if reply == "whats_in_it":
        return w.contents()
    if reply == "which_address":
        return w.address_not_read_out()
    if reply == "different_address":
        return w.different_address()
    return await w.something_else(unclear=reply in (None, "unclear"))


async def _offer_as_worded(ctx: TurnContext, out: StepOutcome) -> str | None:
    """The wording of the offer the caller is answering: the previous reply as it was sent; on the turn that makes
    the offer, the offer's own text before the wording step."""
    if ctx.entered_by_hop:
        return next((p.plain for p in out.parts if p.instruction == "offer_email"), None)
    row = await (await ctx.conn.execute(
        "SELECT reply FROM turns WHERE conversation_id = %s AND turn_no = %s ORDER BY id DESC LIMIT 1",
        [ctx.conversation.id, ctx.memory.counters.turn_no - 1])).fetchone()
    return None if row is None else row[0]


async def step(ctx: TurnContext) -> StepOutcome:
    """The wrap-up step, with an event that records the offer, the caller's reply, the form's reading of it and what
    was done (the rule that ended the step; for a sent summary, the email's text)."""
    out = await _reply(ctx)
    if ctx.conversation.verified_party_id is None:
        return out
    action = next((f["rule"] for f in reversed(out.fired) if f.get("rule") not in (None, "post_process")), None)
    sent = next((f for f in out.fired if f.get("rule") == "email_sent"), None)
    email = None
    if sent is not None:
        row = await (await ctx.conn.execute("SELECT body FROM outbox WHERE id = %s", [sent["outbox_id"]])).fetchone()
        email = None if row is None else row[0]
    reply = ctx.form.email_offer_reply
    ctx.record("wrap_up", offer_as_worded=await _offer_as_worded(ctx, out), caller_reply=ctx.message,
               form_value=None if reply is None else {"value": reply.value, "p": reply.p}, action=action,
               email_text=email)
    return out


STEPS[State.POST_PROCESS] = step
