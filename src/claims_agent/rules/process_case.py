"""PROCESS_CASE: answer the caller's questions about the claims in focus.

Every fact is built by code from each claim's own row (read fresh, by id, through the turn's connection, so the
database's row rules decide what is visible) and from the guidance data; a model only words them and the reply check
rejects anything the facts do not say. A follow-up is offered only when it can be truthfully filled for that claim,
and an offer belongs to the one claim it was made for.

What the caller says about a document is untrusted text (the fast model's pick from the caller's words, or the
caller's own words for a document beyond the ones offered as choices): alternatives are given only for documents on a
claim in focus, and recorded under that claim's own name for the document.

The `request` section is this state's: the need being answered, the answers given per claim, the documents whose
alternatives were offered per claim, and the follow-ups declined per claim. The claims in focus are RESOLVE_INTENT's;
this step only reads them.
"""

import re
from collections.abc import Sequence
from datetime import date

from claims_agent import prepared
from claims_agent.answers.documents import norm
from claims_agent.answers.facts import COMPARE_CLAIMS, AnswerResult, available, build, opening
from claims_agent.answers.followups import exists_for, fill, followup_candidates, next_followup
from claims_agent.db.claims import (Claim, ClaimFilter, chosen_among, count, fetch_ids, filter_from_form,
                                    find_numbered, with_need)
from claims_agent.db.guidance import Guidance
from claims_agent.engine import ANSWER_KINDS, STEPS, StepOutcome, TurnContext, guidance, offered_documents
from claims_agent.form.read import confident_yes, picked, yes
from claims_agent.loader.read import FollowupRow
from claims_agent.memory import Pending, RequestMemory
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.rules import calculate, unsure
from claims_agent.rules.every_message import PERSON_OFFER
from claims_agent.rules.resolve_intent import (GENERAL as GENERAL_NEED, destination_fact, sentences,
                                             wants_more_than_focus)
from claims_agent.settings import Settings
from claims_agent.states import State

# A yes/no offer of one follow-up for one claim: "offer_followup:<case_id>:<topic>". The label holds no claim value.
OFFER_PREFIX = "offer_followup:"
# Questions that are not yes/no live for one caller reply. Labels hold no value.
WHICH_ANSWER = Pending(kind="which_answer", label="which of these answers?", yes_no=False)
WHICH_DOCUMENT = Pending(kind="which_document", label="which document can't you get?", yes_no=False)
ANYTHING_ELSE = Pending(kind="anything_else", label="anything else?", yes_no=False)
# The email offer made when the caller came back from it (POST_PROCESS makes the same offer on entry).
EMAIL_OFFER = Pending(kind="offer_email", label="send the summary?", yes_no=True)

NOT_AN_ANSWER = ("none", "something_else", "not_on_file")
NOT_ON_FILE = "not_on_file"
# Why the need differs from what was asked: a fact about each claim whose fresh row still holds it.
NOTE_TEXT = {"not_denied": "Claim {id} has no denial reason on file, so it was not denied",
             "no_documents_requested": "No documents are requested for claim {id}"}
NOTE_HOLDS = {"not_denied": lambda c: not (c.denial_reason or "").strip(),
              "no_documents_requested": lambda c: not c.documents}
SWITCHING = ("new_search", "correcting")
# The form's which-document answers that are not a document: all of them, one not on the list, none named.
ALL_DOCUMENTS, NOT_LISTED, NONE_NAMED = "all", "not_listed", "none_stated"
GENERAL = "general"  # a document set that is no document on the claims in focus: the general alternatives


def offer_kind(case_id: str, topic: str) -> str:
    """The pending kind of an offer of one follow-up for one claim. Read back only by building it again."""
    return f"{OFFER_PREFIX}{case_id}:{topic}"


def _words(name: str) -> str:
    return name.replace("_", " ")


def _results(answer: str, claims: Sequence[Claim], g: Guidance, s: Settings, today: date,
             document: str | None) -> list[tuple[Claim, AnswerResult]]:
    """One result per claim: a claim-record answer is built from the row; a follow-up is filled only where it
    exists for that claim (the offer check), anything else is not on file for it."""
    followup = next((f for f in g.followups if f.topic == answer), None)
    if answer == COMPARE_CLAIMS and len(claims) >= 2:  # one result over all the claims, given with the first
        whole = build(answer, claims, g, s, today)
        return [(c, whole if i == 0 else AnswerResult(())) for i, c in enumerate(claims)]
    out = []
    for c in claims:
        if answer in ANSWER_KINDS:
            out.append((c, build(answer, [c], g, s, today)))
        elif followup is not None and exists_for(followup, c, g, s):
            out.append((c, fill(followup, c, g, s, today, document=document)))
        else:
            out.append((c, build(NOT_ON_FILE, [c], g, s, today)))
    return out


def _parts(results: Sequence[AnswerResult], lead: Sequence[Fact] = ()) -> tuple[list[ReplyPart], bool]:
    """The reply parts for some results. Facts of kind not_on_file go in their own part, whatever `grounded` says,
    and anything not on file (or a result that asks for it) offers a person. Grounded only when something was
    actually answered from the data."""
    told = [f for r in results for f in r.facts if f.kind != NOT_ON_FILE]
    missing = [f for r in results for f in r.facts if f.kind == NOT_ON_FILE]
    parts: list[ReplyPart] = []
    if lead or told:
        shown = (*lead, *told)
        parts.append(ReplyPart("answer", shown, sentences("Here is what I have on file.", shown)))
    if missing:
        parts.append(ReplyPart(NOT_ON_FILE, tuple(missing), sentences("", missing)))
    if missing or any(r.offer_person for r in results):
        parts.append(ReplyPart("offer_person", pending=PERSON_OFFER))
    return parts, bool(told) and any(r.grounded for r in results)


async def answer_parts(ctx: TurnContext, answer: str, claims: Sequence[Claim], *,
                       document: str | None = None) -> tuple[list[ReplyPart], bool]:
    """The parts that give one answer (a claim-record answer or a follow-up topic) for the claims given, and whether
    the reply is grounded in the data. Writes no memory, so any state may use it."""
    g = await guidance(ctx.deps, ctx.conn)
    results = _results(answer, claims, g, ctx.deps.settings, ctx.today, document)
    return _parts([r for _, r in results])


def _answered_for(result: AnswerResult) -> bool:
    return any(f.kind != NOT_ON_FILE for f in result.facts)


def _alternatives_topic(s: Settings) -> str | None:
    """The follow-up settings mark as giving per-document alternatives (Ruling 8: attached by topic name)."""
    return next((t for t, sp in s.followups.special.items() if sp.per_document_alternatives), None)


def _named_in(message: str, document: str) -> bool:
    """The caller's message names the document exactly (case and spacing aside), as whole words."""
    wanted = norm(document)
    return bool(wanted) and re.search(rf"(?<!\w){re.escape(wanted)}(?!\w)", norm(message)) is not None


def _named_in_part(message: str, document: str) -> bool:
    """The caller's message holds a word of the document's name (a word of four letters or more), as a whole word."""
    return any(len(w) >= 4 and _named_in(message, w) for w in norm(document).split())


class _Case:
    def __init__(self, ctx: TurnContext, claims: list[Claim], g: Guidance):
        self.ctx, self.claims, self.g = ctx, claims, g
        self.s, self.t, self.today = ctx.deps.settings, ctx.deps.thresholds, ctx.today
        self.form = ctx.form
        self.request: RequestMemory = ctx.memory.section("request", State.PROCESS_CASE)
        self.topics = {f.topic: f for f in g.followups}
        self.fired: list[dict] = [{"rule": "process_case", "claims": [c.case_id for c in claims]}]
        self.slot: prepared.Slot | None = None  # one follow-up for one claim: a prepared reply may answer it

    @property
    def need(self) -> str:
        return self.request.need or self.ctx.memory.focus.need or GENERAL_NEED

    def give_answer(self, answer: str, lead: Sequence[Fact] = ()) -> StepOutcome:
        """One specific answer asked for: given for the claims in focus it exists for (the mismatch recorded when it
        is not linked to the need), else not on file; then the next offer as usual."""
        where = [c for c in self.claims if answer not in NOT_AN_ANSWER and answer in self.available(c)]
        if not where:  # X3: the answer does not exist for any claim in focus
            self.fired.append({"rule": "not_on_file", "answer": answer})
            return self.finish(*_parts(self.results(NOT_ON_FILE, self.claims), lead))
        followup = self.topics.get(answer)
        if followup is not None and self.need not in followup.needs:
            self.fired.append({"mismatch": answer})
        self.fired.append({"rule": "answer", "answer": answer, "claims": [c.case_id for c in where]})
        results = self.results(answer, where)
        made = _parts(results, lead)
        if followup is not None and not lead and len(where) == 1:
            self.slot = prepared.Slot(where[0], answer, results[0], tuple(made[0]))
        return self.finish(*made)

    def available(self, claim: Claim) -> list[str]:
        return available(claim, self.g, len(self.claims), self.s)

    def mark(self, claim: Claim, answer: str) -> None:
        given = self.request.answered.setdefault(claim.case_id, [])
        if answer not in given:
            given.append(answer)

    def results(self, answer: str, claims: Sequence[Claim], document: str | None = None) -> list[AnswerResult]:
        """One answer for the claims given; each claim it was actually answered for is marked answered."""
        out = []
        for c, r in _results(answer, claims, self.g, self.s, self.today, document):
            if _answered_for(r):
                self.mark(c, answer)
            out.append(r)
        return out

    def stay(self, parts: list[ReplyPart], grounded: bool = False) -> StepOutcome:
        return StepOutcome(State.PROCESS_CASE, parts, grounded=grounded, fired=self.fired)

    def hop_back(self, rule: dict) -> StepOutcome:
        self.fired.append(rule)
        return StepOutcome(State.RESOLVE_INTENT, hop=True, fired=self.fired)

    def offer_next(self) -> ReplyPart:
        """The next follow-up for the need, for the first claim in focus that has one (neither answered nor
        declined for it); else ask if there is anything else. One question, carrying its pending."""
        for c in self.claims:
            given = self.request.answered.get(c.case_id, [])
            declined = self.request.declined.get(c.case_id, [])
            f = next_followup(self.need, c, [*given, *declined], self.g, self.s)
            if f is None:
                continue
            self.fired.append({"rule": "followup_offered", "need": self.need, "claim": c.case_id,
                               "answered": list(given), "declined": list(declined), "topic": f.topic})
            candidates = [x.topic for x in followup_candidates(self.need, c, [*given, *declined], self.g, self.s)]
            self.ctx.record("followup_choice", need=self.need, claim=c.case_id, answered=list(given),
                            candidates=candidates, chosen=f.topic,
                            why="the first candidate in order" if candidates[0] == f.topic else
                            "the first candidate that is not waiting for another topic to be answered")
            fact = Fact("followup", f"Follow-up available for claim {c.case_id}: {_words(f.topic)}", c.case_id)
            return ReplyPart("offer_followup", (fact,), f"Would you like to hear about {_words(f.topic)} "
                             f"for claim {c.case_id}?", pending=Pending(
                                 kind=offer_kind(c.case_id, f.topic), label=f"hear about {_words(f.topic)}?",
                                 yes_no=True))
        return ReplyPart("anything_else", pending=ANYTHING_ELSE)

    def finish(self, parts: list[ReplyPart], grounded: bool, *, offer: bool = True) -> StepOutcome:
        """After an answer: an offer of a person is the reply's one question. Otherwise, a caller who came back from
        the email offer is offered it again and moves to wrap-up; else the next follow-up (or anything else)."""
        if any(p.pending is not None and p.pending.kind == PERSON_OFFER.kind for p in parts):
            return self.stay(parts, grounded)
        if self.ctx.memory.wrapup.offers_made > 0:
            where = destination_fact(self.ctx.memory)
            parts.append(ReplyPart("offer_email", (where,), "Would you like a summary of what we discussed sent to "
                                   f"{where.text}?", pending=EMAIL_OFFER))
            self.fired.append({"rule": "offer_email_again"})
            return StepOutcome(State.POST_PROCESS, parts, grounded=grounded, fired=self.fired)
        parts.append(self.offer_next() if offer else ReplyPart("anything_else", pending=ANYTHING_ELSE))
        return self.stay(parts, grounded)

    # ---- a total, a difference, a ranking, a count or a day count ---------------------------------------------------

    async def calculate(self) -> StepOutcome | None:
        """A confident calculation is worked out by code over the claims in focus, before any answer pick is acted on
        (a message about another claim goes back to choosing it first)."""
        asked = calculate.wanted(self.ctx)
        if asked is None:
            return None
        form, t = self.form, self.t
        # Entered by a hop, the claim the message is about has just been chosen: work it out over that claim.
        if not self.ctx.entered_by_hop and ((yes(form.mentions_claim, t) or yes(form.quoted_claim_number, t))
                                            and picked(form.search_relation, t) in SWITCHING
                                            or await self.points_elsewhere()):
            return None
        calc, figures = asked
        if calc == "days_left" and self.deadline_followup_asked():
            return None  # the follow-up asked for gives the deadline with its days left, and what to do by then
        made, grounded = await calculate.parts(self.ctx, calc, figures, ClaimFilter(), self.claims,
                                               [c.case_id for c in self.claims])
        self.fired.append({"rule": "calculation", "calculation": calc, "figures": figures})
        if made and made[0].instruction == "ask_which_figure":
            return self.stay(made)
        return self.finish(made, grounded)

    async def points_elsewhere(self) -> bool:
        """Whether the message itself points at claims other than those in focus, however sure the fast model is that
        it starts a new search: a claim number that names none of them; a description (only values the model is sure
        the caller said, with the condition a confident need implies) that one of them does not fit, or that fits
        more claims than are in focus when the caller means several or all; or more claims asked for than are in
        focus ("all my claims", "my other claims")."""
        ctx, conn, today = self.ctx, self.ctx.conn, self.today
        ids = [c.case_id for c in self.claims]
        number = (ctx.stated.claim_number or "").strip()
        if number:  # a number that names another claim, or none: it is looked up (and said to be not on file)
            named = {c.case_id for c in await find_numbered(conn, number, self.s.limits.claims_in_focus)}
            if not named or not named <= set(ids):
                return True
        need = picked(self.form.need, self.t)
        described = with_need(filter_from_form(self.form, self.t), need)  # "denied" means a denial reason on file
        how_many = picked(self.form.how_many, self.t)
        if not described.is_empty():
            if len(await chosen_among(conn, described, today, ids)) < len(ids):
                return True
            return how_many in ("several", "all") and await count(conn, described, today) > len(ids)
        return wants_more_than_focus(how_many, len(ids)) and await count(conn, ClaimFilter(), today) > len(ids)

    def deadline_followup_asked(self) -> bool:
        """The message confidently asks for a follow-up that carries the claim's deadline (settings mark it)."""
        topic = picked(self.form.answer_wanted, self.t)
        special = self.s.followups.special.get(topic or "")
        return special is not None and special.with_deadline \
            and any(topic in self.available(c) for c in self.claims)

    def asks_about_a_figure(self) -> bool:
        """The form is sure the caller asks about one of the money figures and wants nothing worked out with it: the
        answer that gives the figures, each with its meaning on file, is the one asked for."""
        figures = any(confident_yes(v, self.t) for v in self.form.figures.values())
        return figures and calculate.wanted(self.ctx) is None and any("amounts" in self.available(c) for c in self.claims)

    # ---- entry -------------------------------------------------------------------------------------------------

    def entry(self) -> StepOutcome:
        """The need RESOLVE_INTENT settled, its note (for each claim whose fresh row holds it), then each claim's
        opening; or, when this turn's message confidently asked for one specific answer (a question about the claim
        just put in focus) that the opening neither gives nor explains, that answer in place of the opening."""
        focus = self.ctx.memory.focus
        self.request.need = focus.need or GENERAL_NEED
        note = focus.need_note if focus.need_note in NOTE_TEXT else None
        lead = [Fact(note, NOTE_TEXT[note].format(id=c.case_id), c.case_id)
                for c in self.claims if note is not None and NOTE_HOLDS[note](c)]
        answer = picked(self.form.answer_wanted, self.t)
        if answer is not None and answer not in NOT_AN_ANSWER:
            if not gives_opening(self.form, self.t, self.need, note, self.claims, self.s):
                self.fired.append({"rule": "entered_with_a_question", "need": self.need, "answer": answer})
                return self.give_answer(answer, lead)
            self.fired.append({"rule": "question_in_opening", "need": self.need, "answer": answer})
        results = []
        for c in self.claims:
            for a in opening(self.need, c, self.s):
                results.append(build(a, [c], self.g, self.s, self.today))
                self.mark(c, a)
        self.fired.append({"rule": "opening", "need": self.need, "note": note})
        return self.finish(*_parts(results, lead))

    # ---- a yes or a no to the follow-up offered last -----------------------------------------------------------

    def offer_target(self, pending: Pending) -> tuple[Claim, FollowupRow] | None:
        """The claim and follow-up an offer was made for: the one (claim in focus, known topic) pair whose offer kind,
        built again the same way, equals the pending's. Never parsed. None when no pair, or more than one, builds it
        (the offer no longer stands)."""
        matches = [(c, f) for c in self.claims for topic, f in self.topics.items()
                   if offer_kind(c.case_id, topic) == pending.kind]
        return matches[0] if len(matches) == 1 else None

    async def offer_answered(self) -> StepOutcome | None:
        pending = self.ctx.memory.pending
        if pending is None or not pending.kind.startswith(OFFER_PREFIX):
            return None
        reply = picked(self.form.pending_reply, self.t)
        if reply not in ("yes", "no"):
            return None
        target = self.offer_target(pending)
        if target is None:  # made for a claim no longer in focus, or naming nothing known: not an offer any more
            self.fired.append({"rule": "offer_not_standing", "kind": pending.kind})
            return None
        claim, f = target
        if reply == "no":
            declined = self.request.declined.setdefault(claim.case_id, [])
            if f.topic not in declined:
                declined.append(f.topic)
            self.fired.append({"rule": "followup_declined", "claim": claim.case_id, "topic": f.topic})
            return self.cant_get() or await self.asked()  # the rest of the message, if it asks anything
        if self.wants_document_help() and self.document_set():
            return self.cant_get()  # "yes, and I can't get the office note": that document's own alternatives
        if not exists_for(f, claim, self.g, self.s):
            self.fired.append({"rule": "offer_not_standing", "kind": pending.kind})
            return None
        self.fired.append({"rule": "followup_accepted", "claim": claim.case_id, "topic": f.topic})
        results = self.results(f.topic, [claim])
        made = _parts(results)
        self.slot = prepared.Slot(claim, f.topic, results[0], tuple(made[0]))
        return self.finish(*made)

    # ---- "I can't get it" ----------------------------------------------------------------------------------------

    def wants_document_help(self) -> bool:
        """The caller says they cannot get a document, or actually answers the question of which one they cannot get
        (a confident document on the claims in focus, or all of them). Any other reply is an ordinary message."""
        if yes(self.form.cant_get_document, self.t):
            return True
        pending = self.ctx.memory.pending
        if pending is None or pending.kind != WHICH_DOCUMENT.kind:
            return False
        which = picked(self.form.which_document, self.t)
        return which == ALL_DOCUMENTS or any(norm(d) == norm(which or "") for c in self.claims for d in c.documents)

    def document_set(self) -> list[tuple[Claim, str]] | str | None:
        """The (claim, document) pairs the caller cannot get, from the claims in focus only: every pair with the
        confidently named document; every pair for "all"; a document beyond the ones offered to the form when the
        message names it exactly; with no document named, the only one outstanding. GENERAL when the caller named a
        document that is on no claim in focus (the general alternatives: nothing recorded, never a hand-over); []
        when nothing is outstanding; None when it must be asked (several outstanding and none named)."""
        on_file = [(c, d) for c in self.claims for d in c.documents]
        if not on_file:
            return []
        which = picked(self.form.which_document, self.t)
        if which == ALL_DOCUMENTS:
            return on_file
        if which not in (None, NONE_NAMED, NOT_LISTED):
            return [(c, d) for c, d in on_file if norm(d) == norm(which)] or GENERAL
        offered = set(offered_documents(self.claims, self.s))
        named = [(c, d) for c, d in on_file if d not in offered and _named_in(self.ctx.message, d)]
        if named:
            return named
        if which == NOT_LISTED:
            return GENERAL
        if len(on_file) == 1:
            return on_file
        # "the report" when only one outstanding document is a report: named by part of its name.
        partly = [(c, d) for c, d in on_file if _named_in_part(self.ctx.message, d)]
        return partly if partly and len({norm(d) for _, d in partly}) == 1 else None

    def ask_which_document(self) -> StepOutcome:
        """Several documents outstanding and none named: ask which, listing each as an answer the caller can hear."""
        pairs = [(c, d) for c in self.claims for d in c.documents][: self.s.limits.documents_offered]
        facts = tuple(Fact("requested_document", f"Document requested for claim {c.case_id}: {d}", c.case_id)
                      for c, d in pairs)
        self.fired.append({"rule": "which_document"})
        return self.stay([ReplyPart("ask_which_document", facts, "Which document can't you get: " + ", or ".join(
            f"the {d} for claim {c.case_id}" for c, d in pairs) + "?", pending=WHICH_DOCUMENT)])

    def cant_get(self) -> StepOutcome | None:
        """Alternatives for each document in the set, each recorded; a confident "can't get it" when every document
        in the set already had its alternatives hands over (P14)."""
        if not self.wants_document_help():
            return None
        pairs = self.document_set()
        if pairs == []:
            return None  # nothing is outstanding: read the rest of the message
        if pairs is None:
            return self.ask_which_document()
        topic = _alternatives_topic(self.s) or NOT_ON_FILE
        if pairs == GENERAL:  # a document that is not on these claims: the general alternatives, nothing recorded
            self.fired.append({"rule": "alternatives", "documents": "not_listed"})
            return self.finish(*_parts(self.results(topic, [c for c in self.claims if c.documents])))
        offered = self.request.alternatives_offered
        named = [{"claim": c.case_id, "document": d} for c, d in pairs]
        if all(d in offered.get(c.case_id, []) for c, d in pairs) \
                and confident_yes(self.form.cant_get_document, self.t):
            self.fired.append({"rule": "alternatives_exhausted", "documents": named})
            return StepOutcome(State.PROCESS_CASE, handover=("rule", "document_unavailable"), fired=self.fired)
        results = []
        for c, d in pairs:
            got = self.results(topic, [c], d)
            results += got
            if any(_answered_for(r) for r in got) and d not in offered.get(c.case_id, []):
                offered.setdefault(c.case_id, []).append(d)
        self.fired.append({"rule": "alternatives", "documents": named})
        return self.finish(*_parts(results))

    # ---- what the caller asked for -------------------------------------------------------------------------------

    async def asked(self) -> StepOutcome:
        form, t = self.form, self.t
        # A different claim goes back to the claim-choosing state whatever the answer pick (it keeps the focus when
        # the message describes no other claim).
        mentions = yes(form.mentions_claim, t) or yes(form.quoted_claim_number, t)
        if mentions and picked(form.search_relation, t) in SWITCHING or await self.points_elsewhere():
            return self.hop_back({"rule": "another_claim"})
        answer = picked(form.answer_wanted, t)
        if answer in (None, "something_else", "none") and self.asks_about_a_figure():
            answer = "amounts"
        if answer is None:  # unsure: the larger model first, the caller asked only when it settles nothing
            settled = await unsure.answer_wanted(self.ctx)
            self.fired += unsure.fired(settled)
            answer = settled.value
        if answer == "something_else":
            self.fired.append({"rule": "not_on_file", "answer": answer})
            return self.finish(*_parts(self.results(NOT_ON_FILE, self.claims)))
        if answer is not None and answer != "none":
            return self.give_answer(answer)
        if answer is None:
            options = self.form.answer_wanted.options
            usable = {k for c in self.claims for k in self.available(c) if k not in NOT_AN_ANSWER}
            readings = sorted((k for k in options if k in usable), key=lambda k: options[k], reverse=True)[:2]
            if readings:
                facts = tuple(Fact("possible_answer", f"Possible answer: {_words(x)}") for x in readings)
                self.fired.append({"rule": "unsure_answer", "readings": readings})
                lead = "Just to check, which would you like to hear about: " if len(readings) == 2 else \
                    "Just to check, would you like to hear about "
                return self.stay([ReplyPart("ask_which_answer", facts,
                                            lead + ", or ".join(_words(x) for x in readings) + "?",
                                            pending=WHICH_ANSWER)])
        need = picked(form.need, t)
        if need not in (None, "none") and need != self.request.need:
            return self.hop_back({"rule": "new_need", "need": need})  # RESOLVE_INTENT settles it, keeping the focus
        return self.finish([], False, offer=False)


async def step(ctx: TurnContext) -> StepOutcome:
    # Defence in depth: claims are read only for a conversation the database has verified.
    if ctx.conversation.verified_party_id is None:
        return StepOutcome(State.PROCESS_CASE, handover=("rule", "not_verified"),
                           fired=[{"rule": "process_case", "verified": False}])

    # The claims in focus, read fresh. Those no longer visible are left out (the focus itself is RESOLVE_INTENT's to
    # correct); with none left, the caller goes back to choosing a claim.
    claims = await fetch_ids(ctx.conn, ctx.memory.focus.claim_ids)
    if not claims:
        return StepOutcome(State.RESOLVE_INTENT, hop=True, fired=[{"rule": "process_case", "focus_gone": True}])
    case = _Case(ctx, claims, await guidance(ctx.deps, ctx.conn))

    if ctx.entered_by_hop:
        out = await case.calculate() or case.entry()
    else:
        out = await case.offer_answered() or case.cant_get() or await case.calculate() or await case.asked()
    return await prepared.serve(ctx, case.slot, out)  # a follow-up worded ahead, when it may be used


STEPS[State.PROCESS_CASE] = step


# The answer each need note explains: the caller asked for it, and the note says why the claim has none.
NOTE_EXPLAINS = {"not_denied": "denial_reason", "no_documents_requested": "documents_needed"}


def opening_covers(answer: str, need: str, note: str | None, claims: Sequence[Claim], s: Settings) -> bool:
    """Whether the opening for the need already gives the answer the caller asked for (it is among the opening's
    answers for a claim in focus) or explains why there is none (`note`, set when it holds for a claim in focus).
    Then the caller gets the whole opening, which answers the question and says what comes with it."""
    if note is not None and NOTE_EXPLAINS.get(note) == answer:
        return True
    return any(answer in opening(need, c, s) for c in claims)


def gives_opening(form, t, need: str, note: str | None, claims: Sequence[Claim], s: Settings) -> bool:
    """Whether entering with this need gives the claims' opening (their status, and the reason or documents the need
    brings), rather than the one specific answer the message asked for in its place."""
    answer = picked(form.answer_wanted, t)
    if answer is None or answer in NOT_AN_ANSWER:
        return True
    holds = note in NOTE_TEXT and any(NOTE_HOLDS[note](c) for c in claims)
    return opening_covers(answer, need, note if holds else None, claims, s)
