"""RESOLVE_INTENT: settle which of the caller's claims are meant, and what the caller needs.

Every claim read goes through the turn's connection as the server's account, so the database's row rules decide
what is visible: another customer's claim cannot be found, listed or counted. A claim number the caller typed is
untrusted text; it is normalised and compared as a parameter, and a number that is not this caller's gets the same
answer whether or not it exists for someone else. Each fact about a claim is built from that claim's own row.

What the focus section holds after this step: `claim_filter` is the filter actually used for what was last shown or
put in focus (need conditions included) and `match_count` is that filter's count; the need's conditions are dropped
again when the stored filter is read back for a new description, because the need is re-applied every turn.
"""

from collections.abc import Sequence
from dataclasses import replace

from claims_agent.db.claims import (Claim, ClaimFilter, count, fetch, fetch_ids, filter_from_dict, filter_from_form,
                                    filter_to_dict, find_numbered, merge, most_even_split, overview, relax,
                                    with_need)
from claims_agent.db.identity import account_info
from claims_agent.rules import calculate
from claims_agent.engine import PLAIN, STEPS, StepOutcome, TurnContext
from claims_agent.form.read import picked, top_two, yes
from claims_agent.memory import FocusMemory, Memory, Pending
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.states import State

GENERAL = "general_claim_question"  # Ruling 6: claims in focus and no clear need open with Status
NOT_A_NEED = (None, "none", "account_question")
SPLIT_WORDS = {"case_type": "the claim type", "status": "the status", "year": "the year", "month": "the month"}
FULL = ("case_type", "status", "created_at")

# Questions that ask the caller to choose live for one reply (the engine clears them after it). Labels hold no value.
WHICH_CLAIM = Pending(kind="which_claim", label="which of the listed claims?", yes_no=False)
NARROW_DOWN = Pending(kind="narrow_down", label="which detail narrows the claims down?", yes_no=False)
OVERVIEW = Pending(kind="claims_overview", label="narrow the list down, or have it in the summary email?",
                   yes_no=False)
WHICH_NEED = Pending(kind="which_need", label="which of the two needs?", yes_no=False)
# The open need question carries a pending too, so any question the agent asks replaces whatever was pending before.
OPEN_NEED = Pending(kind="ask_need", label="what can I help you with?", yes_no=False)

NO_MATCH = Fact("no_match", "No claim matched that description")
NO_MATCH_LEAD = "I couldn't find a claim matching that description."


def destination_fact(memory: Memory) -> Fact:
    """Where a summary email goes, as the only phrase the reply may use for it (by the caller's role)."""
    third = memory.identity.caller_role == "third_party"
    return Fact("destination", "the policyholder's email" if third else "the email on your policy")


def sentences(lead: str, facts: Sequence[Fact], tail: str = "") -> str:
    """Plain text built only from fixed wording and the facts' own text (each fact a sentence of its own). The one
    copy: PROCESS_CASE builds its plain text with it too."""
    said = " ".join(f.text if f.text.endswith((".", "?", "!")) else f"{f.text}." for f in facts)
    return " ".join(x for x in (lead, said, tail) if x)


def _describe(c: Claim, attrs: tuple[str, ...]) -> Fact:
    """One claim, from its own row only, naming the attributes asked for."""
    said = []
    for a in attrs:
        if a == "case_type" and c.case_type:
            said.append(f"{c.case_type} claim")
        elif a == "status":
            said.append(f"status {c.status}")
        elif a == "created_at" and c.created_at:
            said.append(f"created {c.created_at.isoformat()}")
        elif a == "summary" and c.summary:
            said.append(f"summary: {c.summary}")
    return Fact("status", f"{c.case_id}: " + ", ".join(said) if said else c.case_id, c.case_id)


def _differing(claims: list[Claim]) -> tuple[str, ...]:
    """What tells the listed claims apart: the type, status and date that differ, else their summaries."""
    attrs = tuple(a for a in FULL if len({getattr(c, a) for c in claims}) > 1)
    if attrs:
        return attrs
    return ("summary",) if len({c.summary for c in claims}) > 1 else FULL


def _which_claim(claims: list[Claim]) -> ReplyPart:
    facts = (Fact("count", f"Claims to choose from: {len(claims)}"),
             *(_describe(c, _differing(claims)) for c in claims))
    return ReplyPart("which_claim", facts, sentences("Which claim do you mean?", facts), pending=WHICH_CLAIM)


def _first_pass(focus: FocusMemory) -> bool:
    """Nothing has been resolved since verification: the hints are still the stored description."""
    return focus.claim_filter is None and focus.match_count is None and not focus.claim_ids and not focus.last_list


def _described_part(f: ClaimFilter) -> ClaimFilter:
    """A stored filter as a description: the need's conditions are left out, because the need is applied afresh."""
    return replace(f, has_denial_reason=False, has_documents=False)


class _Resolve:
    def __init__(self, ctx: TurnContext):
        self.ctx, self.conn, self.today = ctx, ctx.conn, ctx.today
        self.t = ctx.deps.thresholds
        self.limit = ctx.deps.settings.limits.claims_in_focus
        self.focus: FocusMemory = ctx.memory.section("focus", State.RESOLVE_INTENT)
        self.parts: list[ReplyPart] = []
        self.fired: list[dict] = []

    def stay(self, **kw) -> StepOutcome:
        return StepOutcome(State.RESOLVE_INTENT, self.parts, fired=self.fired, **kw)

    def shown(self, f: ClaimFilter, n: int, listed: list[Claim]) -> None:
        """Record what the reply shows: the filter used, its count, the list offered (nothing in focus)."""
        self.focus.claim_filter, self.focus.match_count = filter_to_dict(f), n
        self.focus.claim_ids, self.focus.last_list = [], [c.case_id for c in listed]

    async def overview_part(self, f: ClaimFilter, lead: tuple[Fact, ...] = ()) -> ReplyPart:
        o = await overview(self.conn, f, self.today)
        facts = [*lead, Fact("count", f"Claims in this list: {o.count}")]
        if o.by_type:
            facts.append(Fact("by_type", "By type: " + ", ".join(f"{k} {v}" for k, v in sorted(o.by_type.items()))))
        if o.by_year:
            facts.append(Fact("by_year", "By year: " + ", ".join(f"{k} {v}" for k, v in sorted(o.by_year.items()))))
        facts += [_describe(c, FULL) for c in o.most_recent]
        if o.upcoming_deadlines:
            facts.append(Fact("deadlines", "Upcoming appeal deadlines: " + ", ".join(
                f"{c.case_id} on {c.appeal_deadline.isoformat()}" for c in o.upcoming_deadlines)))
        where = destination_fact(self.ctx.memory)
        content = tuple(f for f in facts if f.kind != "no_match")
        return ReplyPart("overview", (*facts, where), sentences(
            (NO_MATCH_LEAD + " " if lead else "") + "Here is an overview of the claims on file.", content,
            f"I can narrow this down, or include the full list in the summary email, sent to {where.text}."),
            pending=OVERVIEW)

    async def narrow_part(self, f: ClaimFilter, n: int, lead: tuple[Fact, ...] = ()) -> ReplyPart:
        split = await most_even_split(self.conn, f, self.today)
        facts = (*lead, Fact("count", f"Claims that match: {n}"),
                 Fact("narrow_by", f"Most useful detail to narrow it down: {SPLIT_WORDS[split]}"))
        content = tuple(x for x in facts if x.kind != "no_match")
        return ReplyPart("narrow_down", facts, sentences(
            (NO_MATCH_LEAD + " " if lead else "") + "There are too many claims to list here.", content,
            "Could you tell me which one you mean?"), pending=NARROW_DOWN)

    async def many(self, f: ClaimFilter, n: int, how_many: str, lead: tuple[Fact, ...] = ()) -> None:
        """More claims than a list can show: the overview when they meant all of them, else the narrowing question."""
        self.shown(f, n, [])
        if how_many in ("several", "all"):
            self.parts.append(await self.overview_part(f, lead))
        else:
            self.parts.append(await self.narrow_part(f, n, lead))

    async def on_file(self, total: int) -> None:
        """What is on file, after a claim number that is not on the policy: every claim listed, or the overview."""
        everything = ClaimFilter()
        if total <= self.limit:
            claims = await fetch(self.conn, everything, self.today, self.limit)
            self.shown(everything, total, claims)
            self.parts.append(_which_claim(claims))
        else:
            self.shown(everything, total, [])
            self.parts.append(await self.overview_part(everything))

    async def no_match(self, described: ClaimFilter, how_many: str) -> None:
        """Nothing matched: say so, then show the nearest matches (the weakest conditions dropped, Ruling 10), or what
        is on file when nothing of the description is left. Never a value the caller gave; a list never claims more
        entries than it shows."""
        relaxed = relax(described)
        while relaxed is not None and relaxed.conditions() and await count(self.conn, relaxed, self.today) == 0:
            relaxed = relax(relaxed)
        if relaxed is None or not relaxed.conditions():
            relaxed = ClaimFilter()
        m = await count(self.conn, relaxed, self.today)
        self.fired.append({"rule": "relaxed", "filter": filter_to_dict(relaxed), "matches": m})
        if m > self.limit:
            await self.many(relaxed, m, how_many, (NO_MATCH,))
            return
        shown = await fetch(self.conn, relaxed, self.today, self.limit)
        self.shown(relaxed, m, shown)
        facts = (NO_MATCH, Fact("count", f"Nearest matches: {len(shown)}"), *(_describe(c, FULL) for c in shown))
        lead = NO_MATCH_LEAD + (" These are the nearest." if relaxed.conditions() else " Here is what is on file.")
        self.parts.append(ReplyPart("nearest_matches", facts, sentences(lead, facts[1:]), pending=WHICH_CLAIM))

    async def describe(self, described: ClaimFilter, need: str | None, how_many: str) -> list[Claim] | None:
        """Steps 3 and 4 for a description: query, then decide by match count and how many were meant. Returns the
        claims now in focus, or None when the caller is shown something and asked."""
        conn, today = self.conn, self.today
        f = with_need(described, need)
        n = await count(conn, f, today)
        if n == 0 and f != described:  # the need is impossible for what was described: it enters at Status later
            f, n = described, await count(conn, described, today)
        self.fired.append({"rule": "resolve_intent", "filter": filter_to_dict(f), "matches": n})
        if n == 0:
            await self.no_match(described, how_many)
            return None
        if n == 1 or (n <= self.limit and how_many in ("several", "all")):
            self.focus.claim_filter, self.focus.match_count = filter_to_dict(f), n
            return await fetch(conn, f, today, self.limit)
        if n <= self.limit:
            claims = await fetch(conn, f, today, self.limit)
            self.shown(f, n, claims)
            self.parts.append(_which_claim(claims))
            return None
        await self.many(f, n, how_many)
        return None


def _need(ctx: TurnContext, t) -> str | None:
    """The confident pick, else the hint; never "none" or an account question (that is answered with no claim)."""
    pick = picked(ctx.form.need, t)
    if pick not in NOT_A_NEED:
        return pick
    hint = ctx.memory.hints.need
    return None if hint in NOT_A_NEED else hint


async def _which(r: _Resolve, total: int, need: str | None, how_many: str) -> list[Claim] | None:
    """Step 3: which claims, the first rule that applies. None when the reply asks or tells instead."""
    ctx, form, t, conn, today, focus = r.ctx, r.ctx.form, r.t, r.conn, r.today, r.focus
    hints = ctx.memory.hints
    first = _first_pass(focus)
    new = filter_from_form(form, t)
    relation = picked(form.search_relation, t) or "no_list_shown"
    number = (ctx.stated.claim_number or "").strip() or ((hints.claim_number or "").strip() if first else "")
    listed = picked(form.listed_claim, t)
    listed = None if listed in (None, "none") else listed
    all_afresh = how_many == "all" and relation == "new_search" and not new.conditions()

    if total == 1:
        claims = await fetch(conn, ClaimFilter(), today, 1)
        if number and not any(c.case_id == claims[0].case_id for c in await find_numbered(conn, number, r.limit)):
            r.parts.append(ReplyPart("claim_not_on_policy", plain="I don't see that claim on your policy."))
        focus.claim_filter, focus.match_count = None, 1
        r.fired.append({"rule": "only_claim"})
        return claims

    if number:
        found = await find_numbered(conn, number, r.limit)
        r.fired.append({"rule": "claim_number", "found": len(found)})
        if len(found) == 1:
            focus.claim_filter, focus.match_count = None, 1
            return found
        if found:  # two of the caller's own claims share that number: ask which
            r.shown(ClaimFilter(), len(found), found)
            r.parts.append(_which_claim(found))
            return None
        r.parts.append(ReplyPart("claim_not_on_policy", plain="I don't see that claim on your policy."))
        await r.on_file(total)
        return None

    if listed is not None and focus.last_list:
        if listed in focus.last_list:
            claims = await fetch_ids(conn, [listed])
            if claims:
                focus.match_count = 1
                r.fired.append({"rule": "listed_claim"})
                return claims
        else:  # X14: the model picked a claim that was not shown
            shown = await fetch_ids(conn, focus.last_list)
            if shown:
                r.parts.append(_which_claim(shown))
                r.fired.append({"rule": "listed_claim", "rejected": True})
                return None
        focus.last_list = []  # the list is no longer visible: treat it as not found and read the message afresh

    # Claims already in focus and nothing in the message points elsewhere: keep them; only the need is settled.
    if focus.claim_ids and not new.conditions() and not all_afresh:
        claims = await fetch_ids(conn, focus.claim_ids)
        if claims:
            r.fired.append({"rule": "kept_focus"})
            return claims
        focus.claim_ids = []

    stored = filter_from_dict(focus.claim_filter) if focus.claim_filter is not None else \
        filter_from_dict(hints.claim_filter) if first else None
    stored = None if stored is None else _described_part(stored)
    if not new.conditions() and how_many == "none" and (stored is None or not stored.conditions()):
        # The open need question is this state's, asked whenever nothing is resolved (also straight after
        # verification: the verified part itself asks nothing).
        r.parts.append(ReplyPart("ask_need", plain=PLAIN["ask_need"], pending=OPEN_NEED))
        r.fired.append({"rule": "no_claim_reference"})
        return None
    described = ClaimFilter() if all_afresh else merge(stored, new, relation)
    if how_many == "same_as_before":
        how_many = "all" if ctx.memory.pending is not None and ctx.memory.pending.kind == OVERVIEW.kind else "one"
    return await r.describe(described, need, how_many)


async def step(ctx: TurnContext) -> StepOutcome:
    r = _Resolve(ctx)
    form, t, conn, today, focus = ctx.form, r.t, r.conn, r.today, r.focus

    # Defence in depth: claims are read only for a conversation the database has verified.
    if ctx.conversation.verified_party_id is None:
        return StepOutcome(State.RESOLVE_INTENT, handover=("rule", "not_verified"),
                           fired=[{"rule": "resolve_intent", "verified": False}])

    # 1. No claims at all: say so and wrap up.
    total = await count(conn, ClaimFilter(), today)
    if total == 0:
        r.parts.append(ReplyPart("no_claims_on_file", plain="I can't see any claims on file for this policy."))
        r.fired.append({"rule": "no_claims_on_file"})
        return StepOutcome(State.POST_PROCESS, r.parts, hop=True, fired=r.fired)

    # 2. An account question needs no claim.
    pick = picked(form.need, t)
    if pick == "account_question" or (_first_pass(focus) and pick in (None, "none")
                                      and ctx.memory.hints.need == "account_question"):
        info = await account_info(conn, ctx.conversation.id)
        facts = (Fact("account", f"Policy number: {info[1]}"),) if info is not None and info[1] else ()
        r.parts.append(ReplyPart("answer", facts, sentences("Here is what I have on file.", facts)))
        r.fired.append({"rule": "account_question"})
        return r.stay(grounded=bool(facts))

    need = _need(ctx, t)
    how_many = picked(form.how_many, t) or "none"

    # The full list of the overview just shown, asked for by email: exactly what the overview described.
    pending = ctx.memory.pending
    if pending is not None and pending.kind == OVERVIEW.kind and yes(form.wants_email_summary, t):
        focus.email_list_filter = focus.claim_filter
        where = destination_fact(ctx.memory)
        facts = (Fact("email", "The summary email will include the full list of claims from the overview"), where)
        r.parts.append(ReplyPart("email_contents", facts,
                                 f"I'll include the full list in the summary email, sent to {where.text}."))
        r.fired.append({"rule": "email_list"})
        new = filter_from_form(form, t)
        listed = picked(form.listed_claim, t)
        if not (ctx.stated.claim_number or "").strip() and listed in (None, "none") and not new.conditions():
            return r.stay()

    # A total, highest, lowest or count over claims in general needs no single claim: it runs over the current
    # filter (or every claim) and the state stays.
    stored = filter_from_dict(focus.claim_filter) if focus.claim_filter is not None else \
        filter_from_dict(ctx.memory.hints.claim_filter) if _first_pass(focus) else None
    done = await calculate.over_filter(ctx, how_many, None if stored is None else _described_part(stored))
    if done is not None:
        return done

    claims = await _which(r, total, need, how_many)
    if claims is None:
        return r.stay()

    # 4 and 5. Claims in focus; then the need.
    focus.claim_ids = [c.case_id for c in claims][: r.limit]
    focus.last_list = []
    note = None
    if need is None:
        readings = top_two(form.need) if picked(form.need, t) is None else []
        readings = [x for x in readings if x not in NOT_A_NEED]
        if len(set(readings)) == 2:
            facts = tuple(Fact("possible_need", f"Possible need: {x.replace('_', ' ')}") for x in readings)
            r.parts.append(ReplyPart("ask_need", facts, "Just to check, which do you mean: "
                                     + ", or ".join(x.replace("_", " ") for x in readings) + "?", pending=WHICH_NEED))
            r.fired.append({"rule": "unsure_need", "readings": readings})
            focus.need, focus.need_note = None, None
            return r.stay()
        need = GENERAL
    if need == "denial_question" and not any((c.denial_reason or "").strip() for c in claims):
        need, note = "status_inquiry", "not_denied"
    elif need == "document_submission" and not any(c.documents for c in claims):
        need, note = "status_inquiry", "no_documents_requested"

    # 6. Both known: on to the answers.
    focus.need, focus.need_note = need, note
    r.fired.append({"rule": "in_focus", "claims": focus.claim_ids, "need": need, "note": note})
    return StepOutcome(State.PROCESS_CASE, r.parts, hop=True, fired=r.fired)


STEPS[State.RESOLVE_INTENT] = step
