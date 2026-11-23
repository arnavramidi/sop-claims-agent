import asyncio
import re
from decimal import Decimal
from pathlib import Path

from claims_agent.db import conversations as store
from claims_agent.form.fields import Form, Pick, StatedValues, YesNo
from claims_agent.form.filler import ScriptedExtractor, ScriptedFiller
from claims_agent.form.switching import SwitchingFiller
from claims_agent.loader.read import read_folder
from claims_agent.memory import Memory, Pending
from claims_agent.reply.brief import Fact, ReplyPart
from claims_agent.reply.wording import ScriptedWording
from claims_agent.settings import ROOT, load_settings, load_thresholds

_YES_NO_DEFAULT = ["wants_person", "skips_step", "asks_why_step", "declines", "done", "other_person_policy",
                   "wants_email_summary", "mentions_claim", "quoted_claim_number", "no_year_said",
                   "multi_question", "cant_get_document", "asks_personal_detail"]
_PICK_DEFAULT = {
    "about": "claim_or_account", "mood": "calm", "pending_reply": "nothing_pending", "how_many": "none",
    "relative_time": "none", "search_relation": "no_list_shown", "listed_claim": "none", "need": "none",
    "answer_wanted": "none", "which_document": "none_stated",
}
_OPTIONAL_YES_NO = ["gave_identity_detail", "correcting_detail", "partial_detail", "consent_recheck"]
_OPTIONAL_PICK = ["caller_role", "relationship", "email_offer_reply"]


def _yn(v) -> YesNo:
    if isinstance(v, YesNo):
        return v
    if v is None:
        return YesNo(p=None)
    return YesNo(p=1.0 if v is True else 0.0 if v is False else float(v))


def _pick(v) -> Pick:
    return v if isinstance(v, Pick) else Pick(value=v, p=1.0, options={v: 1.0})


def make_form(**overrides) -> Form:
    values: dict = {n: YesNo(p=0.0) for n in _YES_NO_DEFAULT}
    values["english"] = YesNo(p=1.0)
    values.update({n: _pick(v) for n, v in _PICK_DEFAULT.items()})
    values["shaped_need"] = [YesNo(p=0.0) for _ in range(6)]
    for n in _OPTIONAL_YES_NO + _OPTIONAL_PICK:
        values[n] = None
    for n in ("claim_types", "claim_statuses", "months", "years", "figures", "refuses"):
        values[n] = {}
    values["no_figure_said"] = YesNo(p=0.0)
    values["calculation"] = _pick("none")
    for key, v in overrides.items():
        if key in ("claim_types", "claim_statuses", "months", "years", "figures", "refuses"):
            values[key] = {k: _yn(x) for k, x in v.items()}
        elif key == "shaped_need":
            values[key] = [_yn(x) for x in v]
        elif key in _PICK_DEFAULT or key in _OPTIONAL_PICK or key == "calculation":
            values[key] = None if v is None else _pick(v)
        else:
            values[key] = None if v is None else _yn(v)
    return Form(**values)


def stated(**values) -> StatedValues:
    return StatedValues(**values)


async def verified_party(where, conversation_id) -> str | None:
    """The customer a conversation is verified as, read through the server's own account.
    `where` is anything exposing an AppDb as `.app` (a TestDb) or `.db` (later, Deps)."""
    app = where.app if hasattr(where, "app") else where.db
    async with app.conversation(conversation_id) as c:  # the row rules show only the bound conversation
        row = await (await c.execute(
            "SELECT verified_party_id FROM conversations WHERE id = %s", [conversation_id])).fetchone()
    return None if row is None else row[0]


def with_held(memory, kind, value):
    """A copy of `memory` with one held identity value (as stated and normalised the same)."""
    from claims_agent.memory import HeldValue
    out = memory.model_copy(deep=True)
    out.identity.held[kind] = HeldValue(as_stated=value, key=value, turn=out.counters.turn_no)
    return out


# ---- whole-turn helpers ----------------------------------------------------------------------------
# Conversations are built by talking through `say`, never by writing rows.

STARTER = ROOT / "data" / "starter"

GAVE = dict(gave_identity_detail=True, caller_role="policyholder")
BRIEF_MSG = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare "
             "claim from January. DOB is 1985-03-15, SSN last four is 4472.")
BRIEF_SAID = stated(policyholder_name="Margaret Chen", policy_number="POL-9921", date_of_birth=["1985-03-15"],
                    id_digits="4472")
# The assignment's message as the fast model should read it: an identity detail from the policyholder, about one denied
# healthcare claim from January (no year said), with no clear need.
FORM = make_form(**GAVE, mentions_claim=True, claim_types={"healthcare": 0.95}, claim_statuses={"denied": 0.95},
                 months={1: 0.95}, no_year_said=True, how_many="one")


class TurnFiller(ScriptedFiller):
    """The scripted filler `say` feeds: one form per turn; what a turn did not use is dropped after it."""

    def discard_unused(self) -> None:
        self._items.clear()


class TurnExtractor(ScriptedExtractor):
    """The scripted extractor `say` feeds: one set of stated values per turn, dropped when the engine did not ask."""

    def discard_unused(self) -> None:
        self._items.clear()


def make_deps(db, **overrides):
    """Engine dependencies over a loaded test database (`TestDb`): scripted filler, extractor and wording."""
    from claims_agent.engine import Deps
    values = dict(settings=load_settings(), thresholds=load_thresholds(), db=db.app, guidance=None,
                  filler=TurnFiller(), extractor=TurnExtractor(), wording=HoldingWording(),
                  settler=ScriptedSettler())
    values.update(overrides)
    return Deps(**values)


async def new_conversation(deps, *, consent_script: str = "default"):
    async with deps.db.conversation(None) as c:
        return await store.create(c, consent_script=consent_script)


async def say(deps, conversation_id, text, form=None, said=None):
    """One whole turn: queue the form and the stated values the scripted models will return, run it, and drop
    whatever the turn did not use (the extractor is not asked after verification unless a claim number is quoted)."""
    from claims_agent.engine import run_turn
    deps.filler.queue(make_form() if form is None else form)
    deps.extractor.queue(stated() if said is None else said)
    try:
        return await run_turn(deps, conversation_id, text)
    finally:
        deps.filler.discard_unused()
        deps.extractor.discard_unused()


def _part(p: dict) -> ReplyPart:
    return ReplyPart(p["instruction"], tuple(Fact(**f) for f in p["facts"]), p["plain"],
                     None if p["pending"] is None else Pending(**p["pending"]))


def parts(r) -> list[ReplyPart]:
    return [_part(p) for p in r.inspector["parts"]]


def instructions(r) -> list[str]:
    return [p["instruction"] for p in r.inspector["parts"]]


def facts(r) -> list[Fact]:
    return [f for p in parts(r) for f in p.facts]


async def _read(deps, conversation_id, query: str, params: list):
    """Read as the server's account, bound to the conversation (the row rules show no other conversation's rows)."""
    async with deps.db.conversation(conversation_id) as c:
        return await (await c.execute(query, params)).fetchall()


async def memory(deps, conversation_id, path: str | None = None):
    """The conversation's saved memory, or one value inside it (`"identity.wrong_values"`, `"pending"`)."""
    rows = await _read(deps, conversation_id, "SELECT memory FROM conversations WHERE id = %s", [conversation_id])
    value = Memory.model_validate(rows[0][0])
    for name in (path.split(".") if path else []):
        value = getattr(value, name)
    return value


async def focus(deps, conversation_id) -> list[str]:
    return (await memory(deps, conversation_id)).focus.claim_ids


async def claims_visible(deps, conversation_id) -> int:
    """How many claim rows the database returns to the server's account for this conversation."""
    return (await _read(deps, conversation_id, "SELECT count(*) FROM claims", []))[0][0]


async def escalation_reasons(deps, conversation_id) -> list[str]:
    rows = await _read(deps, conversation_id, "SELECT reason FROM escalations WHERE conversation_id = %s ORDER BY id",
                       [conversation_id])
    return [r for (r,) in rows]


async def turn_record(deps, turn_id) -> dict:
    """A turn's stored record, by the turn's id alone: read as the loader (a test fixture account), since the
    server's account sees a turn only from a transaction bound to its conversation."""
    import os

    import psycopg
    async with await psycopg.AsyncConnection.connect(os.environ["LOADER_DATABASE_URL"]) as c:
        return (await (await c.execute("SELECT record FROM turns WHERE id = %s", [turn_id])).fetchone())[0]


async def conversation(deps, conversation_id):
    async with deps.db.conversation(conversation_id) as c:
        conv, _ = await store.load_for_turn(c, conversation_id)
    return conv


async def turn_count(deps, conversation_id) -> int:
    return (await _read(deps, conversation_id, "SELECT count(*) FROM turns WHERE conversation_id = %s",
                        [conversation_id]))[0][0]


async def checks_used(deps, conversation_id) -> int:
    """Consent checks the database has counted for this conversation's request (0 with no request)."""
    rows = await _read(deps, conversation_id, "SELECT checks_used FROM consent_requests WHERE conversation_id = %s",
                       [conversation_id])
    return rows[0][0] if rows else 0


async def request_rows(deps, conversation_id) -> int:
    return (await _read(deps, conversation_id, "SELECT count(*) FROM consent_requests WHERE conversation_id = %s",
                        [conversation_id]))[0][0]


async def first_reply(deps, said) -> ReplyPart:
    """A new conversation, one turn with the GAVE form; its `ask_another_detail` part."""
    r = await say(deps, await new_conversation(deps), "here are my details", make_form(**GAVE), said)
    return next(p for p in parts(r) if p.instruction == "ask_another_detail")


def every_field_yes():
    """Every identity, claim and need field at p=1.0 with caller_role policyholder, but wants_person, done and
    other_person_policy at 0, so the turn reaches the state's own rules."""
    form = make_form(
        wants_person=False, done=False, other_person_policy=False, skips_step=True, asks_why_step=True, declines=True, wants_email_summary=True, mentions_claim=True,
        quoted_claim_number=True, no_year_said=True, multi_question=True, cant_get_document=True,
        asks_personal_detail=True, gave_identity_detail=True, correcting_detail=True, partial_detail=True,
        consent_recheck=True, caller_role="policyholder", relationship="son", need="denial_question",
        how_many="all", relative_time="most_recent", answer_wanted="status",
        claim_types={t: 1.0 for t in ("healthcare", "dental", "auto")},
        claim_statuses={s: 1.0 for s in ("denied", "closed", "open")},
        months={m: 1.0 for m in range(1, 13)}, years={y: 1.0 for y in range(2022, 2027)},
        shaped_need=[1.0] * 6)
    return form


def _amount_pattern(amount: Decimal) -> re.Pattern:
    whole, cents = f"{amount:.2f}".split(".")
    grouped = f"{int(whole):,}"
    shapes = {whole, grouped}
    alternatives = "|".join(re.escape(s) for s in sorted(shapes, key=len, reverse=True))
    return re.compile(rf"(?<![\d.,])(?:{alternatives})(?:\.{cents})?(?![\d])" if cents == "00"
                      else rf"(?<![\d.,])(?:{alternatives})\.{cents}(?!\d)")


def assert_no_claim_data(text: str, folder: Path = STARTER) -> None:
    """Fail if any claim number, summary, denial reason, document name, amount or claim date from the data set
    appears in `text`. Reads the files with `read_folder`, never through the app account."""
    data = read_folder(folder)
    low = text.lower()
    for c in data.claims:
        for value in (c.case_id, c.summary, c.denial_reason):
            assert not value or value.lower() not in low, f"claim data in text: {value!r}"
        for d in (c.created_at, c.appeal_deadline):
            assert d is None or d.isoformat() not in text, f"claim date in text: {d}"
        for amount in (c.expected_reimbursement_amount, c.allowed_max_amount, c.net_pay, c.net_fee):
            if amount:  # a zero amount is not a detail of anyone's claim
                assert not _amount_pattern(amount).search(text), f"claim amount in text: {amount}"
    for d in data.documents:
        assert d.name.lower() not in low, f"document name in text: {d.name!r}"



class SwitchingTestFiller(SwitchingFiller):
    """A switching filler whose two scripted fillers are both fed by `say` (one form per turn each)."""

    def queue(self, item) -> None:
        self.primary.queue(item)
        self.fallback.queue(item)

    def discard_unused(self) -> None:
        self.primary._items.clear()
        self.fallback._items.clear()


class ScriptedSettler:
    """A stand-in for the larger model settling an unsure pick: gives out its queued picks in order (a queued error is
    raised), or raises `error` on every call; with nothing queued it cannot tell either."""

    def __init__(self, picks=(), *, error: BaseException | None = None) -> None:
        self._picks: list = list(picks)
        self._error = error
        self.calls: list = []  # (request, question) per call

    def queue(self, pick) -> None:
        self._picks.append(pick)

    async def settle(self, req, question) -> Pick:
        self.calls.append((req, question))
        if self._error is not None:
            raise self._error
        if not self._picks:
            return Pick(value=None, by="larger")
        item = self._picks.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class HoldingWording(ScriptedWording):
    """The scripted wording model, with `hold()`: the next wording call waits until the returned event is set (so a
    test can let a turn run while a reply is still being worded in the background). `entered` is set when that call
    has reached the gate, so a test can wait for it instead of guessing how long it takes."""

    def __init__(self, replies=None):
        super().__init__(replies)
        self._gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    def hold(self) -> asyncio.Event:
        self._gate = asyncio.Event()
        self.entered = asyncio.Event()
        return self._gate

    async def word(self, brief):
        gate, self._gate = self._gate, None
        if gate is not None:
            self.entered.set()
            await gate.wait()
        return await super().word(brief)
