"""Runs one scripted caller conversation against the real models and checks, after every turn, what must be true.

A scenario file (YAML) holds a one-line description, the data set, the date the conversation takes as today, the
consent script and the caller's turns. Each turn has the caller's words and the expectations for what comes back.
Every check is plain code over what the engine returned (state, verified customer, claims in focus, facts, reply
check, inspector entries) or over the reply text; a model judge is used only where a scenario asks for one, and only
for wording qualities.

One check runs on every turn of every scenario whatever the file says: nothing about any claim of the data set may be
said to the caller, or sent to a model, before the caller is verified. A value the caller typed themselves does not
count (repeating the caller's own claim number confirms nothing). On a turn that starts unverified, the requests that
read the caller's message (the form and the extraction) are made before any state rule runs, so before the database
can bind the caller: they are always checked. The rest of that turn's requests (an unsure pick settled, the reply
worded) come after the state rules; they are checked when the turn ends unverified too, and are after verification
when it does not.
"""

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import openai
import yaml

from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps, run_turn
from claims_agent.form.extractor import TextExtractor
from claims_agent.form.settle import ModelSettler
from claims_agent.form.switching import build_switching_filler
from claims_agent.loader.read import read_folder
from claims_agent.models.client import make_client
from claims_agent.reply.wording import ModelWording
from claims_agent.settings import ROOT, load_settings, load_thresholds

SCENARIOS = Path(__file__).parent / "scenarios"
DATA = ROOT / "tests" / "data"

SCENARIO_KEYS = {"id", "about", "group", "dataset", "today", "consent_script", "waiting_for", "turns"}
TURN_KEYS = {"say", "repeat", "expect"}
EXPECT_KEYS = {
    "state", "verified", "focus", "focus_listed", "answers", "instructions", "instructions_lack", "fact_kinds",
    "facts_have", "reply_has", "reply_lacks", "check_passed", "reply_source", "no_claim_data_before_verification",
    "handed_over", "outbox_count", "outbox_lacks", "held_kinds", "held", "stated", "wrong_values", "stalled",
    "consent_checks", "claims_visible", "filter", "mood", "flags", "fired", "inspector_lacks", "memory", "judge",
}
FILTER_KEYS = {"types", "statuses", "months", "years", "relative"}

JudgeFn = Callable[[str, Sequence[dict]], Awaitable[tuple[bool, str]]]


@dataclass(frozen=True)
class ScenarioResult:
    id: str
    passed: bool
    failures: tuple[str, ...]
    transcript: tuple[dict, ...]


@dataclass(frozen=True)
class Turn:
    say: str
    expect: dict


@dataclass(frozen=True)
class Scenario:
    id: str
    about: str
    group: str
    dataset: str
    today: date | None
    consent_script: str
    waiting_for: str | None
    turns: tuple[Turn, ...]


def load_scenario(path: Path) -> Scenario:
    """Read and check one scenario file. An unknown key is an error, so a misspelt expectation can never be skipped
    silently."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: not a mapping")
    unknown = set(raw) - SCENARIO_KEYS
    if unknown:
        raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
    if raw.get("id") != path.stem:
        raise ValueError(f"{path.name}: id must be the file name ({path.stem})")
    for required in ("about", "group", "turns"):
        if not raw.get(required):
            raise ValueError(f"{path.name}: {required} is missing")
    turns = []
    for i, t in enumerate(raw["turns"], 1):
        if set(t) - TURN_KEYS or not isinstance(t.get("say"), str) or not t["say"].strip():
            raise ValueError(f"{path.name}: turn {i} needs readable `say` text and only {sorted(TURN_KEYS)}")
        expect = t.get("expect") or {}
        if set(expect) - EXPECT_KEYS:
            raise ValueError(f"{path.name}: turn {i} has unknown expectations {sorted(set(expect) - EXPECT_KEYS)}")
        if set(expect.get("filter") or {}) - FILTER_KEYS:
            raise ValueError(f"{path.name}: turn {i} filter keys must be among {sorted(FILTER_KEYS)}")
        if expect.get("no_claim_data_before_verification") is False:
            raise ValueError(f"{path.name}: turn {i}: nothing about a claim before verification is always checked")
        text = " ".join([t["say"]] * int(t.get("repeat", 1)))
        turns.append(Turn(text, expect))
    today = raw.get("today")
    return Scenario(id=raw["id"], about=str(raw["about"]), group=str(raw["group"]),
                    dataset=str(raw.get("dataset", "starter")),
                    today=today if isinstance(today, date) or today is None else date.fromisoformat(str(today)),
                    consent_script=str(raw.get("consent_script", "default")),
                    waiting_for=raw.get("waiting_for"), turns=tuple(turns))


def dataset_folder(name: str) -> Path:
    return ROOT / "data" / "starter" if name == "starter" else DATA / name


# ---- what was sent to the models ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sent:
    body: str
    reading: bool  # made while the caller's message was being read (form fill or extraction)


class SentLog:
    """Every request body sent to the models through the client it is attached to, in order."""

    def __init__(self) -> None:
        self.bodies: list[Sent] = []
        self.reading = 0  # reading calls in progress (the form and the extraction run side by side)

    async def record(self, request) -> None:
        self.bodies.append(Sent((await request.aread()).decode("utf-8", "replace"), self.reading > 0))

    def http_client(self):
        return openai.DefaultAsyncHttpxClient(event_hooks={"request": [self.record]})


class _Reading:
    """Passes everything through to `inner`; while one of `methods` runs, requests are marked as reading."""

    def __init__(self, inner, sent: SentLog, methods: tuple[str, ...]):
        self._inner, self._sent, self._methods = inner, sent, methods

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if name not in self._methods:
            return attr

        async def marked(*args, **kwargs):
            self._sent.reading += 1
            try:
                return await attr(*args, **kwargs)
            finally:
                self._sent.reading -= 1
        return marked


def real_deps(db: AppDb, sent: SentLog) -> Deps:
    """The engine's dependencies as the server builds them (switching filler, extractor, wording, settler), all over
    one client made by `make_client` whose requests are recorded in `sent`. No tracing."""
    s = load_settings()
    client = make_client(http_client=sent.http_client())
    kw = {"retries": s.limits.model_retries, "timeout": s.limits.model_timeout_seconds}
    return Deps(settings=s, thresholds=load_thresholds(), db=db, guidance=None,
                filler=_Reading(build_switching_filler(s, client), sent, ("fill",)),
                extractor=_Reading(TextExtractor(client, s.models.text, **kw), sent, ("extract",)),
                wording=ModelWording(client, s.models.text, s.models.larger, **kw),
                settler=ModelSettler(client, s.models.larger, **kw))


def _strings(value: Any) -> list[str]:
    """Every string inside a JSON value; a string that is itself JSON is opened too (the data records are)."""
    if isinstance(value, str):
        try:
            inner = json.loads(value)
        except ValueError:
            return [value]
        return [value] if isinstance(inner, str) else [value, *_strings(inner)]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def body_text(body: str) -> str:
    try:
        return "\n".join(_strings(json.loads(body)))
    except ValueError:
        return body


# ---- the data set's claim details ----------------------------------------------------------------------------------

def _amount_pattern(amount: Decimal) -> re.Pattern:
    whole, cents = f"{amount:.2f}".split(".")
    shapes = sorted({whole, f"{int(whole):,}"}, key=len, reverse=True)
    alternatives = "|".join(re.escape(s) for s in shapes)
    if cents == "00":
        return re.compile(rf"(?<![\d.,])(?:{alternatives})(?:\.00)?(?!\d)")
    return re.compile(rf"(?<![\d.,])(?:{alternatives})\.{cents}(?!\d)")


def claim_details(folder: Path) -> list[tuple[str, re.Pattern]]:
    """(label, pattern) for every claim number, summary, denial reason, document name, amount and claim date in the
    data set. Read from the files, never through the app's account."""
    data = read_folder(folder)
    out: list[tuple[str, re.Pattern]] = []

    def text(label: str, value: str | None) -> None:
        if value:
            out.append((label, re.compile(re.escape(value), re.IGNORECASE)))

    for c in data.claims:
        text(f"claim number {c.case_id}", c.case_id)
        text(f"summary of {c.case_id}", c.summary)
        text(f"denial reason of {c.case_id}", c.denial_reason)
        for name, d in (("opened date", c.created_at), ("deadline", c.appeal_deadline)):
            if d is not None:
                out.append((f"{name} of {c.case_id}", re.compile(re.escape(d.isoformat()))))
        for name in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee"):
            amount = getattr(c, name)
            if amount:  # a zero amount is nobody's detail
                out.append((f"{name} of {c.case_id}", _amount_pattern(amount)))
    for d in data.documents:
        text(f"document name {d.name}", d.name)
    return out


def leaked(text: str, details: list[tuple[str, re.Pattern]], caller_said: str) -> list[str]:
    """Claim details found in `text`, leaving out any the caller typed themselves."""
    return [label for label, pattern in details if pattern.search(text) and not pattern.search(caller_said)]


# ---- the checks ----------------------------------------------------------------------------------------------------

def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def _one_of(value, expected) -> bool:
    return value in expected if isinstance(expected, list) else value == expected


def _dig(value: Any, dotted: str) -> Any:
    for name in dotted.split("."):
        value = value.get(name) if isinstance(value, dict) else None
    return value


def _sub_dict(small: dict, big: dict) -> bool:
    return all(big.get(k) == v for k, v in small.items())


@dataclass
class _Seen:
    """What one turn returned, flattened for the checks."""
    reply: str
    state: str
    verified: str | None
    inspector: dict
    previous_memory: dict
    instructions: list[str] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)

    @property
    def memory(self) -> dict:
        return self.inspector.get("memory") or {}


def check_turn(expect: dict, seen: _Seen) -> list[str]:
    """Every expectation of one turn that does not hold, each as one line."""
    out: list[str] = []
    insp, mem = seen.inspector, seen.memory
    e = expect

    if "state" in e and not _one_of(seen.state, e["state"]):
        out.append(f"state is {seen.state}, expected {e['state']}")
    if "verified" in e:
        want = None if e["verified"] in (None, "none", False) else e["verified"]
        if seen.verified != want:
            out.append(f"verified customer is {seen.verified}, expected {want}")
    focus = list(_dig(mem, "focus.claim_ids") or [])
    if "focus" in e and sorted(focus) != sorted(e["focus"] or []):
        out.append(f"claims in focus are {focus}, expected {e['focus']}")
    if "focus_listed" in e:
        listed = list(_dig(seen.previous_memory, "focus.last_list") or [])
        n = int(e["focus_listed"])
        want = listed[n - 1:n]
        if not want or focus != want:
            out.append(f"claims in focus are {focus}, expected number {n} of the list shown before ({listed})")
    if "answers" in e:
        given = {a for names in (_dig(mem, "request.answered") or {}).values() for a in names}
        missing = [a for a in e["answers"] if a not in given]
        if missing:
            out.append(f"answers not given: {missing} (given: {sorted(given)})")
    if "instructions" in e:
        missing = [i for i in e["instructions"] if i not in seen.instructions]
        if missing:
            out.append(f"reply parts lack {missing} (parts: {seen.instructions})")
    if "instructions_lack" in e:
        extra = [i for i in e["instructions_lack"] if i in seen.instructions]
        if extra:
            out.append(f"reply parts include {extra} (parts: {seen.instructions})")
    kinds = [f["kind"] for f in seen.facts]
    if "fact_kinds" in e:
        missing = [k for k in e["fact_kinds"] if k not in kinds]
        if missing:
            out.append(f"facts lack kinds {missing} (kinds: {kinds})")
    fact_text = "\n".join(f["text"] for f in seen.facts)
    for pattern in e.get("facts_have", []):
        if not _rx(pattern).search(fact_text):
            out.append(f"no fact matches /{pattern}/")
    for pattern in e.get("reply_has", []):
        if not _rx(pattern).search(seen.reply):
            out.append(f"reply does not match /{pattern}/")
    for pattern in e.get("reply_lacks", []):
        m = _rx(pattern).search(seen.reply)
        if m:
            out.append(f"reply matches /{pattern}/ ({m.group()!r})")
    if "check_passed" in e:
        checks = insp.get("checks") or []
        passed = bool(checks) and checks[-1].get("passed") is True and insp.get("reply_source") != "last_resort"
        if passed != bool(e["check_passed"]):
            out.append(f"reply check passed is {passed} (source {insp.get('reply_source')}, "
                       f"failures {[c.get('failures') for c in checks]}), expected {e['check_passed']}")
    if "reply_source" in e and not _one_of(insp.get("reply_source"), e["reply_source"]):
        out.append(f"reply source is {insp.get('reply_source')}, expected {e['reply_source']}")
    if "handed_over" in e and (seen.state == "ESCALATED") != bool(e["handed_over"]):
        out.append(f"handed over is {seen.state == 'ESCALATED'}, expected {e['handed_over']}")
    outbox = insp.get("outbox") or []
    if "outbox_count" in e and len(outbox) != e["outbox_count"]:
        out.append(f"outbox holds {len(outbox)}, expected {e['outbox_count']}")
    mail = "\n".join(f"{o.get('subject', '')}\n{o.get('body', '')}" for o in outbox)
    for pattern in e.get("outbox_lacks", []):
        m = _rx(pattern).search(mail)
        if m:
            out.append(f"outbox matches /{pattern}/ ({m.group()!r})")
    held = _dig(mem, "identity.held") or {}
    if "held_kinds" in e and set(held) != set(e["held_kinds"] or []):
        out.append(f"held kinds are {sorted(held)}, expected {sorted(e['held_kinds'] or [])}")
    for kind, key in (e.get("held") or {}).items():
        got = (held.get(kind) or {}).get("key")
        if got != key:
            out.append(f"held {kind} is {got!r}, expected {key!r}")
    stated = insp.get("stated") or {}
    for name, want in (e.get("stated") or {}).items():
        got = stated.get(name)
        ok = (not got if want is None else want in got) if isinstance(got, list) else got == want
        if not ok:
            out.append(f"stated {name} is {got!r}, expected {want!r}")
    counts = insp.get("counts") or {}
    for key, count in (("wrong_values", "wrong_values"), ("stalled", "stalled"), ("consent_checks", "consent_checks")):
        if key in e and counts.get(count) != e[key]:
            out.append(f"{key} is {counts.get(count)}, expected {e[key]}")
    if "claims_visible" in e and insp.get("claims_visible") != e["claims_visible"]:
        out.append(f"claim rows visible: {insp.get('claims_visible')}, expected {e['claims_visible']}")
    if "filter" in e:
        got = _dig(mem, "hints.claim_filter") or {}
        for key, want in e["filter"].items():
            have = got.get(key)
            if isinstance(want, list):
                same = sorted(str(v) for v in (have or [])) == sorted(str(v) for v in want)
            else:
                same = have == want
            if not same:
                out.append(f"filter {key} is {have!r}, expected {want!r} (filter: {got})")
    if "mood" in e:
        moods = _dig(mem, "counters.mood_by_turn") or []
        mood = moods[-1] if moods else None
        if not _one_of(mood, e["mood"]):
            out.append(f"mood read as {mood}, expected {e['mood']}")
    for flag in e.get("flags", []):
        if flag not in (insp.get("flags") or []):
            out.append(f"flag {flag} not raised (flags: {insp.get('flags')})")
    for entry in e.get("fired", []):
        if not any(isinstance(f, dict) and _sub_dict(entry, f) for f in insp.get("fired") or []):
            out.append(f"no fired entry matches {entry} (fired: {insp.get('fired')})")
    whole = json.dumps(insp, ensure_ascii=False)
    for pattern in e.get("inspector_lacks", []):
        m = _rx(pattern).search(whole)
        if m:
            out.append(f"inspector matches /{pattern}/ ({m.group()!r})")
    for dotted, want in (e.get("memory") or {}).items():
        got = _dig(mem, dotted)
        if got != want:
            out.append(f"memory {dotted} is {got!r}, expected {want!r}")
    return out


# ---- one scenario ----------------------------------------------------------------------------------------------------

async def _verified(deps: Deps, conversation_id) -> str | None:
    async with deps.db.conversation(conversation_id) as c:  # the row rules show only the bound conversation
        row = await (await c.execute("SELECT verified_party_id FROM conversations WHERE id = %s",
                                     [conversation_id])).fetchone()
    return None if row is None else row[0]


def _form_summary(form: dict | None) -> dict:
    """The form's readings that say something: picks with their sureness, and yes/no answers of 0.2 or more."""
    out: dict = {}
    for name, v in (form or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[name] = f"{v.get('value')} ({v.get('p')})"
        elif isinstance(v, dict) and set(v) <= {"p", "by"}:
            if (v.get("p") or 0) >= 0.2:
                out[name] = v.get("p")
        elif isinstance(v, dict):
            some = {k: x.get("p") for k, x in v.items() if isinstance(x, dict) and (x.get("p") or 0) >= 0.2}
            if some:
                out[name] = some
    return out


def _entry(n: int, said: str, seen: _Seen) -> dict:
    insp = seen.inspector
    moods = _dig(seen.memory, "counters.mood_by_turn") or []
    return {
        "turn": n, "caller": said, "reply": seen.reply, "state": seen.state, "verified": seen.verified,
        "focus": list(_dig(seen.memory, "focus.claim_ids") or []), "parts": seen.instructions,
        "facts": seen.facts, "reply_source": insp.get("reply_source"),
        "check_failures": [c.get("failures") for c in insp.get("checks") or []],
        "mood": moods[-1] if moods else None, "fired": insp.get("fired") or [], "flags": insp.get("flags") or [],
        "form": _form_summary(insp.get("form")),
    }


async def run_scenario(path: Path, make_deps: Callable[[str], Awaitable[Deps]], *, sent: SentLog | None = None,
                       judge_with: JudgeFn | None = None) -> ScenarioResult:
    """Talk through the scenario's turns with the deps `make_deps(dataset)` returns and check each turn. `sent` is the
    log of request bodies those deps' client records; without it the before-verification check of what was sent to
    the models cannot pass. `judge_with(rubric, transcript)` answers the scenario's judge sentences."""
    from claims_agent.db import conversations as store

    sc = load_scenario(path)
    deps = await make_deps(sc.dataset)
    details = claim_details(dataset_folder(sc.dataset))
    async with deps.db.conversation(None) as c:
        cid = await store.create(c, consent_script=sc.consent_script, today=sc.today)

    failures: list[str] = []
    transcript: list[dict] = []
    judges: list[tuple[int, str]] = []
    caller_said = ""
    verified_before: str | None = None
    previous_memory: dict = {}
    for n, turn in enumerate(sc.turns, 1):
        caller_said += "\n" + turn.say
        mark = len(sent.bodies) if sent is not None else 0
        r = await run_turn(deps, cid, turn.say)
        parts = r.inspector.get("parts") or []
        seen = _Seen(reply=r.reply, state=r.state.value, verified=await _verified(deps, cid),
                     inspector=r.inspector, previous_memory=previous_memory,
                     instructions=[p["instruction"] for p in parts],
                     facts=[f for p in parts for f in p.get("facts") or []])
        found = [f"turn {n}: {line}" for line in check_turn(turn.expect, seen)]
        if verified_before is None:  # the turn began unverified: see the module's note on which requests count
            if sent is None:
                found.append(f"turn {n}: what was sent to the models was not recorded")
            else:
                for i, item in enumerate(sent.bodies[mark:], 1):
                    if not item.reading and seen.verified is not None:
                        continue
                    for label in leaked(body_text(item.body), details, caller_said):
                        found.append(f"turn {n}: model request {i} carried {label} before verification")
        if seen.verified is None:
            for label in leaked(r.reply, details, caller_said):
                found.append(f"turn {n}: reply said {label} before verification")
        failures.extend(found)
        transcript.append(_entry(n, turn.say, seen))
        if "judge" in turn.expect:
            judges.append((n, turn.expect["judge"]))
        verified_before, previous_memory = seen.verified, seen.memory

    for n, rubric in judges:  # after the conversation, so no judge request is mixed into a turn's requests
        if judge_with is None:
            failures.append(f"turn {n}: no judge available for: {rubric}")
            continue
        ok, why = await judge_with(rubric, transcript[:n])
        transcript[n - 1]["judge"] = {"rubric": rubric, "passed": ok, "why": why}
        if not ok:
            failures.append(f"turn {n}: judge: {rubric} -- {why}")
    return ScenarioResult(sc.id, not failures, tuple(failures), tuple(transcript))
