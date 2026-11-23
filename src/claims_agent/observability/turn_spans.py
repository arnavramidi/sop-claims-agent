"""A turn's stored record, laid out as the steps of a trace.

`turn_spans` is one pure function from (message, reply, record) to a list of span descriptions: the turn first, then
its steps in the order they happened. It touches no library and no network, so a record key that should show up in a
trace is added here and tested here. `BraintrustLog` is the only code that turns these descriptions into SDK calls.

Everything in a description is already masked (`mask_for_trace`), and nothing is added to the record's own data:
this only rearranges what the record holds. A piece the record lacks (an older shape, a turn that made no model
call) is left out; a step that cannot be built is left out and the others stand.

Token counts live only on the model-call steps: Braintrust adds metrics up over a trace, so counts on the turn too
would be counted twice. The form's questions as sent are a large part of a record; they are shown once, as the
input of the form step, and the record kept on the turn leaves them out where a form step shows them.
"""

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from claims_agent.observability.turn_log import _mask_text, mask_for_trace, model_calls, token_metrics

KEPT_RECORD_KEY = "record"
DRAFT_NOTE = "The first wording's text is not kept in the record; what the check rejected is shown instead."


@dataclass
class SpanDesc:
    name: str
    type: str  # "task", "llm", "function" or "tool"
    start: float | None = None
    end: float | None = None
    input: Any = None
    output: Any = None
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _dicts(x) -> list[dict]:
    return [i for i in x if isinstance(i, dict)] if isinstance(x, list) else []


def _number(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


# ---- the form -------------------------------------------------------------------------------------------------------

def _readings(record: dict) -> list[dict]:
    return _dicts(_dict(record.get("form_asked")).get("readings"))


def _question_sent(q: dict) -> dict:
    out = {"id": q.get("id"), "text": q.get("text")}
    if q.get("options") is not None:
        out["options"] = q["options"]
    return out


def _answer_given(q: dict) -> dict:
    answer = _dict(q.get("answer"))
    out: dict = {"id": q.get("id")}
    if "value" in answer:
        out["value"] = answer["value"]
    if answer.get("p") is not None:
        out["probability"] = answer["p"]
    out["mark"] = q.get("mark")
    return out


def _form_input(reading: dict) -> dict:
    out: dict = {"questions": [_question_sent(q) for q in _dicts(reading.get("questions"))]}
    if reading.get("part") is not None:
        out["part"] = reading["part"]
        out["message"] = reading.get("message")
    return out


def _form_output(reading: dict) -> dict:
    return {"answers": [_answer_given(q) for q in _dicts(reading.get("questions"))]}


def _settled(readings: list[dict]) -> list[dict]:
    """Every question a turn had to settle, in the order the form readings list them."""
    return [q for r in readings for q in _dicts(r.get("questions")) if isinstance(q.get("settled"), dict)]


def _settle_step(q: dict) -> tuple[dict, dict]:
    return ({"question": _question_sent(q), "form_answer": _answer_given(q)}, dict(q["settled"]))


# ---- the reply ------------------------------------------------------------------------------------------------------

def _parts_shown(record: dict) -> list[dict]:
    out = []
    for p in _dicts(record.get("parts")):
        shown: dict = {"instruction": p.get("instruction"),
                       "allowed_facts": [f.get("text") for f in _dicts(p.get("facts"))]}
        if p.get("plain"):
            shown["fixed_text"] = p["plain"]
        out.append(shown)
    return out


def _failures(check: dict) -> dict:
    return {k: check[k] for k in ("failures", "leak", "invented", "other") if check.get(k)}


def _word_io(job: str, record: dict, reply: str) -> tuple[dict, Any]:
    checks = _dicts(record.get("checks"))
    source = record.get("reply_source")
    inputs: dict = {"parts": _parts_shown(record)}
    if job == "rewrite" and checks:
        inputs["rejected_by_check"] = _failures(checks[0])
    if source == ("rewrite" if job == "rewrite" else "live"):
        return inputs, reply  # this wording is the one that went out
    rejected = _failures(checks[1 if job == "rewrite" and len(checks) > 1 else 0]) if checks else {}
    return inputs, {"rejected_by_check": rejected, "note": DRAFT_NOTE}


# ---- one step per model call ----------------------------------------------------------------------------------------

def _model_steps(message: str, reply: str, record: dict) -> list[SpanDesc]:
    readings = _readings(record)
    calls = model_calls(record)
    settled = _settled(readings)
    n_settle = sum(1 for c in calls if c.get("job") == "settle")
    with_larger = [q for q in settled if _dict(q["settled"]).get("larger") is not None]
    pairs = with_larger if len(with_larger) == n_settle else settled if len(settled) == n_settle else None
    seen = {"form": 0, "settle": 0}
    steps = []
    for call in calls:
        job = str(call.get("job") or "model call")
        try:
            inputs: Any = None
            output: Any = None
            if job == "form":
                k = seen["form"]
                seen["form"] += 1
                if k < len(readings):
                    inputs, output = _form_input(readings[k]), _form_output(readings[k])
            elif job == "extract":
                inputs, output = message, record.get("stated")
            elif job == "split":
                inputs = message
                output = {"questions": [r.get("message") for r in readings if r.get("part") is not None],
                          "stated": record.get("stated")}
            elif job == "settle":
                k = seen["settle"]
                seen["settle"] += 1
                if pairs is not None and k < len(pairs):
                    inputs, output = _settle_step(pairs[k])
                elif settled:
                    inputs = [_settle_step(q)[0] for q in settled]
                    output = [dict(q["settled"]) for q in settled]
            elif job in ("word", "rewrite"):
                inputs, output = _word_io(job, record, reply)
            steps.append(SpanDesc(name=job, type="llm", start=_number(call.get("started")),
                                  end=_number(call.get("ended")), input=inputs, output=output,
                                  metadata={"model": call.get("model")}, metrics=token_metrics(call)))
        except Exception:
            steps.append(SpanDesc(name=job, type="llm", start=_number(call.get("started")),
                                  end=_number(call.get("ended")), metadata={"model": call.get("model")},
                                  metrics=token_metrics(call)))
    return steps


# ---- code's own steps -----------------------------------------------------------------------------------------------

def _action_steps(record: dict) -> list[SpanDesc]:
    out = []
    for a in _dicts(_dict(record.get("db_actions")).get("actions")):
        out.append(SpanDesc(name=str(a.get("action") or "database action"), type="function",
                            start=_number(a.get("started")), end=_number(a.get("ended")),
                            output=a.get("outcome") if isinstance(a.get("outcome"), dict) else None,
                            metadata={"kind": "database"}))
    return out


def _form_fields(record: dict) -> dict:
    """The form's answered and unsure questions, by name (what the rules were handed from the form)."""
    answered, unsure = [], []
    for r in _readings(record):
        for q in _dicts(r.get("questions")):
            (answered if q.get("answered") else unsure if q.get("mark") in ("unsure", "below") else []).append(q.get("id"))
    return {"answered": answered, "unsure": unsure}


def _rules_step(record: dict, calls: list[dict]) -> SpanDesc | None:
    if not record.get("fired") and not record.get("path") and not record.get("state_after"):
        return None
    reading_jobs = [c for c in calls if c.get("job") in ("form", "extract", "split", "settle")]
    word_starts = [s for c in calls if c.get("job") in ("word", "rewrite") if (s := _number(c.get("started"))) is not None]
    start = max((e for c in reading_jobs if (e := _number(c.get("ended"))) is not None), default=None)
    end = min(word_starts, default=start)
    if start is not None and end is not None and end < start:
        end = start
    return SpanDesc(name="rules", type="function", start=start, end=end,
                    input={"state_before": record.get("state_before"), "form": _form_fields(record)},
                    output={"state_after": record.get("state_after"), "path": record.get("path"),
                            "fired": record.get("fired")})


def _check_step(record: dict, calls: list[dict]) -> SpanDesc | None:
    checks = _dicts(record.get("checks"))
    if not checks:
        return None
    first_word = next((c for c in calls if c.get("job") == "word"), None)
    rewrite = next((c for c in calls if c.get("job") == "rewrite"), None)
    start = _number(first_word.get("ended")) if first_word else None
    end = _number(rewrite.get("started")) if rewrite else start
    attempts = [{"attempt": i + 1, "passed": c.get("passed"), **_failures(c)} for i, c in enumerate(checks)]
    return SpanDesc(name="reply check", type="function", start=start, end=end, output={"attempts": attempts},
                    metadata={"reply_source": record.get("reply_source")})


# ---- scores and the turn itself -------------------------------------------------------------------------------------

def _scores(record: dict) -> dict:
    out: dict = {}
    checks = _dicts(record.get("checks"))
    if checks and isinstance(checks[0].get("passed"), bool):
        out["reply_check_first_pass"] = 1 if checks[0]["passed"] else 0
    marks = [q.get("mark") for r in _readings(record) for q in _dicts(r.get("questions"))
             if q.get("mark") not in (None, "no_answer")]
    if marks:
        out["form_confident"] = round(sum(m == "confident" for m in marks) / len(marks), 4)
    return out


def _kept_record(record: dict) -> dict:
    """The record as kept on the turn: whole, except that a reading whose questions a form step shows is not sent twice."""
    kept = copy.deepcopy(record)
    shown = sum(1 for c in model_calls(record) if c.get("job") == "form")
    readings = _dicts(_dict(kept.get("form_asked")).get("readings"))
    for i, r in enumerate(readings):
        if i < shown and "questions" in r:
            r["questions"] = f"shown once, as the input and output of the form step ({len(_dicts(r['questions']))} questions)"
    return kept


def _summary(record: dict, conversation_id: str | None, turn_no: int | None) -> dict:
    models = _dict(record.get("models"))
    summary = {"conversation_id": conversation_id, "turn_no": turn_no, "state_before": record.get("state_before"),
               "state_after": record.get("state_after"), "path": record.get("path"), "form_filler": record.get("filler"),
               "reply_source": record.get("reply_source"), "flags": record.get("flags"), "models": models or None}
    return {k: v for k, v in summary.items() if v is not None}


Builder = Callable[[], "SpanDesc | list[SpanDesc] | None"]


def turn_spans(*, message: str, reply: str, record: dict, tags: Sequence[str] = (), conversation_id: str | None = None,
               turn_no: int | None = None, started: float | None = None, ended: float | None = None) -> list[SpanDesc]:
    """The turn's span first, then its steps in the order they happened (by start time; steps with no time keep the
    order they were built in, after the steps before them). Masked. Never raises."""
    record = _dict(record)
    calls = model_calls(record)
    root = SpanDesc(name="turn", type="task", start=started, end=ended, input=message, output=reply, tags=list(tags))
    root.metadata = {**_summary(record, conversation_id, turn_no), KEPT_RECORD_KEY: record}
    try:
        root.scores = _scores(record)
        root.metadata[KEPT_RECORD_KEY] = _kept_record(record)
    except Exception:
        pass
    builders: list[Builder] = [lambda: _model_steps(message, reply, record), lambda: _action_steps(record),
                               lambda: _rules_step(record, calls), lambda: _check_step(record, calls)]
    steps: list[SpanDesc] = []
    for build in builders:
        try:
            built = build()
        except Exception:
            continue
        steps.extend(built if isinstance(built, list) else [built] if built is not None else [])
    steps = _in_order(steps)
    out = [root, *steps]
    return [_masked(s) for s in out]


def _in_order(steps: list[SpanDesc]) -> list[SpanDesc]:
    """By start time. A step without one goes where the steps built before it end, so it is placed after them."""
    last = 0.0
    keyed = []
    for i, s in enumerate(steps):
        if s.start is not None:
            last = s.start
        keyed.append((last if s.start is None else s.start, i, s))
    return [s for _, _, s in sorted(keyed, key=lambda k: (k[0], k[1]))]


# Identifiers the app made itself (random, about nobody). The ID-number mask reads a run of digits in one as an ID
# number and would break it, and turns could no longer be grouped by conversation.
OWN_IDS = ("conversation_id",)


def _masked(s: SpanDesc) -> SpanDesc:
    s.input = _mask_value(s.input)
    s.output = _mask_value(s.output)
    own = {k: s.metadata[k] for k in OWN_IDS if k in s.metadata}
    s.metadata = {**mask_for_trace(s.metadata), **own}
    return s


def _mask_value(v):
    return _mask_text(v) if isinstance(v, str) else mask_for_trace(v)
