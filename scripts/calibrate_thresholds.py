"""Fit the thresholds from labelled cases read by the real form-filling model, instead of setting them by hand.

  --record  sends every case in evals/labelled_cases.jsonl ({message, context, expect: {field: value}}) through the
            Decisions filler once and writes each labelled field's reading to evals/probabilities.jsonl. Needs
            OPENAI_API_KEY; costs one form fill per case.
  --fit     reads evals/probabilities.jsonl, picks each threshold's value, writes evals/thresholds.fitted.toml (a
            comment above each value says how it was chosen) and prints it next to the current thresholds.toml.
  --apply   copies evals/thresholds.fitted.toml over thresholds.toml. A separate step, so a person decides.

How a value is chosen: of the candidates 0.05, 0.10, ... 0.95, the one with the fewest wrong reads on the labelled
cases wins, and a tie goes to the more cautious (higher) value. send_email_min and prepared_reply_min may not let a
single labelled "no" through. A threshold with fewer than 20 labelled cases keeps its current value, and says so.

The thresholds fitted are the ones thresholds.toml has (the Thresholds model's fields) that some labelled reading
feeds; a reading that would feed a threshold the file does not have is an error, never silently dropped.

Which reading feeds which threshold: a yes/no question feeds yes; every pick feeds pick_confident (right when it
picked the labelled option); email_offer_reply also feeds send_email_min (a "send" reading at its confidence);
answer_wanted in a case labelled calm with a single question also feeds prepared_reply_min. Claim-filter yes/no
questions (claim type, status, month, year) feed nothing: a filter value counts only at the top of the unsure band
(unsure_high), which every reading that ends, resets or hands over the conversation shares, so one kind of question
cannot fit it. unsure_low and unsure_high are not fitted.

Run:  docker compose --profile test run --rm tests python scripts/calibrate_thresholds.py --record
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tomllib
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from claims_agent.examples import known_choices, load_examples, with_approved
from claims_agent.form.fields import Form, Pick, YesNo
from claims_agent.form.questions import FormRequest, build_questions
from claims_agent.settings import ROOT, Thresholds, load_settings, load_thresholds
from claims_agent.states import State

CASES = ROOT / "evals" / "labelled_cases.jsonl"
PROBABILITIES = ROOT / "evals" / "probabilities.jsonl"
FITTED = ROOT / "evals" / "thresholds.fitted.toml"
THRESHOLDS = ROOT / "thresholds.toml"

NO_FALSE_YES = ("send_email_min", "prepared_reply_min")
FILTER_GROUPS = ("claim_type", "claim_status", "month", "year")
MINIMUM_CASES = 20


@dataclass(frozen=True)
class Labelled:
    threshold: str
    p: float
    expected_yes: bool


# ---- the selection rule ---------------------------------------------------------------------------------------------

def candidates() -> list[float]:
    """0.05 to 0.95 in exact twentieths (an unrounded multiple such as 0.9000000000000001 would read a 0.9 as no)."""
    return [round(0.05 * i, 2) for i in range(1, 20)]


def wrong_reads(cases: Sequence[Labelled], t: float) -> tuple[int, int]:
    """(labelled no read as yes, labelled yes read as no) at threshold t; a reading at t is a yes, as in the app."""
    false_yes = sum(1 for c in cases if not c.expected_yes and c.p >= t)
    missed = sum(1 for c in cases if c.expected_yes and c.p < t)
    return false_yes, missed


def sweep(cases: Sequence[Labelled], *, current: float, forbid_false_yes: bool,
          minimum_cases: int = MINIMUM_CASES) -> tuple[float, str]:
    """The fitted value and why it was chosen."""
    n = len(cases)
    if n < minimum_cases:
        return current, f"kept {current}: too few labelled cases ({n}; at least {minimum_cases} are needed)"
    scored = [(t, *wrong_reads(cases, t)) for t in candidates()]
    if forbid_false_yes:
        scored = [s for s in scored if s[1] == 0]
        if not scored:
            return current, f"kept {current}: every candidate lets a labelled no read as yes ({n} labelled cases)"
    best = min(scored, key=lambda s: (s[1] + s[2], -s[0]))
    t, false_yes, missed = best
    rule = "; no labelled no may read as yes" if forbid_false_yes else ""
    return t, (f"{t}: fewest wrong reads over {n} labelled cases ({false_yes + missed} wrong: {false_yes} labelled no "
               f"read as yes, {missed} labelled yes read as no); a tie goes to the higher value{rule}")


# ---- which reading feeds which threshold ------------------------------------------------------------------------------

def labelled_from(line: Mapping) -> list[Labelled]:
    """The labelled cases one recorded reading gives, one per threshold it feeds."""
    field, expected, p, value = line["field"], line["expected"], line.get("p"), line.get("value")
    p = 0.0 if p is None else float(p)  # a refused yes/no reads as no in the app
    if isinstance(expected, bool):
        return [] if field.split(":", 1)[0] in FILTER_GROUPS else [Labelled("yes", p, expected)]
    out = [Labelled("pick_confident", p, value == expected)]
    if field == "email_offer_reply":
        out.append(Labelled("send_email_min", p if value == "send" else 0.0, expected == "send"))
    if field == "answer_wanted" and line.get("calm_single_question"):
        out.append(Labelled("prepared_reply_min", p, value == expected))
    return out


def readings_of(form: Form, fields: Iterable[str],
                asked: Collection[str] | None = None) -> dict[str, tuple[float | None, str | None]]:
    """(p, picked value) of each named question in a filled form. A question the form left empty (None), or one not
    in `asked` when that is given, is left out."""
    groups = {"claim_type": form.claim_types, "claim_status": form.claim_statuses, "month": form.months,
              "year": form.years, "figure": form.figures}
    out: dict[str, tuple[float | None, str | None]] = {}
    for field in fields:
        if asked is not None and field not in asked:
            continue
        group, _, key = field.partition(":")
        if key and group in groups:
            answer = groups[group].get(int(key) if group in ("month", "year") else key)
        elif key and group == "shaped_need":
            i = int(key)
            answer = form.shaped_need[i] if i < len(form.shaped_need) else None
        else:
            answer = getattr(form, field, None)
        if isinstance(answer, YesNo):
            out[field] = (answer.p, None)
        elif isinstance(answer, Pick):
            out[field] = (answer.p, answer.value)
    return out


# ---- recording --------------------------------------------------------------------------------------------------------

def request_for(case: Mapping, vocabulary: Mapping[str, Iterable[str]], answer_names: Sequence[str],
                examples: Mapping[str, tuple[str, ...]], year: int) -> FormRequest:
    """The form request for one labelled case, built the way the app builds one: the data-wide vocabulary, the answer
    kinds and follow-up topics, the last five calendar years, and the example phrasings."""
    ctx = case.get("context") or {}
    return FormRequest(
        message=case["message"],
        recent_caller_messages=tuple(ctx.get("recent_caller_messages") or ()),
        previous_reply=ctx.get("previous_reply"), pending=ctx.get("pending"),
        state=State(ctx.get("state", "VERIFY_ID")), verified=bool(ctx.get("verified", False)),
        claim_types=tuple(sorted(vocabulary.get("case_type", ()))),
        claim_statuses=tuple(sorted(vocabulary.get("status", ()))),
        years=tuple(range(year - 4, year + 1)),
        needs=tuple(sorted(vocabulary.get("need", ()))),
        relationships=tuple(sorted(vocabulary.get("relationship", ()))),
        answers=tuple((a, a.replace("_", " ")) for a in answer_names),
        examples=dict(examples))


def _calm_single_question(expect: Mapping) -> bool:
    return expect.get("mood") == "calm" and expect.get("multi_question") is False


def read_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def record() -> int:
    from claims_agent.engine import ANSWER_KINDS
    from claims_agent.form.decisions_filler import DecisionsFiller
    from claims_agent.loader.read import FALLBACK_TOPIC, read_folder
    from claims_agent.models.client import make_client

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set: nothing was recorded")
        return 1
    s = load_settings()
    data = read_folder(ROOT / s.data.folder)
    followups = sorted((f for f in data.followups if f.enabled and f.topic != FALLBACK_TOPIC), key=lambda f: f.position)
    answer_names = (*ANSWER_KINDS, *(f.topic for f in followups))
    known = known_choices(data.vocabulary.get("need", ()), data.vocabulary.get("relationship", ()), answer_names)
    examples = with_approved({f"answer_wanted:{f.topic}": f.examples for f in followups if f.examples},
                             load_examples(s, None, set(known), known))  # the file's examples, as the app reads them
    filler = DecisionsFiller(make_client(), s.models.fast, retries=s.limits.model_retries,
                             timeout=s.limits.model_timeout_seconds)
    cases = read_cases()
    gate = asyncio.Semaphore(4)

    async def one(i: int, case: dict) -> list[dict]:
        req = request_for(case, data.vocabulary, answer_names, examples, s.today.date.year)
        async with gate:
            try:
                form = await filler.fill(req)
            except Exception as e:  # noqa: BLE001  one failed case is reported and left out
                print(f"case {i}: not recorded ({type(e).__name__})", file=sys.stderr)
                return []
        expect = case["expect"]
        got = readings_of(form, expect, {q.name for q in build_questions(req)})
        missing = [f for f in expect if f not in got]
        if missing:
            print(f"case {i}: not asked in this context, left out: {missing}", file=sys.stderr)
        return [{"case": i, "from": case.get("from"), "field": f, "expected": expect[f], "p": got[f][0],
                 "value": got[f][1], "calm_single_question": _calm_single_question(expect)}
                for f in expect if f in got]

    lines = [line for chunk in await asyncio.gather(*(one(i, c) for i, c in enumerate(cases))) for line in chunk]
    PROBABILITIES.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    recorded = len({line["case"] for line in lines})
    print(f"recorded {len(lines)} labelled readings from {recorded} of {len(cases)} cases with {s.models.fast} "
          f"into {PROBABILITIES.relative_to(ROOT)}")
    return 0 if recorded == len(cases) else 1


# ---- fitting ----------------------------------------------------------------------------------------------------------

def fitted_toml(current: Thresholds, fitted: Mapping[str, tuple[float, str]], *, source: str) -> str:
    lines = [f"# Thresholds fitted by scripts/calibrate_thresholds.py --fit from {source} on {date.today().isoformat()}.",
             "# Not in use until applied with --apply (which copies this file over thresholds.toml).", ""]
    for name in Thresholds.model_fields:
        value = getattr(current, name)
        if name in fitted:
            value, why = fitted[name]
        else:
            why = f"kept {value}: no labelled reading feeds it"
        lines += [f"# {why}", f"{name} = {value}"]
    return "\n".join(lines) + "\n"


def _shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def fit(probabilities: Path = PROBABILITIES, fitted_file: Path = FITTED) -> int:
    if not probabilities.exists():
        print(f"{_shown(probabilities)} does not exist: run --record first")
        return 1
    readings = [json.loads(line) for line in probabilities.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_threshold: dict[str, list[Labelled]] = {name: [] for name in Thresholds.model_fields}
    for line in readings:
        for case in labelled_from(line):
            if case.threshold not in by_threshold:
                raise ValueError(f"a reading of {line['field']} feeds {case.threshold}, which thresholds.toml lacks")
            by_threshold[case.threshold].append(case)
    current = load_thresholds()
    fitted = {name: sweep(cases, current=getattr(current, name), forbid_false_yes=name in NO_FALSE_YES)
              for name, cases in by_threshold.items() if cases}
    fitted_file.write_text(fitted_toml(current, fitted, source=_shown(probabilities)), encoding="utf-8")
    print(f"wrote {_shown(fitted_file)} (thresholds.toml is unchanged; --apply copies it over)\n")
    print("| threshold | current | fitted | labelled cases | wrong at current | wrong at fitted | how |")
    print("|---|---|---|---|---|---|---|")
    for name in Thresholds.model_fields:
        now = getattr(current, name)
        if name not in fitted:
            print(f"| {name} | {now} | {now} | - | - | - | not fitted |")
            continue
        value, why = fitted[name]
        cases = by_threshold[name]
        at_now, at_fit = sum(wrong_reads(cases, now)), sum(wrong_reads(cases, value))
        print(f"| {name} | {now} | {value} | {len(cases)} | {at_now} | {at_fit} | {why} |")
    return 0


def apply() -> int:
    if not FITTED.exists():
        print(f"{FITTED.relative_to(ROOT)} does not exist: run --fit first")
        return 1
    Thresholds.model_validate(tomllib.loads(FITTED.read_text(encoding="utf-8")))  # refuse a broken file
    shutil.copyfile(FITTED, THRESHOLDS)
    print(f"copied {FITTED.relative_to(ROOT)} over {THRESHOLDS.relative_to(ROOT)}; restart the app to use it")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--record", action="store_true", help="read every labelled case with the real model")
    step.add_argument("--fit", action="store_true", help="fit the thresholds into evals/thresholds.fitted.toml")
    step.add_argument("--apply", action="store_true", help="copy evals/thresholds.fitted.toml over thresholds.toml")
    args = parser.parse_args(argv)
    if args.record:
        return asyncio.run(record())
    return fit() if args.fit else apply()


if __name__ == "__main__":
    sys.exit(main())
