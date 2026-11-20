"""Runs the live caller scenarios as a Braintrust experiment, and keeps the scenarios and the labelled cases as
Braintrust datasets a person can work from.

Each scenario (tests/live/scenarios, those waiting for a feature left out) is talked through with the real models
against the test database, exactly as the live suite does, and scored four ways:
  expectations_met  every plain expectation in the scenario file held (the scenario's own model-judged lines are left
                    to the tone score and not run here)
  no_leak           nothing about a claim was said to the caller, or sent to a model, before verification
  grounded          the share of replies in which every number and date is in that turn's facts
  tone              a larger-model judge: the replies' tone suits how the caller sounds (the one model-judged score)
Before anything is sent, every row passes the same masking as the traces (ID numbers and email addresses hidden),
and a row still holding an address after that is not uploaded. The scenarios' messages are sample data.

Needs BRAINTRUST_API_KEY and OPENAI_API_KEY, and costs one live-suite run plus one judge call per scenario. Run:
  docker compose --profile test run --rm tests python evals/run_braintrust.py
"""

import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests" / "live"), str(ROOT / "tests")]

from judge import judge  # noqa: E402
from runner import SCENARIOS, SentLog, load_scenario, real_deps, run_scenario  # noqa: E402

from claims_agent.examples import for_upload  # noqa: E402
from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart  # noqa: E402
from claims_agent.reply.check import check_reply  # noqa: E402
from claims_agent.settings import load_settings  # noqa: E402
from claims_agent.states import State  # noqa: E402

SCENARIO_DATASET = "live-scenarios"
LABELLED_DATASET = "labelled-cases"
LABELLED_CASES = ROOT / "evals" / "labelled_cases.jsonl"
TONE = ("Every agent reply's tone suits how the caller sounds: plain and calm with a calm caller; acknowledging and "
        "reassuring, without promising any outcome, with a worried, upset or angry caller; never curt, cold or "
        "dismissive.")
_LEAK_LINE = re.compile(r"before verification|what was sent to the models was not recorded")
_JUDGE_LINE = re.compile(r"^turn \d+: (no judge available for|judge): ")
_CLAIM_NUMBER = re.compile(r"\b[A-Za-z]{1,4}-?\d{3,}\b")
_NOT_IN_FACTS = ("number_not_in_facts", "month_not_in_facts")


# ---- the rows ---------------------------------------------------------------------------------------------------------

def scenario_rows() -> list[dict]:
    """One row per scenario that runs today, named as in the live suite."""
    rows = []
    for path in sorted(SCENARIOS.glob("*.yaml")):
        sc = load_scenario(path)
        if sc.waiting_for:
            continue
        rows.append({"id": sc.id, "input": {"scenario": sc.id, "caller_turns": [t.say for t in sc.turns]},
                     "metadata": {"group": sc.group, "about": sc.about, "dataset": sc.dataset}, "tags": [sc.group]})
    return rows


def labelled_rows() -> list[dict]:
    rows = []
    for line in LABELLED_CASES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        key = hashlib.sha256(json.dumps([case["message"], case.get("context")], sort_keys=True).encode()).hexdigest()
        rows.append({"id": f"case-{key[:16]}", "input": {"message": case["message"], "context": case.get("context")},
                     "expected": case["expect"], "metadata": {"from": case.get("from")}})
    return rows


def upload(braintrust, project: str, name: str, rows: list[dict]) -> tuple[int, int]:
    """Insert (or update, by id) every row that passes the masking; (sent, refused)."""
    dataset = braintrust.init_dataset(project=project, name=name)
    sent = refused = 0
    for row in rows:
        masked = for_upload(row)
        if masked is None:
            refused += 1
            continue
        dataset.insert(id=masked["id"], input=masked["input"], expected=masked.get("expected"),
                       metadata=masked.get("metadata"), tags=masked.get("tags"))
        sent += 1
    dataset.flush()
    return sent, refused


# ---- the scores -------------------------------------------------------------------------------------------------------

def _failures(output) -> list[str]:
    return list((output or {}).get("failures") or [])


def expectations_met(input, output, expected=None, **kwargs) -> dict:
    """1 when every plain expectation of the scenario held (leaks are no_leak's; model-judged lines are not run)."""
    missed = [f for f in _failures(output) if not _LEAK_LINE.search(f) and not _JUDGE_LINE.match(f)]
    return {"name": "expectations_met", "score": 0.0 if missed else 1.0, "metadata": {"missed": missed}}


def no_leak(input, output, expected=None, **kwargs) -> dict:
    """1 when nothing about a claim reached the caller or a model before verification."""
    leaks = [f for f in _failures(output) if _LEAK_LINE.search(f)]
    return {"name": "no_leak", "score": 0.0 if leaks else 1.0, "metadata": {"leaks": leaks}}


def grounded(input, output, expected=None, **kwargs) -> dict:
    """The share of replies in which every number and date is in that turn's facts (the reply check's own reading
    of numbers and dates; a claim number the caller typed counts as known)."""
    s = load_settings()
    turns = list((output or {}).get("transcript") or [])
    said, ungrounded = "", []
    for t in turns:
        said += "\n" + (t.get("caller") or "")
        facts = tuple(Fact(kind=str(f.get("kind", "")), text=str(f.get("text", "")), claim=f.get("claim"))
                      for f in t.get("facts") or [] if isinstance(f, dict))
        brief = ReplyBrief(state=State.PROCESS_CASE, parts=(ReplyPart("answer", facts),), mood="calm",
                           caller_message=t.get("caller") or "", previous_reply=None, english=True, verified=True,
                           caller_claim_numbers=tuple(_CLAIM_NUMBER.findall(said)))
        result = check_reply(t.get("reply") or "", brief, wary_status_words=(), data_statuses=(),
                             max_length=10**9)
        bad = [f for f in result.failures if f in _NOT_IN_FACTS]
        if bad:
            ungrounded.append({"turn": t.get("turn"), "failures": bad})
    score = 1.0 if not turns else (len(turns) - len(ungrounded)) / len(turns)
    return {"name": "grounded", "score": score, "metadata": {"ungrounded": ungrounded}}


def tone_scorer(client, larger: str):
    async def tone(input, output, expected=None, **kwargs) -> dict:
        turns = list((output or {}).get("transcript") or [])
        if not turns:
            return {"name": "tone", "score": 0.0, "metadata": {"why": "no conversation"}}
        passed, why = await judge(TONE, turns, client, larger)
        return {"name": "tone", "score": 1.0 if passed else 0.0, "metadata": {"why": why}}
    return tone


# ---- the run ----------------------------------------------------------------------------------------------------------

class Runs:
    """Talks one scenario at a time through the real models on the test database (each load empties it first)."""

    def __init__(self) -> None:
        from dbsupport import TestDatabase
        from claims_agent.loader.read import read_folder
        from claims_agent.loader.write import write
        from runner import dataset_folder

        self._database = TestDatabase()
        self._read, self._write, self._folder = read_folder, write, dataset_folder
        self._open: list = []
        self._known = {p.stem for p in SCENARIOS.glob("*.yaml")}

    async def task(self, input) -> dict:
        name = (input or {}).get("scenario")
        if name not in self._known:  # only a scenario file of this repository is ever run
            return {"passed": False, "failures": [f"not a scenario here: {name!r}"], "transcript": []}
        sent = SentLog()

        async def make_deps(dataset: str):
            await self._close_open()
            db = self._database.start_test()
            self._open.append(db)
            await self._write(self._read(self._folder(dataset)), db.loader_url)
            return real_deps(db.app, sent)

        try:
            result = await run_scenario(SCENARIOS / f"{name}.yaml", make_deps, sent=sent)
        finally:
            await self._close_open()
        return {"passed": result.passed, "failures": list(result.failures), "transcript": list(result.transcript)}

    async def _close_open(self) -> None:
        while self._open:
            await self._open.pop().close()

    def close(self) -> None:
        try:
            self._database.finish()
        finally:
            self._database.close()


async def main() -> int:
    missing = [k for k in ("BRAINTRUST_API_KEY", "OPENAI_API_KEY") if not (os.environ.get(k) or "").strip()]
    if missing:
        print(f"{' and '.join(missing)} not set: the eval talks to the real models and sends its results to "
              "Braintrust, so it needs both keys. Nothing was run.")
        return 2
    import braintrust

    from claims_agent.models.client import make_client
    from claims_agent.observability.turn_log import mask_for_trace

    s = load_settings()
    project = s.braintrust.project
    braintrust.set_masking_function(mask_for_trace)  # every record the library sends, masked the way traces are
    rows = scenario_rows()
    for name, these in ((SCENARIO_DATASET, rows), (LABELLED_DATASET, labelled_rows())):
        try:
            sent, refused = upload(braintrust, project, name, these)
            print(f"dataset {name}: {sent} rows sent" + (f", {refused} refused after masking" if refused else ""))
        except Exception as e:  # noqa: BLE001  the eval still runs; what Braintrust said is shown
            print(f"dataset {name} was not uploaded: {type(e).__name__}: {str(e)[:500]}")

    runs = Runs()
    client = make_client()
    try:
        result = await braintrust.EvalAsync(
            project, experiment_name=f"live scenarios {date.today().isoformat()}",
            data=[{k: r[k] for k in ("input", "metadata", "tags")} for r in rows], task=runs.task,
            scores=[expectations_met, no_leak, grounded, tone_scorer(client, s.models.larger)],
            metadata={"models": {"fast": s.models.fast, "text": s.models.text, "larger": s.models.larger}},
            max_concurrency=1)
    finally:
        runs.close()
    summary = result.summary
    print(f"\nexperiment: {summary.experiment_name}\n{summary.experiment_url or '(no link: not sent)'}")
    totals: dict[str, list[float]] = {}
    for r in result.results:
        for name, score in (r.scores or {}).items():
            if score is not None:
                totals.setdefault(name, []).append(score)
    for name in ("expectations_met", "no_leak", "grounded", "tone"):
        values = totals.get(name, [])
        mean = sum(values) / len(values) if values else float("nan")
        print(f"{name}: {mean:.2%} over {len(values)} scenarios")
    low = [(r.input.get("scenario"), {k: v for k, v in (r.scores or {}).items() if v is not None and v < 1})
           for r in result.results]
    for scenario, scores in low:
        if scores:
            print(f"  below 1: {scenario} {scores}")
    errors = [r for r in result.results if r.error is not None]
    if errors:
        print(f"{len(errors)} scenario(s) raised: " + ", ".join(str(r.input.get("scenario")) for r in errors))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
