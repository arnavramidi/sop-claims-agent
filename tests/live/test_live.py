"""Scripted caller conversations against the real models, one case per scenario file. Each run costs money, so these
tests never run by default: `scripts/test.sh -m live` runs them all, `-k scn_<group>` one group. Every case writes its
outcome and transcript to tests/live/results/ (not tracked); tests/live/report.py turns those into the results page."""

import json
import os
from datetime import datetime, timezone

import pytest
from judge import judge
from runner import SCENARIOS, SentLog, load_scenario, real_deps, run_scenario

from claims_agent.models.client import make_client
from claims_agent.settings import load_settings

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="set OPENAI_API_KEY to run the live suite")]

RESULTS = SCENARIOS.parent / "results"


def _case(path):
    sc = load_scenario(path)
    marks = ([pytest.mark.xfail(reason=f"waiting for a feature: {sc.waiting_for}", run=False)]
             if sc.waiting_for else [])
    return pytest.param(path, id=f"scn_{sc.group}_{sc.id}", marks=marks)


def _write(result, sc) -> None:
    RESULTS.mkdir(exist_ok=True)
    s = load_settings()
    (RESULTS / f"{sc.id}.json").write_text(json.dumps({
        "id": sc.id, "group": sc.group, "about": sc.about, "passed": result.passed,
        "failures": list(result.failures), "transcript": list(result.transcript),
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {"fast": s.models.fast, "text": s.models.text, "larger": s.models.larger},
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


@pytest.mark.parametrize("path", [_case(p) for p in sorted(SCENARIOS.glob("*.yaml"))])
async def test_scenario(path, load_dataset):
    sent = SentLog()

    async def make_deps(dataset: str):
        return real_deps((await load_dataset(dataset)).app, sent)

    judge_client, larger = make_client(), load_settings().models.larger

    async def judge_with(rubric, transcript):
        return await judge(rubric, transcript, judge_client, larger)

    sc = load_scenario(path)
    result = await run_scenario(path, make_deps, sent=sent, judge_with=judge_with)
    _write(result, sc)
    assert result.passed, f"{sc.about}\n" + "\n".join(result.failures)
