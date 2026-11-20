"""Builds docs/live-results.md and docs/live-known-failures.md from the latest live run (tests/live/results/).

Run after `scripts/test.sh -m live`:  docker compose --profile test run --rm tests python tests/live/report.py
A scenario waiting for a feature is listed as such. Each known failure keeps the "Likely cause:" line already written
for it in docs/live-known-failures.md, so the file can be rebuilt after every run.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runner import SCENARIOS, load_scenario  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).parent / "results"
RESULTS_PAGE = ROOT / "docs" / "live-results.md"
FAILURES_PAGE = ROOT / "docs" / "live-known-failures.md"
GROUPS = ("identity", "consent", "memory", "resolving", "processing", "arithmetic", "blocked", "language", "wrapup")
NOT_LOOKED_AT = "not looked at yet."


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _clip(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} characters in all)"


def _causes() -> dict[str, str]:
    if not FAILURES_PAGE.exists():
        return {}
    out, current = {}, None
    for line in FAILURES_PAGE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## `(\w+)`", line)
        if m:
            current = m.group(1)
        elif current and line.startswith("Likely cause: "):
            out[current] = line[len("Likely cause: "):]
    return out


def main() -> int:
    scenarios = sorted((load_scenario(p) for p in SCENARIOS.glob("*.yaml")),
                       key=lambda s: (GROUPS.index(s.group) if s.group in GROUPS else len(GROUPS), s.id))
    results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in RESULTS.glob("*.json")}
    ran = [results[s.id] for s in scenarios if s.id in results and not s.waiting_for]
    models = ran[0]["models"] if ran else {}
    dates = sorted(r["ran_at"][:10] for r in ran)
    passed = sum(1 for r in ran if r["passed"])
    failed = [s for s in scenarios if s.id in results and not s.waiting_for and not results[s.id]["passed"]]
    waiting = [s for s in scenarios if s.waiting_for]
    missing = [s for s in scenarios if not s.waiting_for and s.id not in results]

    lines = [
        "# Live results",
        "",
        "Scripted caller conversations run against the real models, each judged by plain checks on what the engine "
        "returned and on the reply text (a larger-model judge only for wording qualities). Run them with "
        "`scripts/test.sh -m live`; the scenarios are in `tests/live/scenarios/`.",
        "",
        f"- Run: {dates[0] if dates else 'not run'}" + (f" to {dates[-1]}" if dates and dates[-1] != dates[0] else ""),
        f"- Models: fast `{models.get('fast')}`, text `{models.get('text')}`, larger `{models.get('larger')}`",
        f"- Passed: **{passed} of {len(scenarios)}**; failed: {len(failed)} (see `docs/live-known-failures.md`); "
        f"waiting for a feature: {len(waiting)}" + (f"; not run: {len(missing)}" if missing else ""),
        "",
        "| Scenario | Group | Result | Note |",
        "|---|---|---|---|",
    ]
    for s in scenarios:
        if s.waiting_for:
            result, note = "waiting for a feature", f"needs {s.waiting_for}. {s.about}"
        elif s.id not in results:
            result, note = "not run", s.about
        elif results[s.id]["passed"]:
            result, note = "pass", s.about
        else:
            result, note = "fail", results[s.id]["failures"][0]
        lines.append(f"| `{s.id}` | {s.group} | {result} | {_cell(note)} |")
    RESULTS_PAGE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    causes = _causes()
    out = [
        "# Live known failures",
        "",
        "Scenarios that did not pass in the latest live run, with what failed, the conversation, and the likely "
        "cause. A scenario is never weakened to pass.",
        "",
    ]
    if not failed:
        out.append("None.")
    for s in failed:
        r = results[s.id]
        out += [f"## `{s.id}`", "", s.about, "", f"Likely cause: {causes.get(s.id, NOT_LOOKED_AT)}", "", "What failed:", ""]
        out += [f"- {_cell(f)}" for f in r["failures"]]
        out += ["", "Conversation:", ""]
        for t in r["transcript"]:
            out += [f"{t['turn']}. Caller: {_cell(_clip(t['caller']))}",
                    f"   Agent: {_cell(t['reply'])}",
                    f"   (state {t['state']}, verified {t['verified']}, claims in focus {t['focus']}, "
                    f"reply parts {t['parts']}, reply {t['reply_source']})"]
        out.append("")
    FAILURES_PAGE.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{passed} of {len(scenarios)} passed; {len(failed)} failed; {len(waiting)} waiting; {len(missing)} not run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
