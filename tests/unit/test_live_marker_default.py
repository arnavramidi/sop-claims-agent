"""The live suite calls the real models and is paid for, so an ordinary run must select none of it, and asking for
it with `-m live` must select all of it."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "tests" / "live" / "scenarios"


def _collect(*args: str) -> str:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/live", *args],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    return out.stdout + out.stderr


def _selected(text: str) -> int:
    return sum(1 for line in text.splitlines() if "::" in line)


def test_an_ordinary_run_selects_no_live_test():
    text = _collect()
    assert _selected(text) == 0, text
    assert "deselected" in text, text


def test_asking_for_live_selects_every_scenario():
    text = _collect("-m", "live")
    assert _selected(text) == len(list(SCENARIOS.glob("*.yaml"))) > 0, text
