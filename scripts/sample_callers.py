"""Write the sample-callers file and its readable page from the data files.

Run: python scripts/sample_callers.py
Writes data/starter/sample_callers.json and docs/sample-callers.md. A test checks that both are what this produces.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claims_agent import sample_callers  # noqa: E402
from claims_agent.settings import load_settings  # noqa: E402


def main() -> None:
    folder = ROOT / "data" / "starter"
    doc = sample_callers.build(folder, load_settings().today.date)
    (folder / sample_callers.FILE).write_text(sample_callers.to_json(doc), encoding="utf-8")
    (ROOT / "docs" / "sample-callers.md").write_text(sample_callers.to_markdown(doc), encoding="utf-8")


if __name__ == "__main__":
    main()
