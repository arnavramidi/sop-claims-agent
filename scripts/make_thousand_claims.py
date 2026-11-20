"""Write a data set in which one customer has a thousand claims, for checking that the agent stays correct and quick
when a customer has far more claims than a conversation can show.

Run: python scripts/make_thousand_claims.py <folder> [seed]
The folder gets the six data files the loader reads. The customer is Iris Novak (P1000, POL-1000) with claims
CL-10000 to CL-10999: types cycle healthcare, dental, auto; opened dates are spread evenly from 2022 to early 2026;
120 are denied (each with a reason, one document and an appeal deadline), 300 open and 580 closed. Everything is
derived from the claim's index and a fixed seed, so the same seed always gives the same files and the same totals.
The guidance, glossary and consent files are copied unchanged from data/starter.
"""

import json
import random
import shutil
import sys
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

COUNT = 1000
TYPES = ("healthcare", "dental", "auto")
FIRST_DAY = date(2022, 1, 1)
SPAN_DAYS = 1524  # the first day plus this many is 2026-03-05, the day the app treats as today
# Of every 50 consecutive claims: 6 denied, 15 open, 29 closed. 1000 claims therefore have 120, 300 and 580.
_DENIED_SLOTS = (3, 11, 19, 29, 37, 45)
_OPEN_SLOTS = tuple(s for s in range(50) if s not in _DENIED_SLOTS)[::3]
STARTER = Path(__file__).resolve().parent.parent / "data" / "starter"
COPIED = ("required_document_guideline.json", "claim_schema.json", "consent_scenarios.json")

CUSTOMER = {"party_id": "P1000", "name": "Iris Novak", "policy_number": "POL-1000", "dob": "1975-05-05",
            "id_type": "ssn_last4", "id_last4": "1000", "phone": "+15035551000", "email": "iris.novak@example.com"}


def _status(i: int) -> str:
    slot = i % 50
    return "denied" if slot in _DENIED_SLOTS else "open" if slot in _OPEN_SLOTS else "closed"


def _cents(c: int) -> str:
    return f"{Decimal(c) / 100:.2f}"


def _claims(seed: int) -> list[dict]:
    rng = random.Random(seed)
    claims = []
    for i in range(COUNT):
        kind, status = TYPES[i % len(TYPES)], _status(i)
        opened = FIRST_DAY + timedelta(days=i * SPAN_DAYS // COUNT)
        allowed = rng.randrange(10_000, 500_000)         # cents
        expected = allowed * 9 // 10
        claim = {"case_id": f"CL-{10000 + i}", "party_id": "P1000", "case_type": kind, "created_at": opened.isoformat(),
                 "status": status, "summary": f"{kind.capitalize()} claim {status}",
                 "expected_reimbursement_amount": _cents(expected), "allowed_max_amount": _cents(allowed),
                 "net_pay": _cents(expected if status == "closed" else 0), "net_fee": _cents(allowed)}
        if status == "denied":
            claim["denial_reason"] = f"the {kind} review file did not include the supporting report"
            claim["documents_needed"] = ["supporting report"]
            claim["appeal_deadline"] = (opened + timedelta(days=90)).isoformat()
        claims.append(claim)
    return claims


def expected(seed: int = 7) -> dict:
    """The numbers the data set must produce, worked out from the claims themselves (no database)."""
    claims = _claims(seed)
    denied = [c for c in claims if c["status"] == "denied"]
    return {
        "denied": len(denied),
        "open": sum(c["status"] == "open" for c in claims),
        "closed": sum(c["status"] == "closed" for c in claims),
        "total_net_pay": sum((Decimal(c["net_pay"]) for c in claims), Decimal("0.00")),
        "by_type": dict(Counter(c["case_type"] for c in denied)),
        "by_year": dict(Counter(int(c["created_at"][:4]) for c in denied)),
        "healthcare_denied_2025": sum(c["case_type"] == "healthcare" and c["created_at"].startswith("2025")
                                      for c in denied),
    }


def make(folder: Path, seed: int = 7) -> dict:
    """Write the six files into `folder` (created if needed) and return `expected(seed)`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        shutil.copyfile(STARTER / name, folder / name)
    (folder / "policyholders.json").write_text(json.dumps([CUSTOMER], indent=1))
    (folder / "representatives.json").write_text("[]")
    (folder / "claims.json").write_text(json.dumps(_claims(seed), indent=1))
    return expected(seed)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python scripts/make_thousand_claims.py <folder> [seed]")
    print(make(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) == 3 else 7))
