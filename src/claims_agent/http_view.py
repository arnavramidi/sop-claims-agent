"""What an HTTP response may carry. Every route that returns conversation data or the operator's view passes its body
through `for_http`, so this one function decides what leaves the server; the database keeps the full record.

- Hosted mode (strangers): no operator panel at all (a send's and a restored turn's `inspector`) and no load report.
- Every mode: the identity match table (which typed details matched which customer) never leaves before the
  conversation is verified, and afterwards it names only the verified customer. Another customer's party id, and
  what a caller's typed details have in common with that customer's record, are never shown.
"""

from typing import Any

PANEL = "inspector"
LOAD_REPORT = "load_report"


def for_http(body: dict[str, Any], *, hosted: bool, verified_party: str | None = None) -> dict[str, Any]:
    """The body as it may be sent. `verified_party` is the customer the conversation is verified as now (None
    before verification). The input is never changed."""
    out = dict(body)
    if hosted:
        out.pop(PANEL, None)
        out.pop(LOAD_REPORT, None)
    elif isinstance(out.get(PANEL), dict):
        out[PANEL] = _panel(out[PANEL], verified_party)
    if isinstance(out.get("turns"), list):
        out["turns"] = [for_http(t, hosted=hosted, verified_party=verified_party) if isinstance(t, dict) else t
                        for t in out["turns"]]
    return out


def _panel(panel: dict[str, Any], verified_party: str | None) -> dict[str, Any]:
    memory = panel.get("memory")
    identity = memory.get("identity") if isinstance(memory, dict) else None
    if not isinstance(identity, dict) or "tally" not in identity:
        return panel
    identity = dict(identity)
    tally = identity.pop("tally")
    if verified_party is not None and isinstance(tally, dict) and verified_party in tally:
        identity["tally"] = {verified_party: tally[verified_party]}
    return {**panel, "memory": {**memory, "identity": identity}}
