"""What the inspector shows for one turn. The same payload is stored as the turn's record.

A full ID number is kept as typed in the stored message (Ruling 15) and masked here: every string taken from what the
caller stated or from memory passes through `mask_ids`.
"""

import json
import re
from collections.abc import Sequence

from claims_agent.reply.brief import ReplyPart
from claims_agent.reply.compose import Composed

# Nine or more digits in a row, also when typed with single spaces, dots or dashes between them ("123-45-4472").
_ID_RUN = re.compile(r"(?<!\d)\d(?:[ .\-]?\d){8,}(?!\d)")
KEEP_DIGITS = 4


def mask_ids(text: str) -> str:
    """A run of nine or more digits keeps its last four; every other digit in it becomes "*"."""
    def hide(m: re.Match) -> str:
        hidden = sum(c.isdigit() for c in m.group()) - KEEP_DIGITS
        out = []
        for c in m.group():
            if c.isdigit() and hidden > 0:
                out.append("*")
                hidden -= 1
            else:
                out.append(c)
        return "".join(out)
    return _ID_RUN.sub(hide, text)


def mask_all(value):
    """`mask_ids` on every string inside a JSON-like value (dict keys too)."""
    if isinstance(value, str):
        return mask_ids(value)
    if isinstance(value, dict):
        return {mask_all(k) if isinstance(k, str) else k: mask_all(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_all(v) for v in value]
    return value


def part_payload(p: ReplyPart) -> dict:
    return {"instruction": p.instruction,
            "facts": [{"kind": f.kind, "text": f.text, "claim": f.claim} for f in p.facts],
            "plain": p.plain,
            "pending": None if p.pending is None else p.pending.model_dump(mode="json")}


def inspector_payload(*, state_before: str, state_after: str, freedom: dict | None, filler: str, form: dict | None,
                      stated: dict, flags: Sequence[str], fired: Sequence[dict], parts: Sequence[ReplyPart],
                      composed: Composed, memory: dict, counts: dict, claims_visible: int, outbox: Sequence[dict],
                      handover_note: dict | None) -> dict:
    """The Stage 1 inspector payload: exactly these keys, and plain JSON (what is stored is what is shown)."""
    payload = {
        "state_before": state_before,
        "state_after": state_after,
        "freedom": freedom,
        "filler": filler,
        "form": form,
        "stated": mask_all(stated),
        "flags": list(flags),
        "fired": list(fired),
        "parts": [part_payload(p) for p in parts],
        "reply_source": composed.source,
        "checks": [{"passed": c.passed, "failures": list(c.failures), "leak": list(c.leak),
                    "invented": list(c.invented), "other": list(c.other)} for c in composed.checks],
        "memory": mask_all(memory),
        "counts": counts,
        "claims_visible": claims_visible,
        "outbox": [mask_all(o) for o in outbox],
        "handover_note": handover_note,
    }
    return json.loads(json.dumps(payload, default=str))
