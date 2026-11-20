"""The summary email's text. Built by code from the claim rows (read fresh by the caller of this module) and the
answers recorded as given; no model writes or sees it. It carries no name, address or other personal detail, and a
typed full ID number in any text passes through `mask_ids`."""

from collections.abc import Sequence
from datetime import date

from claims_agent.answers.facts import ANSWERS, build, deadline_fact, documents_fact, status_fact
from claims_agent.db.claims import Claim
from claims_agent.db.guidance import Guidance
from claims_agent.inspect import mask_ids
from claims_agent.memory import RequestMemory
from claims_agent.settings import Settings

SUBJECT = "Summary of your claims conversation"
FULL_LIST = "The full list you asked for"
_NOT_FOUND = "not_on_file"


def _words(name: str) -> str:
    return name.replace("_", " ")


def _discussed(claim: Claim, answers: Sequence[str], g: Guidance, s: Settings, today: date) -> list[str]:
    """One line per answer given for the claim: a claim-record answer in the words of its facts, a follow-up topic
    by name."""
    lines = []
    for a in answers:
        if a in ANSWERS and a != _NOT_FOUND:
            lines += [f.text for f in build(a, [claim], g, s, today).facts]
        elif a != _NOT_FOUND:
            lines.append(f"Claim {claim.case_id}: we went over {_words(a)}.")
    return lines


def _outcome(claim: Claim) -> list[str]:
    lines = [status_fact(claim).text]
    if claim.denial_reason:
        lines.append(f"Reason given for claim {claim.case_id}: {claim.denial_reason}")
    return lines


def _follow_up(claim: Claim, g: Guidance, today: date) -> list[str]:
    lines = []
    if claim.documents:
        lines.append(documents_fact(claim, g).text)
    if claim.appeal_deadline:
        lines.append(deadline_fact(claim, today)[0].text)
    return lines or [f"Claim {claim.case_id}: nothing is requested from you right now."]


def _list_line(c: Claim) -> str:
    opened = c.created_at.isoformat() if c.created_at else "date not on file"
    return f"{c.case_id}, {c.case_type or 'claim'}, opened {opened}, status {c.status}"


def write_summary(claims: Sequence[Claim], request: RequestMemory, g: Guidance, s: Settings, today: date, *,
                  full_list: Sequence[Claim] = ()) -> tuple[str, str]:
    """(subject, body). `claims` are the claims discussed; the answers given for each are in `request.answered`."""
    sections: list[tuple[str, list[str]]] = [
        ("What we discussed", [line for c in claims for line in _discussed(c, request.answered.get(c.case_id, []),
                                                                         g, s, today)]
         or ["No specific claim was discussed."]),
        ("Status and outcome", [line for c in claims for line in _outcome(c)] or ["Nothing to report."]),
        ("Follow-up items and deadlines", [line for c in claims for line in _follow_up(c, g, today)]
         or ["Nothing is requested from you right now."]),
    ]
    if full_list:
        sections.append((FULL_LIST, [_list_line(c) for c in full_list]))
    body = "\n\n".join(f"{title}\n" + "\n".join(f"- {line}" for line in lines) for title, lines in sections)
    return SUBJECT, mask_ids(body)
