"""Document names: matching a claim's document to the guidance's name for it, and wording a list of them."""

import re
from collections.abc import Iterable, Sequence


def norm(name: str) -> str:
    return " ".join(name.casefold().split())


def match_document(claim_document: str, guidance_names: Iterable[str]) -> str | None:
    """The one guidance name the claim's document name appears inside (whole words), or None when there is
    no such name or more than one (then the default text is used, never a guess)."""
    wanted = norm(claim_document)
    if not wanted:
        return None
    pattern = re.compile(rf"(?<!\w){re.escape(wanted)}(?!\w)")
    found = [n for n in guidance_names if pattern.search(norm(n))]
    return found[0] if len(found) == 1 else None


def plain_list(documents: Sequence[str]) -> str:
    """"the pathology report and the office note" (three or more: commas, then "and")."""
    named = [f"the {d}" for d in documents]
    if len(named) <= 1:
        return "".join(named)
    return ", ".join(named[:-1]) + " and " + named[-1]
