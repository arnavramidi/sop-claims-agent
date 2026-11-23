"""Claim-record answers as facts. Code builds every fact from the claim row and the guidance data; a model
only words them, and the reply check rejects anything the facts do not say. So facts are written in the
canonical forms the check reads: ISO dates, amounts with two decimals, day counts as "N days", and a count the
reply would naturally state (how many documents) as a digit. Every fact names its claim number.

A grounded result can still contain a fact of kind `not_on_file` (for example a denial reason or deadline the
claim does not have), so a caller of `build` must read the fact kinds and not only `grounded`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from string import Formatter
from decimal import Decimal

from claims_agent.answers.documents import match_document, norm, plain_list
from claims_agent.db.claims import Claim
from claims_agent.db.guidance import Guidance
from claims_agent.loader.read import MONEY_FIELDS, FollowupRow
from claims_agent.reply.brief import Fact
from claims_agent.settings import Settings

COMPARE_CLAIMS = "compare_claims"
ANSWERS = ("status", "denial_reason", "documents_needed", "appeal_deadline", "amounts", "next_steps", "not_on_file",
           COMPARE_CLAIMS)


@dataclass(frozen=True)
class AnswerResult:
    facts: tuple[Fact, ...]
    grounded: bool = True
    reword_live: bool = False
    offer_person: bool = False


def _requirement_lines(claim: Claim, g: Guidance) -> list[tuple[str, str]]:
    """Each document that has requirement text (its own, else the default), with that text."""
    out = []
    for d in claim.documents:
        matched = match_document(d, g.document_text)
        text = g.document_text[matched] if matched else g.default_text
        if text:
            out.append((d, text))
    return out


def _replace(template: str, values: Mapping[str, str]) -> str | None:
    """Plain replacement of each named blank (nothing is evaluated, a value is never expanded again). None when
    a blank has no value or is empty, or the template is malformed."""
    out = []
    try:
        for literal, name, spec, conversion in Formatter().parse(template):
            out.append(literal)
            if name is None:
                continue
            if spec or conversion or not values.get(name):
                return None
            out.append(values[name])
    except ValueError:
        return None
    return "".join(out)


def _alternative_for(claim: Claim, g: Guidance, document: str | None) -> str:
    """Alternatives for one of THIS claim's documents; anything else gets the default."""
    own = next((d for d in claim.documents if document and norm(d) == norm(document)), None)
    matched = match_document(own, g.alternative_text) if own else None
    return g.alternative_text[matched] if matched else g.alternative_default


@dataclass(frozen=True)
class Rendered:
    body: str
    names_documents: bool                 # the body names the claim's documents
    named: tuple[str, ...] = ()           # the claim's documents the body names
    left_out: tuple[str, ...] = ()        # the claim's documents it leaves out (no requirement text on file)


def _names_documents(template: str) -> bool:
    try:
        return any(n == "documents" for _, n, _, _ in Formatter().parse(template))
    except ValueError:
        return False


def render(f: FollowupRow, claim: Claim, g: Guidance, s: Settings, document: str | None = None) -> Rendered | None:
    """The body of a follow-up for this claim, or None when it cannot be truthfully filled. The one place the
    blank values are worked out, used by `exists_for`, `available`, `next_followup` and `fill`, so they cannot
    disagree. The documents named and left out come from the same pass over the claim's own list as the body."""
    special = s.followups.special.get(f.topic)
    if special and special.per_document_alternatives:
        body = _alternative_for(claim, g, document)
        return Rendered(body, False) if body else None
    if special and special.documents_as_requirements:
        lines = _requirement_lines(claim, g)
        documents, named = "; ".join(f"the {d}: {t}" for d, t in lines), tuple(d for d, _ in lines)
    else:
        documents, named = plain_list(claim.documents), tuple(claim.documents)
    body = _replace(f.template, {**g.followup_settings, "case_id": claim.case_id, "documents": documents})
    if body is None:
        return None
    if not _names_documents(f.template):
        return Rendered(body, False)
    return Rendered(body, True, named, tuple(d for d in claim.documents if d not in named))


def exists_for(f: FollowupRow, claim: Claim, g: Guidance, s: Settings) -> bool:
    """A follow-up exists when it is enabled, its document requirement is met and every blank its template
    names has a value for this claim, so it is never offered and then not answerable."""
    if not f.enabled or (f.requires_documents and not claim.documents):
        return False
    return render(f, claim, g, s) is not None


def _answer_exists(answer: str, claim: Claim) -> bool:
    if answer == "denial_reason":
        return bool(claim.denial_reason)
    if answer == "documents_needed":
        return bool(claim.documents)
    if answer == "appeal_deadline":
        return claim.appeal_deadline is not None
    return answer in ANSWERS


def available(claim: Claim, g: Guidance, focus_size: int, s: Settings) -> list[str]:
    """Claim-record answers that exist for the claim, then the topics of follow-ups that exist for it.
    `focus_size` is the number of claims in focus; it adds a comparison across claims when there are two or more."""
    return [a for a in ANSWERS if _answer_exists(a, claim) and (a != COMPARE_CLAIMS or focus_size >= 2)] + [f.topic for f in g.followups if exists_for(f, claim, g, s)]


def opening(need: str, claim: Claim, s: Settings) -> list[str]:
    setting = s.needs.get(need)
    if setting is None:
        return ["status"]
    names = setting.opens_with_when_denied if claim.denial_reason and setting.opens_with_when_denied else setting.opens_with
    return [a for a in names if _answer_exists(a, claim)] or ["status"]


def _days(n: int) -> str:
    return "1 day" if n == 1 else f"{n} days"


def status_fact(claim: Claim) -> Fact:
    return Fact("status", f"Claim {claim.case_id} status: {claim.status}.", claim.case_id)


def deadline_fact(claim: Claim, today: date) -> tuple[Fact, bool]:
    """The deadline fact and whether it has passed. A deadline that is today is said to be today, and has not passed."""
    if claim.appeal_deadline is None:
        return Fact("not_on_file", f"Claim {claim.case_id} has no appeal deadline on file.", claim.case_id), False
    left = (claim.appeal_deadline - today).days
    if left < 0:
        text = (f"The appeal deadline for claim {claim.case_id} was {claim.appeal_deadline.isoformat()} "
                f"and has passed ({_days(-left)} ago).")
        return Fact("deadline", text, claim.case_id), True
    if left == 0:  # said as a day, not as a count: "0 days left" reads as if it had passed
        text = f"Appeal deadline for claim {claim.case_id}: {claim.appeal_deadline.isoformat()}, which is today."
        return Fact("deadline", text, claim.case_id), False
    soon = ", tomorrow" if left == 1 else ""
    text = f"Appeal deadline for claim {claim.case_id}: {claim.appeal_deadline.isoformat()} ({_days(left)} left{soon})."
    return Fact("deadline", text, claim.case_id), False


def documents_fact(claim: Claim, g: Guidance) -> Fact:
    if not claim.documents:
        return Fact("documents", f"No documents are needed for claim {claim.case_id}.", claim.case_id)
    text = f"Documents needed for claim {claim.case_id} ({len(claim.documents)}): {', '.join(claim.documents)}."
    advice = {norm(k): v for k, v in g.case_type_text.items()}.get(norm(claim.case_type or ""))
    if advice:
        text += f" Guidance for {claim.case_type} claims: {advice}"
    return Fact("documents", text, claim.case_id)


def _amount_facts(claim: Claim, g: Guidance, s: Settings, figures: Sequence[str] = ()) -> list[Fact]:
    """Each money figure with its meaning on file: the figures named in `figures` (in the data's order), or all of
    them when none is named."""
    in_progress = claim.status in s.claims.in_progress_statuses
    out = []
    for name in [m for m in MONEY_FIELDS if m in figures] or MONEY_FIELDS:
        value: Decimal | None = getattr(claim, name)
        label = name.replace("_", " ")
        if value is None:
            shown = "unavailable"
        elif name == "net_pay" and value == 0 and in_progress:
            shown = "not paid yet"
        else:
            shown = f"{value:.2f}"
        text = f"Claim {claim.case_id} {label}: {shown}."
        meaning = g.glossary.get(name)
        if meaning:
            text += f" Meaning of {label}: {meaning}"
        out.append(Fact("amount", text, claim.case_id))
    return out


def _one(answer: str, claim: Claim, g: Guidance, s: Settings, today: date,
         figures: Sequence[str] = ()) -> tuple[list[Fact], bool]:
    """The facts for one claim and whether the answer is one that offers a person."""
    if answer == "status":
        return [status_fact(claim)], False
    if answer == "denial_reason":
        if claim.denial_reason:
            return [Fact("denial_reason", f"Denial reason for claim {claim.case_id}: {claim.denial_reason}", claim.case_id)], False
        return [Fact("not_on_file", f"Claim {claim.case_id} has no denial reason on file.", claim.case_id)], False
    if answer == "documents_needed":
        return [documents_fact(claim, g)], False
    if answer == "appeal_deadline":
        fact, passed = deadline_fact(claim, today)
        return [fact], passed
    if answer == "amounts":
        return _amount_facts(claim, g, s, figures), False
    if answer == "next_steps":
        if claim.documents or claim.appeal_deadline:
            facts, passed = [], False
            if claim.documents:
                facts.append(documents_fact(claim, g))
            if claim.appeal_deadline:
                fact, passed = deadline_fact(claim, today)
                facts.append(fact)
            return facts, passed
        if claim.status in s.claims.in_progress_statuses:
            text = f"Next steps for claim {claim.case_id}: nothing is requested right now."
        else:
            text = f"Next steps for claim {claim.case_id}: nothing further is needed."
        return [Fact("next_steps", text, claim.case_id)], False
    if answer == "not_on_file":
        return [Fact("not_on_file", f"That detail is not on file for claim {claim.case_id}.", claim.case_id)], True
    raise ValueError(f"unknown answer: {answer}")


def figure_words(name: str) -> str:
    return name.replace("_", " ")


# Said whenever an expected figure is totalled: it is what the claims are expected to pay, never a promise.
EXPECTED_NOTE = "This is an expected figure, not a guaranteed payment."


def _shown(claim: Claim, name: str, s: Settings) -> tuple[str, Decimal | None]:
    """How a claim's figure is stated, and its value when it is a real figure (None: unavailable or not paid yet)."""
    value: Decimal | None = getattr(claim, name)
    if value is None:
        return "unavailable", None
    if name == "net_pay" and value == 0 and claim.status in s.claims.in_progress_statuses:
        return "not paid yet", None
    return f"{value:.2f}", value


def compare_facts(claims: Sequence[Claim], g: Guidance, s: Settings) -> list[Fact]:
    """The four figures of each claim, each figure's total across the claims, and for two claims the difference per
    figure. Every number a reply could state is a fact. It says nothing about how the claims relate."""
    facts: list[Fact] = []
    for c in claims:
        for name in MONEY_FIELDS:
            facts.append(Fact("amount", f"Claim {c.case_id} {figure_words(name)}: {_shown(c, name, s)[0]}.", c.case_id))
    ids = " and ".join(c.case_id for c in claims)
    for name in MONEY_FIELDS:
        have = [getattr(c, name) for c in claims if getattr(c, name) is not None]
        if not have:
            continue
        note = f" {EXPECTED_NOTE}" if name == "expected_reimbursement_amount" else ""
        facts.append(Fact("total", f"Total {figure_words(name)} across {len(have)} of the {len(claims)} claims "
                                   f"({ids}): {sum(have, Decimal(0)):.2f}.{note}"))
    if len(claims) == 2:
        a, b = claims
        for name in MONEY_FIELDS:
            (_, va), (_, vb) = _shown(a, name, s), _shown(b, name, s)
            if va is not None and vb is not None:
                facts.append(Fact("difference", f"Difference in {figure_words(name)} between claim {a.case_id} and "
                                                f"claim {b.case_id}: {abs(va - vb):.2f}."))
    return facts


def build(answer: str, claims: Sequence[Claim], g: Guidance, s: Settings, today: date, *,
          document: str | None = None, figures: Sequence[str] = ()) -> AnswerResult:
    """The facts for one claim-record answer over every claim given, each fact naming its claim.
    `document` is for the follow-up answers; the claim-record answers ignore it. `figures` names the money figures
    asked about (the amounts answer then gives only those; none named gives all). `compare_claims` is built once over
    all the claims given."""
    if answer == COMPARE_CLAIMS:
        return AnswerResult(tuple(compare_facts(claims, g, s)), grounded=len(claims) >= 2)
    facts: list[Fact] = []
    offer = False
    for claim in claims:
        got, wants_person = _one(answer, claim, g, s, today, figures)
        facts += got
        offer = offer or wants_person
    return AnswerResult(tuple(facts), grounded=answer != "not_on_file", offer_person=offer)


# What each answer a caller can ask for gives, as the form's choices describe it: an answer named only by its key
# ("amounts") leaves the fast model guessing that it also says what each figure means.
ANSWER_MEANS = {
    "status": "the claim's status",
    "denial_reason": "why the claim was denied",
    "documents_needed": "which documents are still needed",
    "appeal_deadline": "the appeal deadline and how many days are left",
    "amounts": "the claim's money figures, and what each figure (expected, allowed, net pay, net fee) means",
    "next_steps": "what the caller should do next",
    COMPARE_CLAIMS: "how the claims being discussed compare",
}
