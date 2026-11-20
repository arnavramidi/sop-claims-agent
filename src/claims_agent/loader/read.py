"""Read the six data files into validated rows plus a two-level load report.

Pure: no database, never raises on content. A bad whole row is set aside (a `row` entry); a bad value
inside an otherwise usable row is left empty (a `field` entry); a missing or unreadable file is one
`file` entry and counts as empty; a follow-up whose template uses an unknown blank is disabled (a
`followup` entry).
"""

import json
import keyword
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from string import Formatter
from typing import Any, Literal

CUSTOMERS_FILE = "policyholders.json"
CLAIMS_FILE = "claims.json"
REPRESENTATIVES_FILE = "representatives.json"
CONSENT_FILE = "consent_scenarios.json"
GUIDANCE_FILE = "required_document_guideline.json"
GLOSSARY_FILE = "claim_schema.json"

FILES = (CUSTOMERS_FILE, CLAIMS_FILE, REPRESENTATIVES_FILE, CONSENT_FILE, GUIDANCE_FILE, GLOSSARY_FILE)
LEVELS = ("file", "row", "field", "followup")
SETTINGS_KIND = "claim_followup_settings"
FALLBACK_TOPIC = "fallback"
ALIAS_KINDS = {"name_aliases": "name", "phone_aliases": "phone", "email_aliases": "email"}
MONEY_FIELDS = ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee")
# Blanks every template may use besides the keys of claim_followup_settings.
BASE_BLANKS = frozenset({"case_id", "documents"})

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.ASCII)
_LAST4 = re.compile(r"^\d{4}$", re.ASCII)
_PHONE = re.compile(r"^\+?\d{7,15}$", re.ASCII)
_PHONE_SEPARATORS = re.compile(r"[\s\-.()]", re.ASCII)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_MONEY = Decimal("9999999999.99")
# A data file larger than this is treated as unreadable (a thousand claims is well under 1 MB).
MAX_FILE_BYTES = 5_000_000
# Longest identifying or indexed value (ids, status, case type, policy number, names, topics, keys; the
# database's indexed columns refuse values near 2.7 KB) and longest free text (summaries, guidance text).
MAX_ID_CHARS = 200
MAX_TEXT_CHARS = 10_000
GATED_REASON = "a value in this field cannot be stored (a character, number or length the database refuses)"
# Deepest nesting a file may have, counted in containers from the top of the file (the starter files reach
# 4). Postgres reads jsonb recursively and refuses a value some thousands of levels deep (stack depth,
# 54001) while Python's json reads it happily, so the gate cuts off anything deeper than this.
MAX_DEPTH = 16
DEEP_REASON = f"a value is nested more than {MAX_DEPTH} levels deep and cannot be stored"

# The load report is readable by the server's account, so no entry may carry a value taken from the
# customers, aliases or representatives data: a reason is fixed wording that names the kind of problem,
# and a ref is a party id, a case id, a topic, or the row's position in its file ("#3", counted in the
# file itself, rows set aside included, so a person can find the row).


@dataclass(frozen=True)
class Origin:
    """Where a row came from: its file, the section or field inside it, and its position there ("#3").
    Never a value from the row, so the writer can report a row the database refused by its origin alone."""
    file: str | None
    field: str | None
    ref: str | None


# Every row carries its origin. It takes no part in equality and is not a column.
def _origin():
    return field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ReportEntry:
    level: Literal["file", "row", "field", "followup"]
    file: str | None
    ref: str | None
    field: str | None
    reason: str


@dataclass(frozen=True)
class CustomerRow:
    party_id: str
    name: str
    policy_number: str | None
    dob: date | None
    id_type: str | None
    id_last4: str | None
    phone: str | None
    email: str | None
    origin: Origin | None = _origin()


@dataclass(frozen=True)
class AliasRow:
    party_id: str
    kind: Literal["name", "phone", "email"]
    value: str
    origin: Origin | None = _origin()


@dataclass(frozen=True)
class RepresentativeRow:
    party_id: str
    rep_name: str
    relationship: str | None
    origin: Origin | None = _origin()


@dataclass(frozen=True)
class ClaimRow:
    case_id: str
    party_id: str
    case_type: str | None
    created_at: date | None
    status: str
    summary: str | None
    denial_reason: str | None
    appeal_deadline: date | None
    expected_reimbursement_amount: Decimal | None
    allowed_max_amount: Decimal | None
    net_pay: Decimal | None
    net_fee: Decimal | None
    origin: Origin | None = _origin()


@dataclass(frozen=True)
class DocumentRow:
    case_id: str
    party_id: str
    position: int
    name: str
    origin: Origin | None = _origin()


@dataclass(frozen=True)
class GuidanceRow:
    kind: str
    key: str | None
    body: Any
    origin: Origin | None = _origin()


@dataclass(frozen=True)
class FollowupRow:
    topic: str
    needs: tuple[str, ...]
    requires_documents: bool
    examples: tuple[str, ...]
    template: str
    position: int
    enabled: bool
    origin: Origin | None = _origin()


@dataclass(frozen=True)
class ConsentScriptRow:
    name: str
    status_sequence: tuple[str, ...]
    origin: Origin | None = _origin()


@dataclass
class LoadResult:
    customers: list[CustomerRow] = field(default_factory=list)
    aliases: list[AliasRow] = field(default_factory=list)
    representatives: list[RepresentativeRow] = field(default_factory=list)
    claims: list[ClaimRow] = field(default_factory=list)
    documents: list[DocumentRow] = field(default_factory=list)
    guidance: list[GuidanceRow] = field(default_factory=list)
    followups: list[FollowupRow] = field(default_factory=list)
    consent_scripts: list[ConsentScriptRow] = field(default_factory=list)
    vocabulary: dict[str, set[str]] = field(default_factory=dict)
    report: list[ReportEntry] = field(default_factory=list)


class _Reporter:
    def __init__(self, file: str, report: list[ReportEntry]) -> None:
        self.file, self.report = file, report

    def row(self, ref: str | None, reason: str) -> None:
        self.report.append(ReportEntry("row", self.file, ref, None, reason))

    def field(self, ref: str | None, name: str, reason: str) -> None:
        self.report.append(ReportEntry("field", self.file, ref, name, reason))


class _Unstorable:
    """Stands where the gate removed a value that cannot be stored. Never a str, a number or a container,
    so every reader treats it as a missing or bad value, and `_all_storable` refuses a body holding it.
    `reason` is the fixed wording for the field or entry that held it, `file_reason` for the file."""

    def __init__(self, reason: str, file_reason: str) -> None:
        self.reason, self.file_reason = reason, file_reason

    def __repr__(self) -> str:
        return "<unstorable>"


UNSTORABLE = _Unstorable(GATED_REASON, "file has values that cannot be stored; they were ignored")
TOO_DEEP = _Unstorable(DEEP_REASON, f"file has values nested more than {MAX_DEPTH} levels deep; they were ignored")


class _Object(dict):
    """A parsed JSON object that knows each key's position among the object's entries as written in the
    file (1-based; for a repeated key, the occurrence whose value is used, the last). Counted at parse time,
    before the gate drops anything, so a ref is the position a person counts in the file."""

    positions: dict[str, int]


def _object(pairs: list[tuple[str, Any]]) -> _Object:
    obj = _Object(pairs)
    obj.positions = {key: i for i, (key, _) in enumerate(pairs, start=1)}
    return obj


def _entries(obj: dict):
    """(position in the file as "#n", key, value) for each entry of a parsed JSON object."""
    positions = getattr(obj, "positions", {})
    for i, (key, value) in enumerate(obj.items(), start=1):
        yield f"#{positions.get(key, i)}", key, value


def _storable_leaf(value: Any) -> bool:
    """Text Postgres can hold (no NUL, UTF-8 encodable) and numbers that are finite (jsonb refuses nan and inf)."""
    if isinstance(value, str):
        if "\x00" in value or len(value) > MAX_TEXT_CHARS:
            return False
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _gate(root: Any) -> set[_Unstorable]:
    """THE gate: the one place every value read from a file passes before any reader sees it.

    Walks the parsed JSON (iteratively, so depth is no risk here), replaces each leaf that cannot be stored
    with UNSTORABLE, each container nested deeper than MAX_DEPTH with TOO_DEEP, and drops each key that
    cannot be stored. Nothing deeper than MAX_DEPTH survives it. Returns the markers it used."""
    used: set[_Unstorable] = set()
    stack = [(root, 1)]   # the top of the file is depth 1
    while stack:
        node, depth = stack.pop()
        for key, value in list(node.items()) if isinstance(node, dict) else list(enumerate(node)):
            if isinstance(node, dict) and not _storable_leaf(key):
                del node[key]
                used.add(UNSTORABLE)
            elif isinstance(value, (dict, list)):
                if depth >= MAX_DEPTH:
                    node[key] = TOO_DEEP
                    used.add(TOO_DEEP)
                else:
                    stack.append((value, depth + 1))
            elif not _storable_leaf(value):
                node[key] = UNSTORABLE
                used.add(UNSTORABLE)
    return used


def _unstorable(value: Any) -> _Unstorable | None:
    """The first marker the gate left anywhere in a value, or None. Iterative: depth is no risk."""
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, _Unstorable):
            return v
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return None


def _all_storable(value: Any) -> bool:
    """True when a body holds nothing the gate removed."""
    return _unstorable(value) is None


def _entry_reason(value: Any, default: str) -> str:
    """Why a whole entry holding `value` was set aside: the depth wording when it was too deep."""
    return DEEP_REASON if _unstorable(value) is TOO_DEEP else default


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() and len(value.strip()) <= MAX_ID_CHARS:
        return value.strip()
    return None


def _clean(rep: _Reporter, ref: str, name: str, raw: dict, check) -> Any:
    """The checked value of an optional field; empty (with a `field` entry) when present but bad."""
    value = raw.get(name)
    if value is None or value == "":
        return None
    bad = _unstorable(value)
    if bad:
        rep.field(ref, name, bad.reason)
        return None
    try:
        return check(value)
    except (ValueError, TypeError, InvalidOperation) as e:
        rep.field(ref, name, str(e) or f"not a valid {name}")
        return None


def _date(value: Any) -> date:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ValueError("not a date written YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError("not a real calendar date") from None


def _money(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("not a number")
    try:
        amount = Decimal(str(value).strip())
        amount = amount.quantize(Decimal("0.01"))
    except InvalidOperation:
        raise ValueError("not a number") from None
    if not amount.is_finite():
        raise ValueError("not a number")
    if abs(amount) > _MAX_MONEY:
        raise ValueError("amount too large")
    return amount


def _last4(value: Any) -> str:
    if not isinstance(value, str) or not _LAST4.fullmatch(value):
        raise ValueError("not four digits")
    return value


def _phone(value: Any) -> str:
    """One canonical form: an optional leading +, then digits only (the database function compares digits)."""
    if not isinstance(value, str):
        raise ValueError("not a phone number")
    canonical = _PHONE_SEPARATORS.sub("", value)
    if not _PHONE.match(canonical):
        raise ValueError("not a phone number")
    return canonical


def _email(value: Any) -> str:
    if not isinstance(value, str) or len(value) > MAX_ID_CHARS or not _EMAIL.match(value.strip()):
        raise ValueError("not an email address")
    return value.strip()


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("not text")
    if len(value.strip()) > MAX_ID_CHARS:
        raise ValueError("too long")
    return value.strip()


def _long_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("not text")
    return value.strip()


def _load_json(folder: Path, name: str, expect: type, report: list[ReportEntry]) -> Any:
    """The parsed file, or an empty value of the expected shape (with one `file` entry)."""
    empty = expect()
    path = folder / name
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            report.append(ReportEntry("file", name, None, None, "file is too large to read"))
            return empty
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except FileNotFoundError:
        report.append(ReportEntry("file", name, None, None, "file is missing"))
        return empty
    except Exception:  # unreadable, not JSON, too deeply nested: whatever the content, never stop the load
        report.append(ReportEntry("file", name, None, None, "file cannot be read as JSON"))
        return empty
    if not isinstance(data, expect):
        report.append(ReportEntry("file", name, None, None, f"expected {expect.__name__} at the top level"))
        return empty
    used = _gate(data)
    for marker in (UNSTORABLE, TOO_DEEP):
        if marker in used:
            report.append(ReportEntry("file", name, None, None, marker.file_reason))
    return data


def _rows(data: list, rep: _Reporter):
    for i, raw in enumerate(data, start=1):
        if isinstance(raw, dict):
            yield i, raw
        else:
            rep.row(f"#{i}", "row is not an object")


def _read_customers(folder: Path, report: list[ReportEntry]) -> tuple[list[CustomerRow], list[AliasRow]]:
    rep = _Reporter(CUSTOMERS_FILE, report)
    customers, aliases, seen = [], [], set()
    for i, raw in _rows(_load_json(folder, CUSTOMERS_FILE, list, report), rep):
        party_id, name = _text(raw.get("party_id")), _text(raw.get("name"))
        if party_id is None:
            rep.row(f"#{i}", "customer has no usable party_id")
            continue
        if name is None:
            rep.row(party_id, "customer has no usable name")
            continue
        if party_id in seen:
            rep.row(party_id, "party_id already used")
            continue
        seen.add(party_id)
        customers.append(CustomerRow(
            party_id=party_id, name=name,
            policy_number=_clean(rep, party_id, "policy_number", raw, _string),
            dob=_clean(rep, party_id, "dob", raw, _date),
            id_type=_clean(rep, party_id, "id_type", raw, _string),
            id_last4=_clean(rep, party_id, "id_last4", raw, _last4),
            phone=_clean(rep, party_id, "phone", raw, _phone),
            email=_clean(rep, party_id, "email", raw, _email),
            origin=Origin(CUSTOMERS_FILE, None, f"#{i}"),
        ))
        check = {"name": _string, "phone": _phone, "email": _email}
        for key, kind in ALIAS_KINDS.items():
            values = raw.get(key)
            for value in values if isinstance(values, list) else ([] if values is None else [values]):
                bad = _unstorable(value)
                if bad:
                    rep.field(party_id, key, bad.reason)
                    continue
                try:
                    aliases.append(AliasRow(party_id, kind, check[kind](value), Origin(CUSTOMERS_FILE, key, f"#{i}")))
                except (ValueError, TypeError):
                    rep.field(party_id, key, f"an alias is not a valid {kind}")
    return customers, aliases


def _read_representatives(folder: Path, parties: set[str], report: list[ReportEntry]) -> list[RepresentativeRow]:
    rep = _Reporter(REPRESENTATIVES_FILE, report)
    rows = []
    for i, raw in _rows(_load_json(folder, REPRESENTATIVES_FILE, list, report), rep):
        party_id, name = _text(raw.get("buyer_party_id")), _text(raw.get("rep_name"))
        if name is None:
            rep.row(f"#{i}", "representative has no usable rep_name")
        elif party_id not in parties:
            rep.row(f"#{i}", "buyer_party_id matches no customer")
        else:
            rows.append(RepresentativeRow(party_id, name, _clean(rep, f"#{i}", "relationship", raw, _string),
                                          Origin(REPRESENTATIVES_FILE, None, f"#{i}")))
    return rows


def _read_claims(folder: Path, parties: set[str], report: list[ReportEntry]) -> tuple[list[ClaimRow], list[DocumentRow]]:
    rep = _Reporter(CLAIMS_FILE, report)
    claims, documents, seen = [], [], set()
    for i, raw in _rows(_load_json(folder, CLAIMS_FILE, list, report), rep):
        case_id, status, party_id = _text(raw.get("case_id")), _text(raw.get("status")), _text(raw.get("party_id"))
        if case_id is None:
            rep.row(f"#{i}", "claim has no usable case_id")
            continue
        if status is None:
            rep.row(case_id, "claim has no usable status")
            continue
        if party_id not in parties:
            rep.row(case_id, "party_id matches no customer")
            continue
        if case_id in seen:
            rep.row(case_id, "case_id already used")
            continue
        seen.add(case_id)
        claims.append(ClaimRow(
            case_id=case_id, party_id=party_id, status=status,
            case_type=_clean(rep, case_id, "case_type", raw, _string),
            created_at=_clean(rep, case_id, "created_at", raw, _date),
            summary=_clean(rep, case_id, "summary", raw, _long_string),
            denial_reason=_clean(rep, case_id, "denial_reason", raw, _long_string),
            appeal_deadline=_clean(rep, case_id, "appeal_deadline", raw, _date),
            **{m: _clean(rep, case_id, m, raw, _money) for m in MONEY_FIELDS},
            origin=Origin(CLAIMS_FILE, None, f"#{i}"),
        ))
        needed = raw.get("documents_needed")
        if needed is not None and not isinstance(needed, list):
            rep.field(case_id, "documents_needed", "not a list")
            needed = []
        for name in needed or []:
            if _text(name) is None:
                rep.field(case_id, "documents_needed", "a document name is not usable text")
            else:
                documents.append(DocumentRow(case_id, party_id, 0, name.strip(),
                                             Origin(CLAIMS_FILE, "documents_needed", f"#{i}")))
    return claims, _numbered(documents)


def _numbered(documents: list[DocumentRow]) -> list[DocumentRow]:
    """Number each claim's documents 1, 2, ... in file order."""
    counts: dict[str, int] = {}
    out = []
    for d in documents:
        counts[d.case_id] = counts.get(d.case_id, 0) + 1
        out.append(replace(d, position=counts[d.case_id]))
    return out


def _read_consent(folder: Path, report: list[ReportEntry]) -> list[ConsentScriptRow]:
    rep = _Reporter(CONSENT_FILE, report)
    rows = []
    for at, name, raw in _entries(_load_json(folder, CONSENT_FILE, dict, report)):
        sequence = raw.get("status_sequence") if isinstance(raw, dict) else None
        if len(name) > MAX_ID_CHARS:
            rep.row(at, "consent script name is too long")
        elif isinstance(sequence, list) and sequence and all(isinstance(s, str) and s for s in sequence):
            rows.append(ConsentScriptRow(name, tuple(sequence), Origin(CONSENT_FILE, None, at)))
        else:
            rep.row(name, "consent script has no status_sequence of words")
    return rows


# The derived things: what is computed from rows rather than read from a file. Each has one definition,
# used by the reader for its own result and by the writer on the rows the database actually stored.

def usable_blank(key: Any) -> bool:
    """A settings key can be a template blank only if it is a plain identifier and not a keyword (so a
    blank can never reach into an attribute or a position: `a.b`, `a[0]`, `0`, empty)."""
    return isinstance(key, str) and key.isidentifier() and not keyword.iskeyword(key)


def allowed_blanks(settings_keys: Iterable[Any]) -> set[str]:
    """The blanks a template may use: the base blanks plus the usable keys of the given settings rows."""
    return set(BASE_BLANKS) | {k for k in settings_keys if usable_blank(k)}


def template_problem(template: str, allowed: set[str]) -> str | None:
    """A template may hold only plain blanks that are exactly an allowed name: no attribute access, no
    indexing, no conversion, no format spec (so nothing can be evaluated or inflated when it is filled).
    A template with a problem is disabled, and a disabled template is never filled by anything."""
    try:
        for _, name, spec, conversion in Formatter().parse(template):
            if name is None:
                continue
            if name not in allowed:
                return "template uses a blank that is not allowed"
            if spec or conversion:
                return "template uses a format spec or conversion"
    except ValueError:
        return "template is not well formed"
    return None


def vocabulary(claims: Iterable[ClaimRow], representatives: Iterable[RepresentativeRow],
               followups: Iterable[FollowupRow]) -> dict[str, set[str]]:
    """Claim types and statuses of the claims, relationships of the representatives, needs of every follow-up."""
    claims = list(claims)
    return {
        "case_type": {c.case_type for c in claims if c.case_type},
        "status": {c.status for c in claims},
        "relationship": {p.relationship for p in representatives if p.relationship},
        "need": {n for f in followups for n in f.needs},
    }


def disabled_followup(f: FollowupRow, problem: str) -> ReportEntry:
    file = f.origin.file if f.origin else None
    return ReportEntry("followup", file, f.topic, "en", f"{problem}; follow-up disabled")


def _followup(raw: dict, position: int, origin: Origin, allowed: set[str],
              report: list[ReportEntry]) -> FollowupRow | None:
    topic, template = _text(raw.get("topic")), raw.get("en")
    if topic is None or not isinstance(template, str):
        report.append(ReportEntry("row", origin.file, origin.ref, origin.field, "follow-up has no topic or no English text"))
        return None
    problem = template_problem(template, allowed)

    def strings(key: str) -> tuple[str, ...]:
        values = raw.get(key)
        out = []
        for v in values if isinstance(values, list) else []:
            if isinstance(v, str) and v.strip() and v.strip() not in out:
                out.append(v.strip())
        return tuple(out)

    f = FollowupRow(topic, strings("intent_hints"), raw.get("requires_documents") is True, strings("match_any"),
                    template, position, problem is None, origin)
    if problem:
        report.append(disabled_followup(f, problem))
    return f


def _read_guidance(folder: Path, report: list[ReportEntry]) -> tuple[list[GuidanceRow], list[FollowupRow]]:
    rep = _Reporter(GUIDANCE_FILE, report)
    data = _load_json(folder, GUIDANCE_FILE, dict, report)

    def section(name: str, expect: type):
        value = data.get(name)
        if value is None:
            return expect()
        if not isinstance(value, expect):
            report.append(ReportEntry("file", GUIDANCE_FILE, None, name, f"section is not a {expect.__name__}; ignored"))
            return expect()
        return value

    rows: list[GuidanceRow] = []
    default = section("default_guidance", dict)
    if default and _all_storable(default):
        rows.append(GuidanceRow("default_guidance", None, default, Origin(GUIDANCE_FILE, "default_guidance", None)))
    elif default:
        rep.row("default_guidance", _entry_reason(default, "entry has text that cannot be stored"))
    for kind in ("case_type_guidance", "document_guidance", "document_alternative_guidance", SETTINGS_KIND):
        for at, key, body in _entries(section(kind, dict)):
            if len(key) > MAX_ID_CHARS:
                rep.row(kind, "entry key is too long")
            elif not isinstance(body, dict):
                rep.row(f"{kind}.{key}", _entry_reason(body, "entry is not an object"))
            elif not _all_storable(body):
                rep.row(f"{kind}.{key}", _entry_reason(body, "entry has a value that cannot be stored"))
            elif kind == SETTINGS_KIND and not usable_blank(key):
                rep.row(SETTINGS_KIND, "a setting key is not a plain identifier; ignored")
            else:
                rows.append(GuidanceRow(kind, key, body, Origin(GUIDANCE_FILE, kind, at)))
    # A blank comes only from a settings row that was accepted (usable_blank says which keys can be one).
    allowed = allowed_blanks(row.key for row in rows if row.kind == SETTINGS_KIND)

    followups = []
    for i, raw in _rows(section("claim_followup_guidance", list), rep):
        f = _followup(raw, len(followups) + 1, Origin(GUIDANCE_FILE, "claim_followup_guidance", f"#{i}"), allowed, report)
        if f:
            followups.append(f)
    fallback = section("claim_followup_fallback", dict)
    if fallback:
        f = _followup({**fallback, "topic": FALLBACK_TOPIC, "intent_hints": [], "requires_documents": True},
                      len(followups) + 1, Origin(GUIDANCE_FILE, "claim_followup_fallback", None), allowed, report)
        if f:
            followups.append(f)
    return rows, followups


def _read_glossary(folder: Path, report: list[ReportEntry]) -> list[GuidanceRow]:
    data = _load_json(folder, GLOSSARY_FILE, dict, report)
    fields = data.get("field_descriptions")
    if not isinstance(fields, dict):
        return []
    rep = _Reporter(GLOSSARY_FILE, report)
    rows = []
    for at, name, body in _entries(fields):
        if len(name) > MAX_ID_CHARS:
            report.append(ReportEntry("row", GLOSSARY_FILE, at, "field_descriptions", "entry key is too long"))
        elif isinstance(body, dict) and _all_storable(body):
            rows.append(GuidanceRow("field_description", name, body, Origin(GLOSSARY_FILE, "field_descriptions", at)))
        else:
            rep.row(f"field_description.{name}",
                    _entry_reason(body, "entry is not an object or has a value that cannot be stored"))
    return rows


def read_folder(folder: Path) -> LoadResult:
    folder = Path(folder)
    r = LoadResult()
    r.customers, r.aliases = _read_customers(folder, r.report)
    parties = {c.party_id for c in r.customers}
    r.representatives = _read_representatives(folder, parties, r.report)
    r.claims, r.documents = _read_claims(folder, parties, r.report)
    r.consent_scripts = _read_consent(folder, r.report)
    r.guidance, r.followups = _read_guidance(folder, r.report)
    r.guidance += _read_glossary(folder, r.report)
    r.vocabulary = vocabulary(r.claims, r.representatives, r.followups)
    return r
