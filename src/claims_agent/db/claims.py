"""Claim filters and claim queries.

A filter is built from the form's answers (pure code, below). Every query runs as the server's account inside the
conversation's transaction, so the database's row rules return only the verified customer's claims (none before
verification); no query names a customer. The SQL text is constant: every value, lists included, is a parameter.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from psycopg import AsyncConnection

from claims_agent.form.fields import Form, ListedClaim
from claims_agent.form.read import confident_yes, picked, yes
from claims_agent.identity.normalize import claim_number_key
from claims_agent.loader.read import MONEY_FIELDS
from claims_agent.settings import Thresholds

# How many most-recent claims and upcoming deadlines a fixed-size overview names (the plan's limit of five listed).
OVERVIEW_SIZE = 5

RELATIVE = ("most_recent", "oldest", "this_year", "last_year")
# Ruling 10: when sureness ties, the condition dropped first comes first here.
RELAX_ORDER = ("months", "years", "relative", "statuses", "types")
NEED_CONDITION = {"denial_question": "has_denial_reason", "document_submission": "has_documents"}
SPLITS = ("case_type", "status", "year", "month")


@dataclass(frozen=True)
class ClaimFilter:
    types: frozenset[str] = frozenset()
    statuses: frozenset[str] = frozenset()
    months: frozenset[int] = frozenset()
    years: frozenset[int] = frozenset()
    relative: str = "none"
    has_denial_reason: bool = False
    has_documents: bool = False
    sureness: Mapping[str, float] = field(default_factory=dict)  # keys: types, statuses, months, years, relative

    def conditions(self) -> tuple[str, ...]:
        """The described conditions this filter sets (the need's implied conditions are not among them)."""
        return tuple(k for k in RELAX_ORDER if getattr(self, k) not in (frozenset(), "none"))

    def is_empty(self) -> bool:
        return not self.conditions() and not self.has_denial_reason and not self.has_documents


@dataclass(frozen=True)
class Claim:
    case_id: str
    case_type: str | None
    created_at: date | None
    status: str
    summary: str | None
    denial_reason: str | None
    documents: tuple[str, ...]
    appeal_deadline: date | None
    expected_reimbursement_amount: Decimal | None
    allowed_max_amount: Decimal | None
    net_pay: Decimal | None
    net_fee: Decimal | None
    updated_at: datetime


@dataclass(frozen=True)
class Overview:
    count: int
    by_type: dict[str, int]
    by_year: dict[int, int]
    most_recent: tuple[Claim, ...]
    upcoming_deadlines: tuple[Claim, ...]


# ---- the filter: pure ---------------------------------------------------------------------------------------------

def filter_from_form(form: Form, t: Thresholds) -> ClaimFilter:
    """A value joins only when the fast model is sure the caller named it (a confident yes, at or above the top of
    the unsure band). A value the caller names reads near 1; one the caller never said can still get a small yes, and
    every condition narrows what the caller sees, so a reading in the unsure band is never acted on: it would hide the
    caller's own claims ("all my claims" answered for two of four). With fewer conditions more claims match, and the
    caller is shown them or asked which one. "No year said" means no year condition. A condition's sureness is the
    lowest p among its values."""
    chosen: dict[str, frozenset] = {}
    sureness: dict[str, float] = {}
    answers = {"types": form.claim_types, "statuses": form.claim_statuses, "months": form.months, "years": form.years}
    if yes(form.no_year_said, t):
        answers["years"] = {}
    for key, values in answers.items():
        kept = {v: a.p for v, a in values.items() if confident_yes(a, t)}
        if kept:
            chosen[key] = frozenset(kept)
            sureness[key] = min(kept.values())
    relative = picked(form.relative_time, t)
    if relative in RELATIVE:
        sureness["relative"] = form.relative_time.p
    else:
        relative = "none"
    return ClaimFilter(**chosen, relative=relative, sureness=sureness)


def with_need(f: ClaimFilter, need: str | None) -> ClaimFilter:
    """The need implies a condition (Ruling 9: "denied" means the claim carries a denial reason)."""
    condition = NEED_CONDITION.get(need or "")
    return replace(f, **{condition: True}) if condition else f


def merge(previous: ClaimFilter | None, new: ClaimFilter, relation: str) -> ClaimFilter:
    """`narrowing` adds the new conditions to the old; `correcting` replaces only the conditions the new filter sets;
    `new_search` and `no_list_shown` use the new filter alone. A new filter that describes nothing never replaces a
    stored one, whatever the relation: a message with no claim content (or an unsure relation) cannot widen the search
    to every claim. Asking for all claims afresh is the caller's step to take, not a merge."""
    if previous is None:
        return new
    if not new.conditions():
        return previous
    if relation not in ("narrowing", "correcting"):
        return new
    values: dict = {}
    sureness = dict(previous.sureness)
    for key in RELAX_ORDER:
        old, now = getattr(previous, key), getattr(new, key)
        if now in (frozenset(), "none"):
            values[key] = old
            continue
        if relation == "narrowing" and key != "relative" and old:
            # Both must hold: the values both allow. A narrowing that leaves none cannot mean "neither": the new stands.
            values[key] = (old & now) or now
            if key in new.sureness:
                sureness[key] = min(new.sureness[key], sureness.get(key, new.sureness[key]))
        else:
            values[key] = now
            sureness.pop(key, None)
            if key in new.sureness:
                sureness[key] = new.sureness[key]
    return ClaimFilter(**values, has_denial_reason=previous.has_denial_reason or new.has_denial_reason,
                       has_documents=previous.has_documents or new.has_documents,
                       sureness={k: v for k, v in sureness.items() if k in values and values[k] not in (frozenset(), "none")})


def relax(f: ClaimFilter) -> ClaimFilter | None:
    """Drop the described condition the fast model was least sure of (Ruling 10); ties go month, year, relative time,
    status, type. A condition with no recorded sureness counts as sure. None when no condition is left."""
    present = f.conditions()
    if not present:
        return None
    weakest = min(present, key=lambda k: (f.sureness.get(k, 1.0), RELAX_ORDER.index(k)))
    return replace(f, **{weakest: "none" if weakest == "relative" else frozenset()},
                   sureness={k: v for k, v in f.sureness.items() if k != weakest})


def filter_to_dict(f: ClaimFilter) -> dict:
    """The shape memory keeps (the same keys as hints.claim_filter): only the conditions that are set."""
    out: dict = {k: sorted(getattr(f, k)) for k in ("types", "statuses", "months", "years") if getattr(f, k)}
    if f.relative != "none":
        out["relative"] = f.relative
    for flag in ("has_denial_reason", "has_documents"):
        if getattr(f, flag):
            out[flag] = True
    out["sureness"] = {k: v for k, v in f.sureness.items() if k in out}
    return out


def filter_from_dict(d: Mapping | None) -> ClaimFilter | None:
    if d is None:
        return None
    relative = d.get("relative", "none")
    return ClaimFilter(
        types=frozenset(d.get("types", ())), statuses=frozenset(d.get("statuses", ())),
        months=frozenset(int(m) for m in d.get("months", ())), years=frozenset(int(y) for y in d.get("years", ())),
        relative=relative if relative in RELATIVE else "none",
        has_denial_reason=d.get("has_denial_reason") is True, has_documents=d.get("has_documents") is True,
        sureness={k: float(v) for k, v in (d.get("sureness") or {}).items()})


# ---- the queries --------------------------------------------------------------------------------------------------

_DOCUMENTS = ("ARRAY(SELECT d.name FROM claim_documents d WHERE d.case_id = c.case_id AND d.party_id = c.party_id "
              "ORDER BY d.position)")

# One claim as a JSON object, read back by `_claim`; every query hands claims over in this one shape.
_CLAIM_JSON = """json_build_object(
  'case_id', case_id, 'case_type', case_type, 'created_at', created_at, 'status', status, 'summary', summary,
  'denial_reason', denial_reason, 'documents', documents, 'appeal_deadline', appeal_deadline,
  'expected_reimbursement_amount', expected_reimbursement_amount::text, 'allowed_max_amount', allowed_max_amount::text,
  'net_pay', net_pay::text, 'net_fee', net_fee::text, 'updated_at', updated_at)"""

# The claims a filter chooses, as a CTE named `chosen`. Each condition is off when its parameter is empty or false.
_CHOSEN = f"""WITH matched AS (
  SELECT c.case_id, c.case_type, c.created_at, c.status, c.summary, c.denial_reason, {_DOCUMENTS} AS documents,
         c.appeal_deadline, c.expected_reimbursement_amount, c.allowed_max_amount, c.net_pay, c.net_fee, c.updated_at,
         row_number() OVER (ORDER BY c.created_at DESC NULLS LAST, c.case_id) AS newest_rank,
         row_number() OVER (ORDER BY c.created_at ASC NULLS LAST, c.case_id) AS oldest_rank
  FROM claims c
  WHERE (cardinality(%(types)s::text[]) = 0 OR c.case_type = ANY(%(types)s::text[]))
    AND (cardinality(%(statuses)s::text[]) = 0 OR c.status = ANY(%(statuses)s::text[]))
    AND (cardinality(%(months)s::int[]) = 0 OR extract(month FROM c.created_at)::int = ANY(%(months)s::int[]))
    AND (cardinality(%(years)s::int[]) = 0 OR extract(year FROM c.created_at)::int = ANY(%(years)s::int[]))
    AND (%(relative)s::text <> 'this_year' OR extract(year FROM c.created_at)::int = %(this_year)s::int)
    AND (%(relative)s::text <> 'last_year' OR extract(year FROM c.created_at)::int = %(this_year)s::int - 1)
    AND (NOT %(denied)s::boolean OR nullif(btrim(c.denial_reason), '') IS NOT NULL)
    AND (NOT %(documents)s::boolean OR EXISTS (
          SELECT 1 FROM claim_documents d WHERE d.case_id = c.case_id AND d.party_id = c.party_id))
), chosen AS (
  SELECT * FROM matched
  WHERE (%(relative)s::text <> 'most_recent' OR newest_rank = 1) AND (%(relative)s::text <> 'oldest' OR oldest_rank = 1)
)
"""

_NEWEST_FIRST = "ORDER BY created_at DESC NULLS LAST, case_id"


def _params(f: ClaimFilter, today: date, **more) -> dict:
    return {"types": sorted(f.types), "statuses": sorted(f.statuses), "months": sorted(f.months),
            "years": sorted(f.years), "relative": f.relative, "this_year": today.year,
            "denied": f.has_denial_reason, "documents": f.has_documents, "today": today, **more}


def _money(v: str | None) -> Decimal | None:
    return None if v is None else Decimal(v)


def _day(v: str | None) -> date | None:
    return None if v is None else date.fromisoformat(v)


def _claim(j: Mapping) -> Claim:
    return Claim(
        case_id=j["case_id"], case_type=j["case_type"], created_at=_day(j["created_at"]), status=j["status"],
        summary=j["summary"], denial_reason=j["denial_reason"], documents=tuple(j["documents"] or ()),
        appeal_deadline=_day(j["appeal_deadline"]),
        expected_reimbursement_amount=_money(j["expected_reimbursement_amount"]),
        allowed_max_amount=_money(j["allowed_max_amount"]), net_pay=_money(j["net_pay"]), net_fee=_money(j["net_fee"]),
        updated_at=datetime.fromisoformat(j["updated_at"]))


async def count(conn: AsyncConnection, f: ClaimFilter, today: date) -> int:
    row = await (await conn.execute(_CHOSEN + "SELECT count(*) FROM chosen", _params(f, today))).fetchone()
    return row[0]


async def fetch(conn: AsyncConnection, f: ClaimFilter, today: date, limit: int) -> list[Claim]:
    """The chosen claims, newest first, at most `limit`."""
    cur = await conn.execute(_CHOSEN + f"SELECT {_CLAIM_JSON} FROM chosen {_NEWEST_FIRST} LIMIT %(limit)s",
                             _params(f, today, limit=limit))
    return [_claim(j) for (j,) in await cur.fetchall()]


async def fetch_ids(conn: AsyncConnection, case_ids: Sequence[str]) -> list[Claim]:
    """The named claims this conversation may see, in the order named; any other id is simply absent."""
    cur = await conn.execute(
        f"SELECT {_CLAIM_JSON} FROM (SELECT c.*, {_DOCUMENTS} AS documents FROM claims c "
        "WHERE c.case_id = ANY(%(ids)s::text[])) x ORDER BY array_position(%(ids)s::text[], case_id)",
        {"ids": list(case_ids)})
    return [_claim(j) for (j,) in await cur.fetchall()]


_KEY = "upper(regexp_replace(c.case_id, '[^A-Za-z0-9]', '', 'g'))"  # claim_number_key, on the stored side
_DIGITS = "regexp_replace(c.case_id, '[^0-9]', '', 'g')"


async def find_numbered(conn: AsyncConnection, stated: str, limit: int) -> list[Claim]:
    """Every claim of this conversation's customer (at most `limit`, newest first) that a claim number as typed could
    name: those whose `claim_number_key` equals it, or, when that names none, those whose digits equal the digits
    typed (so "2048" and "claim 2048" find CL-2048). Another customer's number finds nothing, exactly like a made-up
    one."""
    key = claim_number_key(stated)
    digits = "".join(ch for ch in key if ch.isdigit())
    if not key:
        return []
    for column, value in ((_KEY, key), (_DIGITS, digits)) if digits else ((_KEY, key),):
        cur = await conn.execute(
            f"SELECT {_CLAIM_JSON} FROM (SELECT c.*, {_DOCUMENTS} AS documents FROM claims c "
            f"WHERE {column} = %(key)s) x {_NEWEST_FIRST} LIMIT %(limit)s", {"key": value, "limit": limit})
        rows = await cur.fetchall()
        if rows:
            return [_claim(j) for (j,) in rows]
    return []


async def find_by_number(conn: AsyncConnection, stated: str) -> Claim | None:
    """The one claim a typed number names, or None (no claim, or two of the caller's claims share that number)."""
    found = await find_numbered(conn, stated, 2)
    return found[0] if len(found) == 1 else None


def as_listed(c: Claim) -> ListedClaim:
    """A shown claim as the form is told about it: the values the list showed, from the claim's own row."""
    return ListedClaim(c.case_id, c.case_type, c.status, c.created_at.isoformat() if c.created_at else None)


async def overview(conn: AsyncConnection, f: ClaimFilter, today: date) -> Overview:
    """A fixed-size overview in one statement: count, by type, by year, the most recent, the deadlines coming up."""
    cur = await conn.execute(_CHOSEN + f"""SELECT
      (SELECT count(*) FROM chosen),
      (SELECT coalesce(json_object_agg(case_type, n), '{{}}') FROM
         (SELECT case_type, count(*) AS n FROM chosen WHERE case_type IS NOT NULL GROUP BY case_type) t),
      (SELECT coalesce(json_object_agg(y, n), '{{}}') FROM
         (SELECT extract(year FROM created_at)::int AS y, count(*) AS n FROM chosen
          WHERE created_at IS NOT NULL GROUP BY 1) t),
      (SELECT coalesce(json_agg(j ORDER BY k), '[]') FROM
         (SELECT {_CLAIM_JSON} AS j, row_number() OVER ({_NEWEST_FIRST}) AS k FROM chosen
          {_NEWEST_FIRST} LIMIT %(size)s) r),
      (SELECT coalesce(json_agg(j ORDER BY k), '[]') FROM
         (SELECT {_CLAIM_JSON} AS j, row_number() OVER (ORDER BY appeal_deadline, case_id) AS k FROM chosen
          WHERE appeal_deadline >= %(today)s::date ORDER BY appeal_deadline, case_id LIMIT %(size)s) d)""",
                             _params(f, today, size=OVERVIEW_SIZE))
    n, by_type, by_year, recent, deadlines = await cur.fetchone()
    return Overview(n, dict(by_type), {int(y): c for y, c in by_year.items()},
                    tuple(_claim(j) for j in recent), tuple(_claim(j) for j in deadlines))


async def most_even_split(conn: AsyncConnection, f: ClaimFilter, today: date) -> str:
    """Which of case type, status, year and month splits the chosen claims most evenly: the one whose largest group
    is smallest, among those with two values or more that every chosen claim has (the caller is never asked which
    month of claims that have no date). Ties go in that order."""
    cur = await conn.execute(_CHOSEN + """SELECT attribute FROM (
        SELECT 'case_type' AS attribute, case_type::text AS v, count(*) AS n FROM chosen GROUP BY 2
        UNION ALL SELECT 'status', status, count(*) FROM chosen GROUP BY 2
        UNION ALL SELECT 'year', extract(year FROM created_at)::text, count(*) FROM chosen GROUP BY 2
        UNION ALL SELECT 'month', extract(month FROM created_at)::text, count(*) FROM chosen GROUP BY 2) g
      GROUP BY attribute HAVING count(*) > 1 AND count(*) = count(v)
      ORDER BY max(n), array_position(%(splits)s::text[], attribute) LIMIT 1""",
                             _params(f, today, splits=list(SPLITS)))
    row = await cur.fetchone()
    return SPLITS[0] if row is None else row[0]


async def customer_vocabulary(conn: AsyncConnection) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """The claim types, statuses and years of the claims this conversation may see, each sorted."""
    rows = await (await conn.execute(
        "SELECT DISTINCT case_type, status, extract(year FROM created_at)::int FROM claims")).fetchall()
    types = tuple(sorted({t for t, _, _ in rows if t}))
    statuses = tuple(sorted({s for _, s, _ in rows if s}))
    years = tuple(sorted({y for _, _, y in rows if y is not None}))
    return types, statuses, years


# ---- arithmetic: the database adds up, code only reads the result ----------------------------------------------------

TOP_SIZE = 5


@dataclass(frozen=True)
class Aggregate:
    total: Decimal                                   # the sum of the figures that are on file
    count: int                                       # the claims covered, with or without a figure
    missing: tuple[str, ...]                         # case ids of the claims with no figure on file
    top: tuple[tuple[str, Decimal], ...]             # at most five (case_id, value), best first
    not_paid_yet: tuple[str, ...] = ()               # case ids of in-progress claims whose figure is 0.00


# `scoped` is `chosen`, narrowed to the named claims when ids are given.
_SCOPE = """, scoped AS (SELECT * FROM chosen WHERE %(ids)s::text[] IS NULL OR case_id = ANY(%(ids)s::text[]))
"""


def _aggregate_sql(column: str, direction: str) -> str:
    """The one statement for one money column and one sort direction. Built only at import time, from the fixed
    names in MONEY_FIELDS and the two fixed directions below, never from a value."""
    return _CHOSEN + _SCOPE + f"""SELECT count(*), coalesce(sum({column}), 0)::text,
      coalesce(array_agg(case_id ORDER BY case_id) FILTER (WHERE {column} IS NULL), '{{}}'),
      coalesce(array_agg(case_id ORDER BY case_id)
               FILTER (WHERE {column} = 0 AND status = ANY(%(in_progress)s::text[])), '{{}}'),
      (SELECT coalesce(json_agg(json_build_array(case_id, v::text) ORDER BY k), '[]') FROM
         (SELECT case_id, {column} AS v, row_number() OVER (ORDER BY {column} {direction}, case_id) AS k
          FROM scoped WHERE {column} IS NOT NULL ORDER BY {column} {direction}, case_id LIMIT %(top)s) t)
    FROM scoped"""


_AGGREGATES = {(name, order): _aggregate_sql(name, direction)
               for name in MONEY_FIELDS for order, direction in (("highest", "DESC"), ("lowest", "ASC"))}


async def aggregate(conn: AsyncConnection, f: ClaimFilter, today: date, field: str, *,
                    order: Literal["highest", "lowest"] = "highest", in_progress: Sequence[str] = (),
                    case_ids: Sequence[str] | None = None) -> Aggregate:
    """The sum, count, missing figures and best five of one money column over the chosen claims (or, with `case_ids`,
    over those of them that are named). Runs as the server's account, so only the caller's own claims are counted.
    `field` must be a money field of the glossary; anything else raises before any SQL is sent."""
    sql = _AGGREGATES.get((field, order))
    if sql is None:
        raise ValueError(f"not a money field: {field!r}")
    cur = await conn.execute(sql, _params(f, today, top=TOP_SIZE, in_progress=list(in_progress),
                                          ids=None if case_ids is None else list(case_ids)))
    n, total, missing, unpaid, top = await cur.fetchone()
    return Aggregate(Decimal(total), n, tuple(missing), tuple((c, Decimal(v)) for c, v in top), tuple(unpaid))


async def chosen_among(conn: AsyncConnection, f: ClaimFilter, today: date, case_ids: Sequence[str]) -> set[str]:
    """Which of the named claims the filter chooses (a relative time is over every claim of the caller, as always)."""
    cur = await conn.execute(_CHOSEN + "SELECT case_id FROM chosen WHERE case_id = ANY(%(ids)s::text[])",
                             _params(f, today, ids=list(case_ids)))
    return {c for (c,) in await cur.fetchall()}

