"""Write a LoadResult to Postgres as the loader account: empty every table, then insert, in one transaction.

Every row goes in inside its own savepoint, so a row the database refuses for its data is set aside and
reported while the rest of the load goes on. Rows that depend on a refused row are set aside too. Any
other failure (a lost connection, a timeout) aborts the load: it is one transaction, so the data loaded
before stays in place.

Everything derived is derived from the rows that were actually stored, never from the reader's copy:
which settings keys are usable blanks (and so which follow-ups are enabled) and the vocabulary. Report
entries are written last, so the report covers every row; an entry the database refuses leaves a fixed
entry in its place.
"""

from dataclasses import astuple, replace

import psycopg
from psycopg import errors, sql
from psycopg.types.json import Jsonb

from claims_agent.loader import read
from claims_agent.loader.read import LoadResult, Origin, ReportEntry

REFUSED = "the database could not store this row"
REPORT_REFUSED = "a load-report entry could not be stored"

# The errors that mean the database refused THIS row's data, and nothing else:
#  - DataError (SQLSTATE class 22): a value the column cannot take (a NUL in text, an untranslatable
#    character, a number out of range, jsonb it cannot parse); psycopg raises it before sending for a NUL.
#  - IntegrityError (class 23): a broken constraint (a duplicate key, a missing required value, a check).
#  - ProgramLimitExceeded (exactly 54000): a value over a size limit, such as an index row too large. Its
#    siblings in class 54 (statement too complex, too many columns or arguments) describe the query, not
#    the row, and are not subclasses of it.
#  - UnicodeEncodeError: text that cannot be sent at all (a lone surrogate); it fails in Python.
# Everything else says nothing about the row (connection lost, admin shutdown, timeout or cancel, deadlock,
# out of memory, a permission, a programming error): it propagates, and the whole load rolls back.
ROW_REFUSED = (psycopg.DataError, psycopg.IntegrityError, errors.ProgramLimitExceeded, UnicodeEncodeError)

_UNKNOWN = Origin(None, None, None)   # a row built by hand, not by the reader

CUSTOMER_COLUMNS = ("party_id", "name", "policy_number", "dob", "id_type", "id_last4", "phone", "email")
ALIAS_COLUMNS = ("party_id", "kind", "value")
REPRESENTATIVE_COLUMNS = ("party_id", "rep_name", "relationship")
CLAIM_COLUMNS = ("case_id", "party_id", "case_type", "created_at", "status", "summary", "denial_reason",
                 "appeal_deadline", "expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee")
DOCUMENT_COLUMNS = ("case_id", "party_id", "position", "name")
GUIDANCE_COLUMNS = ("kind", "key", "body")
CONSENT_COLUMNS = ("name", "status_sequence")
VOCABULARY_COLUMNS = ("kind", "value")
REPORT_COLUMNS = ("level", "file", "ref", "field", "reason")


def _followup_body(f) -> dict:
    return {"needs": list(f.needs), "requires_documents": f.requires_documents, "examples": list(f.examples),
            "template": f.template, "position": f.position, "enabled": f.enabled}


def _columns(columns):
    return lambda row: tuple(getattr(row, c) for c in columns)


class _Writer:
    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self.conn = conn
        self.report: list[ReportEntry] = []   # the writer's own entries, written after every data row

    async def _insert(self, table: str, columns: tuple[str, ...], values: tuple) -> bool:
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() * len(columns)))
        try:
            async with self.conn.transaction():  # a savepoint: rolled back alone if the insert fails
                await self.conn.execute(query, values)
            return True
        except ROW_REFUSED:  # the database's own message is never kept: it can quote the row
            return False

    def set_aside(self, origin: Origin | None) -> None:
        """Report a row that was not stored by where it came from only, never by a value from it."""
        o = origin or _UNKNOWN
        self.report.append(ReportEntry("row", o.file, o.ref, o.field, REFUSED))

    async def store(self, table: str, columns: tuple[str, ...], rows, values=None, origin=lambda row: row.origin) -> list:
        """Insert each row in its own savepoint; return the rows the database took."""
        values = values or _columns(columns)
        stored = []
        for row in rows:
            if await self._insert(table, columns, values(row)):
                stored.append(row)
            else:
                self.set_aside(origin(row))
        return stored

    def only(self, rows, keep) -> list:
        """The rows whose parent was stored; the others are set aside."""
        kept = []
        for row in rows:
            if keep(row):
                kept.append(row)
            else:
                self.set_aside(row.origin)
        return kept

    async def write_report(self, entries: list[ReportEntry]) -> None:
        """Last of all. An entry the database refuses is replaced by a fixed one that keeps only its level
        and file, and only when they are the reader's own constants, so the replacement cannot be refused
        for its data and no entry is lost without trace."""
        for e in entries:
            if not await self._insert("load_report", REPORT_COLUMNS, astuple(e)):
                level = e.level if e.level in read.LEVELS else "row"
                file = e.file if e.file in read.FILES else None
                await self.conn.execute(
                    "INSERT INTO load_report (level, file, ref, field, reason) VALUES (%s, %s, NULL, NULL, %s)",
                    (level, file, REPORT_REFUSED))


async def write(result: LoadResult, loader_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(loader_url) as conn:
        async with conn.transaction():
            tables = await (await conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
            await conn.execute(sql.SQL("TRUNCATE {}").format(
                sql.SQL(", ").join(sql.Identifier(t) for (t,) in tables)))
            w = _Writer(conn)

            # The base rows, each kept only if what it depends on was stored.
            customers = await w.store("customers", CUSTOMER_COLUMNS, result.customers)
            parties = {c.party_id for c in customers}
            await w.store("customer_aliases", ALIAS_COLUMNS, w.only(result.aliases, lambda a: a.party_id in parties))
            representatives = await w.store("representatives", REPRESENTATIVE_COLUMNS,
                                            w.only(result.representatives, lambda p: p.party_id in parties))
            claims = await w.store("claims", CLAIM_COLUMNS, w.only(result.claims, lambda c: c.party_id in parties))
            case_ids = {c.case_id for c in claims}
            await w.store("claim_documents", DOCUMENT_COLUMNS, w.only(result.documents, lambda d: d.case_id in case_ids))
            guidance = await w.store("guidance", GUIDANCE_COLUMNS, result.guidance,
                                     lambda g: (g.kind, g.key, Jsonb(g.body)))
            await w.store("consent_scripts", CONSENT_COLUMNS, result.consent_scripts,
                          lambda s: (s.name, list(s.status_sequence)))

            # Derived from what was stored: a follow-up is enabled only if every blank it uses has a stored value.
            allowed = read.allowed_blanks(g.key for g in guidance if g.kind == read.SETTINGS_KIND)
            followups = []
            for f in result.followups:
                problem = read.template_problem(f.template, allowed)
                derived = replace(f, enabled=problem is None)
                if await w.store("guidance", GUIDANCE_COLUMNS, [derived],
                                 lambda x: ("followup", x.topic, Jsonb(_followup_body(x)))):
                    followups.append(derived)
                    if problem and f.enabled:   # the reader enabled it, but a blank's settings row was not stored
                        w.report.append(read.disabled_followup(derived, problem))

            words = read.vocabulary(claims, representatives, followups)
            await w.store("vocabulary", VOCABULARY_COLUMNS, [(k, v) for k, vs in words.items() for v in sorted(vs)],
                          lambda kv: kv, lambda kv: Origin(None, "vocabulary", kv[0]))

            await w.write_report([*result.report, *w.report])
