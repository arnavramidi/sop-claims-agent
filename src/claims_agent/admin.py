"""The demo control that changes a claim: a status, an appeal deadline or the documents still needed.

It exists only when ADMIN_DATABASE_URL is set: the address of the demo account (claims_demo), which may change only
a claim's status and appeal deadline and the claim's documents, and only for the customer the conversation it is used
from is verified as (its row rules read that conversation, as the server's do). The server's own account cannot write
claims and is not given that power: a change is written over a separate connection opened for that one change, bound
to the conversation. Values are checked by the same rules the data loader applies.
"""

from datetime import date, datetime
from typing import Any
from uuid import UUID

import psycopg

from claims_agent.loader.read import _date, _text


class InvalidChange(ValueError):
    """A value the loader would not accept (the message is fixed wording, never the value)."""


def checked_status(value: Any, known: set[str]) -> str:
    status = _text(value)
    if status is None or status not in known:
        raise InvalidChange("status is not one of the known statuses")
    return status


def checked_deadline(value: Any) -> date:
    try:
        return date.fromisoformat(str(_date(value)))
    except ValueError:
        raise InvalidChange("appeal_deadline is not a date") from None


def checked_documents(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise InvalidChange("documents_needed is not a list")
    names = [_text(name) for name in value]
    if any(name is None for name in names):
        raise InvalidChange("a document name is not usable text")
    return names


async def change_claim(url: str, conversation_id: UUID, case_id: str, party_id: str, *, status: str | None = None,
                       appeal_deadline: date | None = None, documents_needed: list[str] | None = None) -> datetime:
    """Write the given columns of one claim as the demo account, in one transaction bound to the conversation; return
    its new updated_at. Documents are replaced as a whole and numbered 1, 2, ... like the loader numbers them."""
    async with await psycopg.AsyncConnection.connect(url) as conn:
        await conn.execute("SELECT set_config('app.conversation_id', %s, true)", [str(conversation_id)])
        sets, args = ["updated_at = now()"], []
        if status is not None:
            sets.append("status = %s")
            args.append(status)
        if appeal_deadline is not None:
            sets.append("appeal_deadline = %s")
            args.append(appeal_deadline)
        row = await (await conn.execute(
            f"UPDATE claims SET {', '.join(sets)} WHERE case_id = %s AND party_id = %s RETURNING updated_at",
            [*args, case_id, party_id])).fetchone()
        if row is None:
            raise LookupError("no such claim")
        if documents_needed is not None:
            await conn.execute("DELETE FROM claim_documents WHERE case_id = %s AND party_id = %s", [case_id, party_id])
            for position, name in enumerate(documents_needed, start=1):
                await conn.execute("INSERT INTO claim_documents (case_id, party_id, position, name) VALUES (%s, %s, %s, %s)",
                                   [case_id, party_id, position, name])
        return row[0]
