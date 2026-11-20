"""Thin callers of the identity functions in db/004_functions.sql.

The comparisons, the three-kind rule and every write to who a conversation is verified as happen inside
the database. What comes back is only party ids, matched kinds, yes/no answers and consent statuses.
"""

from typing import Literal, cast
from uuid import UUID

from psycopg import AsyncConnection

from claims_agent.identity.types import CustomerMatch, StatedIdentity

_STATED = "%s::text[], %s::date[], %s::text, %s::text, %s::text, %s::text"


def _stated(s: StatedIdentity) -> list:
    return [list(s.names), list(s.dobs), s.phone, s.email, s.id_last4, s.policy]


async def _one(conn: AsyncConnection, query: str, params: list):
    return await (await conn.execute(query, params)).fetchone()


async def match_identity(conn: AsyncConnection, s: StatedIdentity) -> list[CustomerMatch]:
    rows = await (await conn.execute(
        f"SELECT party_id, kinds, policy_ok FROM match_identity({_STATED})", _stated(s))).fetchall()
    return [CustomerMatch(party_id, frozenset(kinds), policy_ok is True) for party_id, kinds, policy_ok in rows]


async def bind_policyholder(conn: AsyncConnection, conversation_id: UUID, party_id: str, s: StatedIdentity) -> bool:
    row = await _one(conn, f"SELECT bind_policyholder(%s::uuid, %s::text, {_STATED})",
                     [conversation_id, party_id, *_stated(s)])
    return row[0] is True


async def request_consent(conn: AsyncConnection, conversation_id: UUID, party_id: str, s: StatedIdentity,
                          rep_names: tuple[str, ...], relationship: str | None,
                          ) -> Literal["requested", "not_listed", "refused"]:
    row = await _one(conn, f"SELECT request_consent(%s::uuid, %s::text, {_STATED}, %s::text[], %s::text)",
                     [conversation_id, party_id, *_stated(s), list(rep_names), relationship])
    return cast(Literal["requested", "not_listed", "refused"], row[0])


async def check_consent(conn: AsyncConnection, conversation_id: UUID) -> str:
    return (await _one(conn, "SELECT check_consent(%s::uuid)", [conversation_id]))[0]


async def clear_verified(conn: AsyncConnection, conversation_id: UUID) -> None:
    await conn.execute("SELECT clear_verified(%s::uuid)", [conversation_id])


async def account_info(conn: AsyncConnection, conversation_id: UUID) -> tuple[str, str] | None:
    row = await _one(conn, "SELECT name, policy_number FROM account_info(%s::uuid)", [conversation_id])
    return None if row is None else (row[0], row[1])


async def send_summary(conn: AsyncConnection, conversation_id: UUID, subject: str, body: str) -> int:
    return (await _one(conn, "SELECT send_summary(%s::uuid, %s::text, %s::text)",
                       [conversation_id, subject, body]))[0]
