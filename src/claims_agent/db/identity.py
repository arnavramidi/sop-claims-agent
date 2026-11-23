"""Thin callers of the identity functions in db/004_functions.sql.

The comparisons, the three-kind rule and every write to who a conversation is verified as happen inside
the database. What comes back is only party ids, matched kinds, yes/no answers and consent statuses.
The database also judges each compared value and counts the wrong ones per conversation (db/012_identity_lockout.sql);
a conversation at its limit is refused, which reaches the caller here as IdentityLocked, never as "no match".
"""

from dataclasses import dataclass
from typing import Literal, cast
from uuid import UUID

import psycopg
from psycopg import AsyncConnection

from claims_agent.db.actions import action, answer, found
from claims_agent.identity.types import CustomerMatch, StatedIdentity

_STATED = "%s::text, %s::date[], %s::text, %s::text, %s::text, %s::text"


def _stated(s: StatedIdentity) -> list:
    return [s.name, list(s.dobs), s.phone, s.email, s.id_last4, s.policy]


async def _one(conn: AsyncConnection, query: str, params: list):
    return await (await conn.execute(query, params)).fetchone()


LOCKED = "IL001"  # what the database raises for a conversation that has reached its limit of wrong values


class IdentityLocked(Exception):
    """The conversation has reached its limit of wrong identity values: the database compares nothing more for it."""


async def _compare(conn: AsyncConnection, query: str, params: list) -> list:
    """Run a call that compares identity values, in a savepoint, so the database refusing it leaves the turn's own
    transaction usable. That refusal is raised as IdentityLocked."""
    try:
        async with conn.transaction():
            return await (await conn.execute(query, params)).fetchall()
    except psycopg.DatabaseError as e:
        if e.sqlstate == LOCKED:
            raise IdentityLocked() from None
        raise


@action("match identity")  # no outcome: how many customers matched is not shown before verification
async def match_identity(conn: AsyncConnection, conversation_id: UUID, s: StatedIdentity) -> list[CustomerMatch]:
    """What matches, for the bound conversation; the database counts each wrong value as it compares."""
    rows = await _compare(conn, f"SELECT party_id, kinds, policy_ok FROM check_identity(%s::uuid, {_STATED})",
                          [conversation_id, *_stated(s)])
    return [CustomerMatch(party_id, frozenset(kinds), policy_ok is True) for party_id, kinds, policy_ok in rows]


@action("bind policyholder", lambda verified: {"verified": verified})
async def bind_policyholder(conn: AsyncConnection, conversation_id: UUID, party_id: str, s: StatedIdentity) -> bool:
    rows = await _compare(conn, f"SELECT bind_policyholder(%s::uuid, %s::text, {_STATED})",
                          [conversation_id, party_id, *_stated(s)])
    return rows[0][0] is True


@action("request consent", answer)
async def request_consent(conn: AsyncConnection, conversation_id: UUID, party_id: str, s: StatedIdentity,
                          rep_name: str | None) -> Literal["requested", "not_listed", "refused"]:
    """`rep_name` is the caller's own name as they said it; the list is matched on it alone."""
    rows = await _compare(conn, f"SELECT request_consent(%s::uuid, %s::text, {_STATED}, %s::text)",
                          [conversation_id, party_id, *_stated(s), rep_name])
    return cast(Literal["requested", "not_listed", "refused"], rows[0][0])


@action("check consent", answer)
async def check_consent(conn: AsyncConnection, conversation_id: UUID) -> str:
    return (await _one(conn, "SELECT check_consent(%s::uuid)", [conversation_id]))[0]


@action("reset verification")
async def clear_verified(conn: AsyncConnection, conversation_id: UUID) -> None:
    await conn.execute("SELECT clear_verified(%s::uuid)", [conversation_id])


@action("read account details", found)
async def account_info(conn: AsyncConnection, conversation_id: UUID) -> tuple[str, str] | None:
    row = await _one(conn, "SELECT name, policy_number FROM account_info(%s::uuid)", [conversation_id])
    return None if row is None else (row[0], row[1])


@action("store summary email")
async def send_summary(conn: AsyncConnection, conversation_id: UUID, subject: str, body: str) -> int:
    return (await _one(conn, "SELECT send_summary(%s::uuid, %s::text, %s::text)",
                       [conversation_id, subject, body]))[0]


@dataclass(frozen=True)
class NameShape:
    shape: Literal["whole", "single", "partial"]
    key: str  # the name's words in a fixed order: the same name said again in another order has the same key


@action("read name shape", lambda n: {"shape": n.shape})
async def name_shape(conn: AsyncConnection, name: str) -> NameShape:
    """What a stated name is, by the one name rule (db/004_functions.sql, name_words): only the database reads a name."""
    shape, key = await _one(conn, "SELECT shape, key FROM name_shape(%s::text)", [name])
    return NameShape(cast(Literal["whole", "single", "partial"], shape), key)


@action("compare names", answer)
async def name_is_part_of(conn: AsyncConnection, stated: str, held: str) -> bool:
    """Whether every word of the stated name is a word of the held name (or a typo of one), by the database's own
    name comparison: naming a policyholder by part of the name is naming her."""
    return (await _one(conn, "SELECT name_is_part_of(%s::text, %s::text)", [stated, held]))[0] is True


@action("read wrong values")  # no outcome: the record never says what matched before verification
async def wrong_values(conn: AsyncConnection, conversation_id: UUID) -> list[str]:
    """The wrong values the database has counted for the conversation, oldest first: each a fingerprint (the kind and a
    digest of the value, never the value). Only the database writes them; the server's account may only read its own
    conversation's."""
    rows = await (await conn.execute(
        "SELECT fingerprint FROM identity_wrong_values WHERE conversation_id = %s ORDER BY counted_at, fingerprint",
        [conversation_id])).fetchall()
    return [fingerprint for (fingerprint,) in rows]
