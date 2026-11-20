"""Loading and saving a conversation, one turn at a time.

`load_for_turn` takes a row lock that lasts until the surrounding transaction ends (AppDb.conversation),
so a second message for the same conversation waits and then sees the first one's saved memory.
The server's account cannot write the two party columns, so `save` never tries.
"""

from dataclasses import dataclass
from datetime import date
from uuid import UUID, uuid4

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from claims_agent.memory import Memory
from claims_agent.states import State


@dataclass
class Conversation:
    id: UUID
    state: State
    verified_party_id: str | None
    pending_party_id: str | None
    consent_script: str
    today: date | None


async def _require_bound(conn: AsyncConnection, conversation_id: UUID) -> None:
    """The transaction must be bound (AppDb.conversation(id)) to exactly this conversation: the binding alone
    decides whose claims are open, so the store never touches another conversation's rows."""
    row = await (await conn.execute("SELECT current_setting('app.conversation_id', true)")).fetchone()
    if row[0] != str(conversation_id):
        raise LookupError("the transaction is not bound to this conversation")


async def create(conn: AsyncConnection, *, consent_script: str = "default", today: date | None = None) -> UUID:
    cid = uuid4()
    await conn.execute(
        "INSERT INTO conversations (id, state, memory, consent_script, today) VALUES (%s, %s, %s, %s, %s)",
        [cid, State.VERIFY_ID.value, Jsonb(Memory.fresh().model_dump(mode="json")), consent_script, today],
    )
    return cid


async def load_for_turn(conn: AsyncConnection, conversation_id: UUID) -> tuple[Conversation, Memory]:
    await _require_bound(conn, conversation_id)
    row = await (await conn.execute(
        "SELECT state, verified_party_id, pending_party_id, consent_script, today, memory "
        "FROM conversations WHERE id = %s FOR UPDATE", [conversation_id])).fetchone()
    if row is None:
        raise LookupError("no such conversation")
    state, verified, pending, script, today, memory = row
    conv = Conversation(conversation_id, State(state), verified, pending, script, today)
    return conv, Memory.model_validate(memory)


async def save(conn: AsyncConnection, conversation: Conversation, memory: Memory) -> None:
    await _require_bound(conn, conversation.id)
    cur = await conn.execute(
        "UPDATE conversations SET state = %s, memory = %s, updated_at = now() WHERE id = %s",
        [conversation.state.value, Jsonb(memory.model_dump(mode="json")), conversation.id],
    )
    if cur.rowcount != 1:
        raise LookupError("no such conversation")


async def append_turn(conn: AsyncConnection, conversation_id: UUID, turn_no: int, message: str, reply: str,
                      state_before: str, state_after: str, record: dict, braintrust_span_id: str | None = None) -> int:
    await _require_bound(conn, conversation_id)
    row = await (await conn.execute(
        "INSERT INTO turns (conversation_id, turn_no, message, reply, state_before, state_after, record, "
        "braintrust_span_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        [conversation_id, turn_no, message, reply, state_before, state_after, Jsonb(record),
         braintrust_span_id])).fetchone()
    return row[0]


async def transcript(conn: AsyncConnection, conversation_id: UUID) -> list[dict]:
    await _require_bound(conn, conversation_id)
    cur = await conn.execute(
        "SELECT turn_no, message, reply, state_after, record, id FROM turns WHERE conversation_id = %s "
        "ORDER BY turn_no, id", [conversation_id])
    return [{"turn_no": n, "message": m, "reply": r, "state_after": s, "record": rec, "id": i}
            for n, m, r, s, rec, i in await cur.fetchall()]


async def add_escalation(conn: AsyncConnection, conversation_id: UUID, turn_no: int, kind: str, reason: str,
                         note: dict) -> None:
    await _require_bound(conn, conversation_id)
    await conn.execute(
        "INSERT INTO escalations (conversation_id, turn_no, kind, reason, note) VALUES (%s, %s, %s, %s, %s)",
        [conversation_id, turn_no, kind, reason, Jsonb(note)])


async def set_controls(conn: AsyncConnection, conversation_id: UUID, *, consent_script: str | None = None,
                       today: date | None = None) -> None:
    await _require_bound(conn, conversation_id)
    await conn.execute(
        "UPDATE conversations SET consent_script = COALESCE(%s, consent_script), today = COALESCE(%s, today), "
        "updated_at = now() WHERE id = %s", [consent_script, today, conversation_id])


async def set_rating(conn: AsyncConnection, turn_id: int, rating: int, comment: str | None) -> tuple[bool, str | None]:
    """Store a turn's rating (a second one replaces the first). Returns (the turn exists, its Braintrust span id or
    None)."""
    row = await (await conn.execute("SELECT braintrust_span_id FROM turns WHERE id = %s", [turn_id])).fetchone()
    if row is None:
        return False, None
    await conn.execute(
        "INSERT INTO feedback (turn_id, rating, comment) VALUES (%s, %s, %s) "
        "ON CONFLICT (turn_id) DO UPDATE SET rating = EXCLUDED.rating, comment = EXCLUDED.comment",
        [turn_id, str(rating), comment])
    return True, row[0]
