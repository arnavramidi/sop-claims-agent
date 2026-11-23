"""Every per-conversation table shows the server's account only the rows of the conversation its transaction is bound
to: a transaction bound to A sees none of B's rows and cannot write one, and an unbound transaction sees none at all."""

import uuid

import psycopg
import pytest

from claims_agent.db import conversations as store

pytestmark = pytest.mark.db

# Each table, the column that names its conversation, and one row for a conversation (%s is its id).
TABLES = {
    "conversations": ("id", "INSERT INTO conversations (id, state) VALUES (%s, 'VERIFY_ID')"),
    "turns": ("conversation_id", "INSERT INTO turns (conversation_id, turn_no, message, reply) VALUES (%s, 1, 'm', 'r')"),
    "escalations": ("conversation_id", "INSERT INTO escalations (conversation_id, turn_no, kind) VALUES (%s, 1, 'k')"),
    "outbox": ("conversation_id", "INSERT INTO outbox (conversation_id, party_id, subject, body) VALUES (%s, 'P9', 's', 'b')"),
    "consent_requests": ("conversation_id",
                         "INSERT INTO consent_requests (conversation_id, party_id, script, status) "
                         "VALUES (%s, 'P9', 'default', 'requested')"),
    "feedback": ("conversation_id", "INSERT INTO feedback (conversation_id, turn_id, rating) VALUES (%s, %s, '1')"),
}
# What the server's account may write in each table, aimed at another conversation's row (every one must be refused
# or change nothing).
INSERTS_FOR_B = {
    "conversations": "INSERT INTO conversations (id, state, memory, consent_script) VALUES (%(b)s, 'VERIFY_ID', '{}', 'default')",
    "turns": "INSERT INTO turns (conversation_id, turn_no, message) VALUES (%(b)s, 9, 'x')",
    "escalations": "INSERT INTO escalations (conversation_id, turn_no, kind) VALUES (%(b)s, 9, 'x')",
    "feedback": "INSERT INTO feedback (conversation_id, turn_id, rating) VALUES (%(b)s, 999, '-1')",
}
UPDATES_OF_B = {
    "conversations": "UPDATE conversations SET state = 'CLOSED', memory = '{}' WHERE id = %(b)s",
    "feedback": "UPDATE feedback SET rating = '-1', comment = 'x' WHERE conversation_id = %(b)s",
}
NOT_WRITABLE = {   # the server's account was never given these writes at all
    "turns": "UPDATE turns SET reply = 'x' WHERE conversation_id = %(b)s",
    "escalations": "UPDATE escalations SET reason = 'x' WHERE conversation_id = %(b)s",
    "outbox": "INSERT INTO outbox (conversation_id, party_id) VALUES (%(b)s, 'P9')",
    "consent_requests": "UPDATE consent_requests SET status = 'approved' WHERE conversation_id = %(b)s",
}


@pytest.fixture
async def two(loader_conn):
    """Two conversations, A and B, each with one row in every per-conversation table (written as the loader)."""
    a, b = uuid.uuid4(), uuid.uuid4()
    for cid in (a, b):
        for table, (_, insert) in TABLES.items():
            if table == "feedback":
                turn = await (await loader_conn.execute(
                    "SELECT id FROM turns WHERE conversation_id = %s", [cid])).fetchone()
                await loader_conn.execute(insert, [cid, turn[0]])
            else:
                await loader_conn.execute(insert, [cid])
    await loader_conn.commit()
    return a, b


async def _count(app_db, bound, table, cid) -> int:
    column = TABLES[table][0]
    async with app_db.conversation(bound) as c:
        return (await (await c.execute(
            f"SELECT count(*) FROM {table} WHERE {column} = %s", [cid])).fetchone())[0]


async def _total(app_db, bound, table) -> int:
    async with app_db.conversation(bound) as c:
        return (await (await c.execute(f"SELECT count(*) FROM {table}")).fetchone())[0]


@pytest.mark.parametrize("table", list(TABLES))
async def test_a_bound_transaction_sees_only_its_own_conversations_rows(app_db, two, table):
    a, b = two
    assert await _count(app_db, a, table, a) == 1
    assert await _count(app_db, a, table, b) == 0
    assert await _total(app_db, a, table) == 1


@pytest.mark.parametrize("table", list(TABLES))
async def test_an_unbound_transaction_sees_no_conversation_rows(app_db, two, table):
    assert await _total(app_db, None, table) == 0


@pytest.mark.parametrize("table", list(INSERTS_FOR_B))
async def test_an_insert_for_another_conversation_is_refused(app_db, two, table, loader_conn):
    a, b = two
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(a) as c:
            await c.execute(INSERTS_FOR_B[table], {"b": uuid.uuid4() if table == "conversations" else b})


@pytest.mark.parametrize("table", list(INSERTS_FOR_B))
async def test_an_unbound_insert_is_refused(app_db, two, table):
    _, b = two
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(None) as c:
            await c.execute(INSERTS_FOR_B[table], {"b": uuid.uuid4() if table == "conversations" else b})


@pytest.mark.parametrize("table", list(UPDATES_OF_B))
async def test_an_update_of_another_conversations_rows_changes_nothing(app_db, two, table, loader_conn):
    a, b = two
    async with app_db.conversation(a) as c:
        assert (await c.execute(UPDATES_OF_B[table], {"b": b})).rowcount == 0
    async with app_db.conversation(None) as c:
        assert (await c.execute(UPDATES_OF_B[table], {"b": b})).rowcount == 0
    row = await (await loader_conn.execute(
        "SELECT state FROM conversations WHERE id = %s" if table == "conversations"
        else "SELECT rating FROM feedback WHERE conversation_id = %s", [b])).fetchone()
    assert row[0] in ("VERIFY_ID", "1")


@pytest.mark.parametrize("table", list(NOT_WRITABLE))
async def test_tables_the_server_never_writes_stay_closed(app_db, two, table):
    a, b = two
    for bound in (a, b, None):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with app_db.conversation(bound) as c:
                await c.execute(NOT_WRITABLE[table], {"b": b})


async def test_creating_a_conversation_binds_the_transaction_to_it(app_db):
    async with app_db.conversation(None) as c:
        cid = await store.create(c)
        assert (await (await c.execute("SELECT count(*) FROM conversations WHERE id = %s", [cid])).fetchone())[0] == 1
    assert await _count(app_db, cid, "conversations", cid) == 1
    assert await _total(app_db, None, "conversations") == 0


async def test_a_transaction_bound_to_one_conversation_cannot_create_another(app_db, two):
    a, _ = two
    with pytest.raises(LookupError):
        async with app_db.conversation(a) as c:
            await store.create(c)


async def test_the_verified_customer_is_read_from_the_bound_row_only(app_db, seed):
    """The claims rule still works: it reads the bound conversation, which the conversations rule shows."""
    async with app_db.conversation(seed.verified_p12) as c:
        assert (await (await c.execute("SELECT current_verified_party()")).fetchone())[0] == "P12"
        assert (await (await c.execute("SELECT count(*) FROM claims")).fetchone())[0] == 1
    async with app_db.conversation(None) as c:
        assert (await (await c.execute("SELECT current_verified_party()")).fetchone())[0] is None
