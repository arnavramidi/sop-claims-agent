import psycopg
import psycopg.conninfo
import psycopg.sql
import pytest

from claims_agent.db.pool import AppDb

pytestmark = pytest.mark.db


async def visible_claims(app_db, conversation_id) -> list[str]:
    async with app_db.conversation(conversation_id) as c:
        return [r[0] for r in await (await c.execute("SELECT case_id FROM claims ORDER BY 1")).fetchall()]


async def visible_documents(app_db, conversation_id) -> list[str]:
    async with app_db.conversation(conversation_id) as c:
        return [r[0] for r in await (await c.execute("SELECT name FROM claim_documents ORDER BY 1")).fetchall()]


async def test_no_conversation_returns_zero_claims(app_db, seed):
    assert await visible_claims(app_db, None) == []


async def test_unverified_conversation_returns_zero_claims(app_db, seed):
    assert await visible_claims(app_db, seed.unverified) == []


async def test_verified_conversation_sees_exactly_its_customer(app_db, seed):
    assert await visible_claims(app_db, seed.verified_p12) == ["CL-3001"]


async def test_asking_for_another_customers_claim_returns_nothing(app_db, seed):
    async with app_db.conversation(seed.verified_p12) as c:
        assert await (await c.execute("SELECT 1 FROM claims WHERE party_id = 'P9'")).fetchall() == []


async def test_app_account_cannot_read_identity_values(app_db, seed):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(None) as c:
            await c.execute("SELECT dob, phone, email, id_last4 FROM customers")


async def test_app_account_cannot_mark_a_conversation_verified(app_db, seed):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(None) as c:
            await c.execute("UPDATE conversations SET verified_party_id = 'P9' WHERE id = %s", [seed.unverified])


async def test_table_owner_sees_zero_claims(owner_conn, seed):
    assert (await (await owner_conn.execute("SELECT count(*) FROM claims")).fetchone())[0] == 0


async def test_reused_connection_carries_nothing_over(test_db, seed):
    one = AppDb(test_db.app_url, max_size=1)
    assert await visible_claims(one, seed.verified_p12) == ["CL-3001"]
    assert await visible_claims(one, None) == []


async def test_outbox_recipient_is_not_readable_by_the_app(app_db, seed):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(None) as c:
            await c.execute("SELECT recipient FROM outbox")


# Extra checks beyond the core cases: the same lock on claim documents, and inserts that need no sequence rights.


async def test_claim_documents_follow_the_same_rule(app_db, seed, owner_conn):
    assert await visible_documents(app_db, None) == []
    assert await visible_documents(app_db, seed.unverified) == []
    assert await visible_documents(app_db, seed.verified_p12) == ["diagnosis report"]
    assert (await (await owner_conn.execute("SELECT count(*) FROM claim_documents")).fetchone())[0] == 0


async def test_app_account_cannot_change_claims(app_db, seed):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(seed.verified_p12) as c:
            await c.execute("UPDATE claims SET status = 'closed'")


async def test_app_account_can_start_a_conversation_and_record_a_turn(app_db, seed):
    new_id = "00000000-0000-4000-8000-0000000000c1"
    async with app_db.conversation(new_id) as c:  # a conversation is written from a transaction bound to it
        await c.execute(
            "INSERT INTO conversations (id, state, memory, consent_script, today) "
            "VALUES (%s, 'VERIFY_ID', '{}', 'default', '2026-03-05')", [new_id]
        )
    async with app_db.conversation(seed.unverified) as c:
        await c.execute("INSERT INTO turns (conversation_id, turn_no, message) VALUES (%s, 1, 'hi')", [seed.unverified])
        row = await (await c.execute("SELECT id FROM turns WHERE conversation_id = %s", [seed.unverified])).fetchone()
    assert row is not None


async def test_claims_updated_at_moves_on_every_change(loader_conn, seed):
    before = (await (await loader_conn.execute("SELECT updated_at FROM claims WHERE case_id = 'CL-3001'")).fetchone())[0]
    await loader_conn.execute("UPDATE claims SET status = 'closed' WHERE case_id = 'CL-3001'")
    await loader_conn.commit()
    after = (await (await loader_conn.execute("SELECT updated_at FROM claims WHERE case_id = 'CL-3001'")).fetchone())[0]
    assert before is not None and after > before


# The lock cannot be fooled by a temporary table, and a new function is closed.

SHADOW_ID = "00000000-0000-0000-0000-000000000001"


async def test_a_shadow_conversations_table_opens_nothing(test_db, app_db, seed):
    """Proves the function-level fix alone: temporary tables are re-allowed for this test only."""
    db_name = psycopg.conninfo.conninfo_to_dict(test_db.superuser_url)["dbname"]
    grant = psycopg.sql.SQL("GRANT TEMPORARY ON DATABASE {} TO claims_app").format(psycopg.sql.Identifier(db_name))
    revoke = psycopg.sql.SQL("REVOKE TEMPORARY ON DATABASE {} FROM claims_app").format(psycopg.sql.Identifier(db_name))
    async with await psycopg.AsyncConnection.connect(test_db.superuser_url, autocommit=True) as su:
        await su.execute(grant)
        try:
            async with app_db.conversation(SHADOW_ID) as c:
                await c.execute("CREATE TEMP TABLE conversations (id uuid, verified_party_id text) ON COMMIT DROP")
                await c.execute("INSERT INTO conversations VALUES (%s, 'P9')", [SHADOW_ID])
                claims = await (await c.execute("SELECT case_id FROM claims")).fetchall()
                documents = await (await c.execute("SELECT name FROM claim_documents")).fetchall()
        finally:
            await su.execute(revoke)
    assert claims == [] and documents == []


async def test_the_app_account_cannot_create_a_temporary_table(app_db, seed):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(None) as c:
            await c.execute("CREATE TEMP TABLE probe (a int)")


async def test_a_new_owner_function_is_closed_to_the_app_unless_granted(owner_conn, app_db, seed):
    await owner_conn.execute("CREATE FUNCTION public.probe_closed() RETURNS int LANGUAGE sql AS 'SELECT 1'")
    await owner_conn.commit()
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(None) as c:
            await c.execute("SELECT public.probe_closed()")


async def _add_event(app_db, conversation_id, body="{}"):
    async with app_db.conversation(conversation_id) as c:
        await c.execute("INSERT INTO events (conversation_id, turn_no, kind, body) VALUES (%s, 1, 'extraction', %s)",
                        [conversation_id, body])


async def test_a_conversation_reads_only_its_own_events(app_db, seed):
    await _add_event(app_db, seed.unverified)
    await _add_event(app_db, seed.verified_p12)
    async with app_db.conversation(seed.unverified) as c:
        rows = await (await c.execute("SELECT conversation_id FROM events")).fetchall()
    assert [r[0] for r in rows] == [seed.unverified]
    async with app_db.conversation(None) as c:
        assert await (await c.execute("SELECT count(*) FROM events")).fetchone() == (0,)


async def test_a_conversation_cannot_write_an_event_for_another_conversation(app_db, seed):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        await _add_event_for(app_db, seed.unverified, seed.verified_p12)


async def _add_event_for(app_db, as_conversation, for_conversation):
    async with app_db.conversation(as_conversation) as c:
        await c.execute("INSERT INTO events (conversation_id, turn_no, kind, body) VALUES (%s, 1, 'extraction', '{}')",
                        [for_conversation])


async def test_the_app_account_can_change_only_an_events_body_and_never_delete_one(app_db, seed):
    await _add_event(app_db, seed.unverified)
    async with app_db.conversation(seed.unverified) as c:
        await c.execute("UPDATE events SET body = '{\"a\": 1}'")
    for statement in ("UPDATE events SET kind = 'other'", "UPDATE events SET conversation_id = gen_random_uuid()",
                      "UPDATE events SET turn_no = 9", "DELETE FROM events"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with app_db.conversation(seed.unverified) as c:
                await c.execute(statement)
