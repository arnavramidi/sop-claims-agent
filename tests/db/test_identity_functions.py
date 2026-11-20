import uuid

import psycopg
import pytest

from claims_agent.db.identity import (
    account_info, bind_policyholder, check_consent, clear_verified, match_identity, request_consent, send_summary,
)
from claims_agent.identity.normalize import name_candidates, normalise
from claims_agent.identity.types import StatedIdentity
from helpers import stated, verified_party

pytestmark = pytest.mark.db

MARGARET = dict(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472")


# Fixtures and helpers the tests name. Every dataset fixture empties the test database and loads its own data set.


@pytest.fixture
async def db(load_dataset):
    return await load_dataset("starter")


@pytest.fixture
async def shared_name_db(load_dataset):
    return await load_dataset("shared_name")


@pytest.fixture
async def bad_data_db(load_dataset):
    return await load_dataset("bad_data")


@pytest.fixture
async def extra_consent_db(load_dataset):
    return await load_dataset("extra_consent")


async def new_conversation(db, script: str = "default") -> uuid.UUID:
    """A new unverified conversation, created through the server's own account."""
    conversation_id = uuid.uuid4()
    async with db.app.conversation(None) as c:
        await c.execute("INSERT INTO conversations (id, state, consent_script, today) "
                        "VALUES (%s, 'VERIFY_ID', %s, '2026-03-05')", [conversation_id, script])
    return conversation_id


@pytest.fixture
async def conv(db) -> uuid.UUID:
    return await new_conversation(db)


@pytest.fixture
async def conv_short(extra_consent_db) -> uuid.UUID:
    return await new_conversation(extra_consent_db, "short")


@pytest.fixture
async def loader_conn(db):
    """The loader account on the dataset the test loaded (overrides the conftest one, which is built on `test_db`)."""
    async with await psycopg.AsyncConnection.connect(db.loader_url) as conn:
        yield conn


async def kinds(db, **v) -> dict[str, frozenset[str]]:
    s, _ = normalise(stated(**v))
    async with db.app.conversation(None) as c:
        return {m.party_id: m.kinds for m in await match_identity(c, s)}


async def bind(db, conversation_id, party_id, **v) -> bool:
    s, _ = normalise(stated(**v))
    async with db.app.conversation(conversation_id) as c:
        return await bind_policyholder(c, conversation_id, party_id, s)


async def consent(db, conversation_id, party_id, *, rep, relationship, **v) -> str:
    s, _ = normalise(stated(**v))
    async with db.app.conversation(conversation_id) as c:
        return await request_consent(c, conversation_id, party_id, s, name_candidates(rep), relationship)


async def check(db, conversation_id) -> str:
    async with db.app.conversation(conversation_id) as c:
        return await check_consent(c, conversation_id)


async def account(db, conversation_id):
    async with db.app.conversation(conversation_id) as c:
        return await account_info(c, conversation_id)


async def request_rows(db, conversation_id) -> int:
    async with db.app.conversation(conversation_id) as c:
        row = await (await c.execute(
            "SELECT count(*) FROM consent_requests WHERE conversation_id = %s", [conversation_id])).fetchone()
    return row[0]


# The brief's tests.


async def test_v1_phone_one_digit_off_does_not_count_for_margaret(db):
    k = await kinds(db, policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], phone="650-521-2830")
    assert k["P9"] == {"name", "dob"} and k["P13"] == {"phone"}


async def test_v3_aliases_never_score_twice(db):
    k = await kinds(db, email="yawen.li@example.com", phone="+1 650 521 2830"); assert k["P13"] == {"email", "phone"}


async def test_v6_id_counts_whatever_it_is_called(db):
    assert (await kinds(db, id_digits="6688"))["P12"] == {"id"}


async def test_v7_either_date_reading_matches(db):
    assert "dob" in (await kinds(db, date_of_birth=["1989-03-12", "1989-12-03"]))["P13"]


async def test_v10_alias_spacing_order_and_one_letter(db):
    for name in ("Yaven Li", "Yawen Li", "Li Ya Wen", "Ya Wen Lo"):
        assert "name" in (await kinds(db, policyholder_name=name)).get("P13", set()), name
    assert "P13" not in await kinds(db, policyholder_name="Li")


async def test_v22_shared_name_counts_for_both(shared_name_db):
    k = await kinds(shared_name_db, policyholder_name="Margaret Chen"); assert k["P9"] == k["P21"] == {"name"}


async def test_d8_shared_phone_counts_for_both(shared_name_db):
    k = await kinds(shared_name_db, phone="6503882920"); assert k["P7"] == k["P22"] == {"phone"}


async def test_d7_malformed_stored_id_cannot_match(bad_data_db):
    assert "id" not in (await kinds(bad_data_db, policyholder_name="Bad Fields", id_digits="44A2")).get("B3", set())


async def test_bind_needs_three_kinds(db, conv):
    assert await bind(db, conv, "P9", policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"]) is False
    assert await verified_party(db, conv) is None
    assert await bind(db, conv, "P9", **MARGARET) is True and await verified_party(db, conv) == "P9"


async def test_bind_refuses_someone_elses_policy_number(db, conv):
    assert await bind(db, conv, "P9", **MARGARET, policy_number="POL-1044") is False


async def test_listed_representative_gets_a_request_not_claims(db, conv):
    assert await consent(db, conv, "P9", rep="David Chen", relationship="son", **MARGARET) == "requested"
    assert await verified_party(db, conv) is None


async def test_c9_contradicting_relationship_is_not_listed(db, conv):
    assert await consent(db, conv, "P9", rep="David Chen", relationship="other", **MARGARET) == "not_listed"


async def test_c6_unlisted_relative_gets_no_request(db, conv):
    assert await consent(db, conv, "P9", rep="Tom Chen", relationship=None, **MARGARET) == "not_listed"
    assert await request_rows(db, conv) == 0


async def test_c2_default_script_is_pending_then_approved(db, conv):
    await consent(db, conv, "P9", rep="David Chen", relationship="son", **MARGARET)
    assert [await check(db, conv), await check(db, conv)] == ["pending", "approved"] and await verified_party(db, conv) == "P9"


async def test_x12_approval_without_three_matched_details_unlocks_nothing(db, conv, loader_conn):
    await loader_conn.execute("INSERT INTO consent_requests (conversation_id, party_id, script, status) VALUES (%s, 'P9', 'default', 'approved')", [conv]); await loader_conn.commit()
    await check(db, conv); await check(db, conv); assert await verified_party(db, conv) is None


async def test_ruling_13_a_short_script_repeats_its_last_value(extra_consent_db, conv_short):
    await consent(extra_consent_db, conv_short, "P9", rep="David Chen", relationship="son", **MARGARET)
    assert [await check(extra_consent_db, conv_short) for _ in range(3)] == ["pending"] * 3


async def test_account_info_only_for_the_verified_customer(db, conv):
    assert await account(db, conv) is None; await bind(db, conv, "P9", **MARGARET)
    assert await account(db, conv) == ("Margaret Chen", "POL-9921")


# Extra checks beyond the brief: the functions treat every argument as hostile and are locked down.


async def test_the_database_itself_refuses_a_partial_phone(db):
    """The server could pass a short phone straight to the function; a suffix match on it would let it
    read a stored number digit by digit."""
    async with db.app.conversation(None) as c:
        assert await match_identity(c, StatedIdentity((), (), "2830", None, None, None)) == []
        assert await match_identity(c, StatedIdentity((), (), None, None, "44A2", None)) == []


async def test_the_database_refuses_a_list_of_guessed_birth_dates(db):
    """Two readings is the most one stated date can have; a long list would test many dates in one call."""
    from datetime import date, timedelta
    guesses = tuple(date(1985, 1, 1) + timedelta(days=i) for i in range(3))
    with pytest.raises(psycopg.errors.RaiseException) as e:
        async with db.app.conversation(None) as c:
            await match_identity(c, StatedIdentity((), guesses, None, None, None, None))
    assert "1985" not in str(e.value)


async def test_check_consent_with_no_request_says_so(db, conv):
    assert await check(db, conv) == "no_request"


async def test_clear_verified_takes_away_customer_request_and_pending(db, conv):
    await consent(db, conv, "P9", rep="David Chen", relationship="son", **MARGARET)
    await check(db, conv); await check(db, conv)
    assert await verified_party(db, conv) == "P9"
    async with db.app.conversation(conv) as c:
        await clear_verified(c, conv)
    assert await verified_party(db, conv) is None and await request_rows(db, conv) == 0
    assert await check(db, conv) == "no_request" and await verified_party(db, conv) is None


async def test_binding_drops_a_pending_request_for_someone_else(db, conv):
    """A later approval must not switch a verified conversation to the earlier pending customer."""
    await consent(db, conv, "P9", rep="David Chen", relationship="son", **MARGARET)
    assert await bind(db, conv, "P12", policyholder_name="Ma Tian", date_of_birth=["1964-09-10"], id_digits="6688")
    await check(db, conv); await check(db, conv)
    assert await verified_party(db, conv) == "P12"


async def test_send_summary_goes_to_the_stored_email_and_needs_a_verified_customer(db, conv, loader_conn):
    with pytest.raises(psycopg.errors.RaiseException) as e:
        async with db.app.conversation(conv) as c:
            await send_summary(c, conv, "Your claims", "body")
    assert "@" not in str(e.value)
    await bind(db, conv, "P9", **MARGARET)
    async with db.app.conversation(conv) as c:
        outbox_id = await send_summary(c, conv, "Your claims", "body")
    row = await (await loader_conn.execute(
        "SELECT conversation_id, party_id, recipient, subject FROM outbox WHERE id = %s", [outbox_id])).fetchone()
    assert row == (conv, "P9", "margaret@email.com", "Your claims")


async def test_every_function_is_locked_down(db):
    """Owner-created functions: closed to PUBLIC, pinned search path with the temporary schema last;
    the seven the server calls are security definer and granted to it; nothing else is granted to it."""
    async with await psycopg.AsyncConnection.connect(db.superuser_url) as su:
        rows = await (await su.execute(
            "SELECT p.proname, p.prosecdef, p.proconfig, "
            "       has_function_privilege('claims_app', p.oid, 'EXECUTE'), "
            "       coalesce(aclexplode_public.public_exec, false) "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "LEFT JOIN LATERAL (SELECT bool_or(a.grantee = 0 AND a.privilege_type = 'EXECUTE') AS public_exec "
            "                   FROM aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) a) aclexplode_public ON true "
            "WHERE n.nspname = 'public' AND p.proowner = 'claims_owner'::regrole")).fetchall()
    served = {"match_identity", "bind_policyholder", "request_consent", "check_consent", "clear_verified",
              "account_info", "send_summary"}
    by_name = {r[0]: r for r in rows}
    assert served <= set(by_name)
    for name, definer, config, app_may, public_may in rows:
        assert not public_may, name
        assert config and "search_path=pg_catalog, public, pg_temp" in config, name
        assert app_may == (name in served or name == "current_verified_party"), name
        if name in served:
            assert definer, name


# Fix round 1 (binding): a function that acts on a conversation acts only on the one the transaction is
# bound to, because that binding alone decides whose claims are visible.

_S = normalise(stated(**MARGARET))[0]
ACTS_ON_A_CONVERSATION = {
    "bind_policyholder": lambda c, cid: bind_policyholder(c, cid, "P9", _S),
    "request_consent": lambda c, cid: request_consent(c, cid, "P9", _S, name_candidates("David Chen"), "son"),
    "check_consent": lambda c, cid: check_consent(c, cid),
    "clear_verified": lambda c, cid: clear_verified(c, cid),
    "account_info": lambda c, cid: account_info(c, cid),
    "send_summary": lambda c, cid: send_summary(c, cid, "Your claims", "body"),
}


async def primed(db, loader_conn) -> uuid.UUID:
    """A conversation every one of the six functions would change: verified and pending as P9, with a
    consent request."""
    b = await new_conversation(db)
    await loader_conn.execute("UPDATE conversations SET verified_party_id = 'P9', pending_party_id = 'P9' WHERE id = %s", [b])
    await loader_conn.execute("INSERT INTO consent_requests (conversation_id, party_id, script, status) "
                              "VALUES (%s, 'P9', 'default', 'requested')", [b])
    await loader_conn.commit()
    return b


async def snapshot(loader_conn, conversation_id) -> tuple:
    conv_row = await (await loader_conn.execute(
        "SELECT verified_party_id, pending_party_id, updated_at FROM conversations WHERE id = %s", [conversation_id])).fetchone()
    request = await (await loader_conn.execute(
        "SELECT party_id, checks_used, status FROM consent_requests WHERE conversation_id = %s", [conversation_id])).fetchall()
    outbox = await (await loader_conn.execute(
        "SELECT count(*) FROM outbox WHERE conversation_id = %s", [conversation_id])).fetchone()
    await loader_conn.commit()
    return conv_row, request, outbox


@pytest.mark.parametrize("name", sorted(ACTS_ON_A_CONVERSATION))
async def test_a_function_refuses_a_conversation_other_than_the_bound_one(name, db, conv, loader_conn):
    b = await primed(db, loader_conn)
    before = await snapshot(loader_conn, b)
    with pytest.raises(psycopg.errors.InsufficientPrivilege) as e:
        async with db.app.conversation(conv) as c:
            await ACTS_ON_A_CONVERSATION[name](c, b)
    assert "bound" in str(e.value) and str(b) not in str(e.value)
    assert await snapshot(loader_conn, b) == before


@pytest.mark.parametrize("name", sorted(ACTS_ON_A_CONVERSATION))
async def test_a_function_refuses_when_no_conversation_is_bound(name, db, loader_conn):
    b = await primed(db, loader_conn)
    before = await snapshot(loader_conn, b)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with db.app.conversation(None) as c:
            await ACTS_ON_A_CONVERSATION[name](c, b)
    assert await snapshot(loader_conn, b) == before


@pytest.mark.parametrize("name", sorted(ACTS_ON_A_CONVERSATION))
async def test_a_function_works_on_the_bound_conversation(name, db, loader_conn):
    b = await primed(db, loader_conn)
    async with db.app.conversation(b) as c:
        await ACTS_ON_A_CONVERSATION[name](c, b)


async def test_a_binding_left_in_the_session_by_an_earlier_transaction_does_not_count(db, conv, loader_conn):
    """The binding is per transaction: a later unbound transaction on the same connection is unbound."""
    from claims_agent.db.pool import AppDb
    one = AppDb(db.app_url, max_size=1)
    try:
        b = await primed(db, loader_conn)
        async with one.conversation(b) as c:
            await account_info(c, b)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with one.conversation(None) as c:
                await account_info(c, b)
    finally:
        await one.close()


async def test_every_function_the_server_may_call_with_a_conversation_id_checks_the_binding(db):
    """Catches a function added later that takes a conversation id and forgets the shared check."""
    async with await psycopg.AsyncConnection.connect(db.superuser_url) as su:
        rows = await (await su.execute(
            "SELECT p.proname, p.prosrc FROM pg_proc p "
            "WHERE p.pronamespace = 'public'::regnamespace AND p.proowner = 'claims_owner'::regrole "
            "  AND 'uuid'::regtype = ANY (p.proargtypes::oid[]) "
            "  AND has_function_privilege('claims_app', p.oid, 'EXECUTE')")).fetchall()
    assert {name for name, _ in rows} == set(ACTS_ON_A_CONVERSATION)
    for name, source in rows:
        assert "PERFORM public.bound_conversation(p_conversation);" in source, name
