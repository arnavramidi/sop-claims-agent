"""[data] reset_on_start: on (the default) each start rebuilds the database and reloads the data; off, a database built
from these same migration files keeps its conversations, and one built from other files is refused untouched."""

import psycopg
import pytest
from psycopg import sql

from claims_agent import setup
from claims_agent.db import migrate
from claims_agent.db import conversations as store
from claims_agent.settings import load_settings

pytestmark = pytest.mark.db


def _settings(reset: bool):
    s = load_settings()
    return s.model_copy(update={"data": s.data.model_copy(update={"reset_on_start": reset})})


def _count(url: str, table: str) -> int:
    with psycopg.connect(url) as conn:
        return conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]


@pytest.fixture
async def one_conversation(test_db):
    async with test_db.app.conversation(None) as c:
        await store.create(c)
    await test_db.app.close()
    return test_db


def test_the_setting_defaults_to_a_reset():
    assert load_settings().data.reset_on_start is True


def test_with_reset_off_the_database_and_its_conversations_are_kept(one_conversation):
    db = one_conversation
    line = setup.prepare(db.superuser_url, db.loader_url, _settings(False))
    assert line == "kept the database and its conversations as they are (reset_on_start = false)"
    assert _count(db.loader_url, "conversations") == 1


def test_with_reset_on_the_database_is_rebuilt_and_the_data_reloaded(one_conversation):
    db = one_conversation
    line = setup.prepare(db.superuser_url, db.loader_url, _settings(True))
    assert line.startswith("loaded ")
    assert _count(db.loader_url, "conversations") == 0 and _count(db.loader_url, "claims") > 0


@pytest.mark.parametrize("comment", ["claims_agent build 0000", None])
def test_with_reset_off_a_database_built_from_other_files_is_refused_untouched(one_conversation, comment):
    db = one_conversation
    stamp = migrate.build_stamp()
    with psycopg.connect(db.superuser_url, autocommit=True) as su:
        su.execute(sql.SQL("COMMENT ON SCHEMA public IS {}").format(sql.Literal(comment)))
        try:
            with pytest.raises(migrate.BuiltFromOtherFiles):
                setup.prepare(db.superuser_url, db.loader_url, _settings(False))
            assert _count(db.loader_url, "conversations") == 1
        finally:
            su.execute(sql.SQL("COMMENT ON SCHEMA public IS {}").format(sql.Literal(stamp)))
