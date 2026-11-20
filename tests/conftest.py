from pathlib import Path

import pytest
from dbsupport import SchemaLeak, SessionBusy, TestDatabase, TestDb
from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, make_deps, make_form, new_conversation, say, verified_party

from claims_agent.loader.read import read_folder
from claims_agent.loader.write import write
from claims_agent.settings import ROOT
from claims_agent.states import State

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def session_db():
    """The test database, migrated once for the whole session (only when a test asks for the database). The
    whole run stops if another session holds it, or if the accounts or the database are not in the state a
    migration leaves. At the end, whatever the last test did is undone."""
    try:
        database = TestDatabase()
    except (SessionBusy, SchemaLeak) as e:
        pytest.exit(str(e), returncode=1)
    try:
        yield database
        database.finish()
    finally:
        database.close()


@pytest.fixture
def clean_db(session_db, request):
    """Before a database test: other sessions ended, the session's schema (migrated again if an earlier test
    changed it), every table empty and every sequence at its start. After it: a change the test made to the
    schema or the roles is undone, and one a migration cannot undo fails this test. Tests never rebuild the
    schema themselves."""
    yield session_db.start_test()
    session_db.after_test(request.node.nodeid)


@pytest.fixture
async def load_dataset(session_db, clean_db):
    """Factory: `await load_dataset("bad_data")` empties the test database and loads tests/data/<name>
    (the starter data for "starter") through the real loader."""
    made: list[TestDb] = []

    async def load(name: str = "starter") -> TestDb:
        db = session_db.reset()
        made.append(db)
        folder = ROOT / "data" / "starter" if name == "starter" else DATA / name
        await write(read_folder(folder), db.loader_url)
        return db

    yield load
    for db in made:
        await db.close()


# ---- whole-turn fixtures (Task 11). Every conversation is built by talking through `say`. ----

@pytest.fixture
async def deps(load_dataset):
    """Engine dependencies over the starter data: scripted filler, extractor and wording."""
    return make_deps(await load_dataset("starter"))


@pytest.fixture
async def conv(deps):
    """A new conversation, unverified, in VERIFY_ID."""
    return await new_conversation(deps)


@pytest.fixture
async def verified_conv(deps):
    """Verified as P9 (Margaret Chen) by the brief's own message; now past VERIFY_ID."""
    cid = await new_conversation(deps)
    await say(deps, cid, BRIEF_MSG, make_form(**GAVE), BRIEF_SAID)
    assert await verified_party(deps, cid) == "P9"
    return cid


@pytest.fixture
async def escalated_conv(deps):
    """Handed over on the first message: the caller asked for a person."""
    cid = await new_conversation(deps)
    assert (await say(deps, cid, "I want to talk to a person", make_form(wants_person=True))).state == State.ESCALATED
    return cid


@pytest.fixture
async def deps_with_switch(load_dataset):
    """Engine dependencies whose filler is a SwitchingFiller over two scripted fillers (decisions, then structured)."""
    from helpers import SwitchingTestFiller
    from claims_agent.form.filler import ScriptedFiller
    filler = SwitchingTestFiller(ScriptedFiller(name="decisions"), ScriptedFiller(name="structured"))
    return make_deps(await load_dataset("starter"), filler=filler)
