"""The database compares a name only when it is whole by the shared word rule (two words that are more than an
initial), or when it is one word and the name on file is itself one word. A one-token full name ("MargaretChen")
never matches a two-word name on file, whatever the calling code sends."""

import pytest

from claims_agent.db.identity import match_identity
from claims_agent.identity.types import StatedIdentity

pytestmark = pytest.mark.db


async def _name_matches(db, token: str) -> set[str]:
    s = StatedIdentity(names=(token,), dobs=(), phone=None, email=None, id_last4=None, policy=None)
    async with db.app.conversation(None) as c:
        return {m.party_id for m in await match_identity(c, s) if "name" in m.kinds}


@pytest.mark.parametrize("token,matches", [
    ("Margaret Chen", True), ("margaret chen", True), ("Margaret-Chen", True), ("Margaret.Chen", True),
    ("MargaretChen", False), ("margaretchen", False), ("Margaret", False), ("Chen", False),
])
async def test_each_form_of_margarets_name(load_dataset, token, matches):
    db = await load_dataset("starter")
    assert ("P9" in await _name_matches(db, token)) is matches


async def test_a_one_word_name_matches_a_one_word_name_on_file(load_dataset):
    db = await load_dataset("one_word_name")
    assert "P7" in await _name_matches(db, "Oyelaran")
    assert "P7" in await _name_matches(db, "oyelaran")
