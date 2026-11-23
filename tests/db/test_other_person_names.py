"""Whether a caller acting for a policyholder has named a different policyholder is decided by the database's name
rule: a name typed without its accents, or in a non-Latin script, is the same policyholder; another name is not."""

import pytest
from helpers import stated

from claims_agent.memory import HeldValue, Memory
from claims_agent.rules.other_person import names_a_policyholder_not_held, names_someone_else

pytestmark = pytest.mark.db


def _holding(name: str | None) -> Memory:
    m = Memory()
    if name is not None:
        m.identity.held["name"] = HeldValue(as_stated=name, key=name, turn=1, shape="whole")
    return m


async def _other(test_db, said: str | None, held: str | None) -> bool:
    async with test_db.app.conversation(None) as conn:
        return await names_someone_else(conn, stated(policyholder_name=said), _holding(held))


@pytest.mark.parametrize("said,held", [
    ("Margaret Chen", "Margaret Chen"),
    ("chen margaret", "Margaret Chen"),
    ("Margaret", "Margaret Chen"),                       # part of the name is still naming her
    ("Margret Chen", "Margaret Chen"),                   # one letter off in a long word
    ("Jose Garcia Nunez", "José García-Núñez"),          # accents left off
    ("José García-Núñez", "Jose Garcia-Nunez"),          # accents typed as separate marks
    ("राम शर्मा", "राम शर्मा"),                              # Devanagari
    ("शर्मा राम", "राम शर्मा"),
    ("ΝΙΚΟΛΑΟΣ ΠΑΠΑΔΟΠΟΥΛΟΣ", "Νικόλαος Παπαδόπουλος"),  # Greek capitals without accents
    ("عَلِي مُحَمَّد", "مُحَمَّد عَلِي"),                      # Arabic, words in the other order
])
async def test_the_same_policyholder_is_not_someone_else(test_db, said, held):
    assert await _other(test_db, said, held) is False


@pytest.mark.parametrize("said,held", [
    ("Robert Chen", "Margaret Chen"),
    ("Robert", "Margaret Chen"),
    ("Jose Garcia Perez", "José García-Núñez"),
    ("राम वर्मा", "राम शर्मा"),
    ("Νικόλαος Γεωργίου", "Νικόλαος Παπαδόπουλος"),
])
async def test_a_different_name_is_someone_else(test_db, said, held):
    assert await _other(test_db, said, held) is True


@pytest.mark.parametrize("said,held", [(None, "Margaret Chen"), ("Robert Chen", None), ("J.", "Margaret Chen"), ("", "Margaret Chen")])
async def test_with_no_name_said_or_held_nobody_else_is_named(test_db, said, held):
    assert await _other(test_db, said, held) is False


async def test_a_name_with_only_initials_is_no_name_to_hold_against(test_db):
    async with test_db.app.conversation(None) as conn:
        m = _holding(None)
        m.consent.requested = True
        assert await names_a_policyholder_not_held(conn, stated(policyholder_name="J. K."), m) is False
        assert await names_a_policyholder_not_held(conn, stated(policyholder_name="José García"), m) is True
