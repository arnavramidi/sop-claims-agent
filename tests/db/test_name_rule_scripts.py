"""One name rule, in the database: names in any script, typed with or without accents, in capitals, with a hyphen, an
apostrophe or an invisible character, are read the same way when the server asks what a name is (name_shape) and
when it is compared. Each name either verifies against the same name on file or is asked for in full; a name that is
in fact right is never counted as a wrong value."""

import uuid

import pytest
from helpers import GAVE, instructions, make_deps, make_form, memory, new_conversation, say, stated, verified_party

from claims_agent.db.identity import match_identity, name_shape
from claims_agent.identity.types import StatedIdentity
from claims_agent.rules import verify_id

pytestmark = pytest.mark.db

# The data set's customers whose names are not plain Latin letters, with two more of their details.
ON_FILE = {
    "P7": ("राम शर्मा", ["1990-08-21"], "9180"),               # Devanagari, a conjunct with a virama
    "P12": ("مُحَمَّد عَلِي", ["1964-09-10"], "6688"),          # Arabic with vowel marks
    "P13": ("Νικόλαος Παπαδόπουλος", ["1989-12-03"], "5317"),  # Greek, ending in a final sigma
    "P30": ("José García-Núñez", ["1979-11-02"], "3030"),      # accents and a hyphen
    "P31": ("Seán O’Brien", ["1981-01-31"], "3131"),           # an accent and a curly apostrophe
}

# How a caller may type each of those names: each is the same name.
SAID = [
    ("P7", "राम शर्मा"),
    ("P7", "शर्मा राम"),
    ("P12", "مُحَمَّد عَلِي"),
    ("P13", "Νικόλαος Παπαδόπουλος"),
    ("P13", "ΝΙΚΌΛΑΟΣ ΠΑΠΑΔΌΠΟΥΛΟΣ"),
    ("P13", "ΝΙΚΟΛΑΟΣ ΠΑΠΑΔΟΠΟΥΛΟΣ"),
    ("P30", "José García-Núñez"),
    ("P30", "José García-Núñez"),  # the accents typed as separate marks
    ("P30", "Jose Garcia Nunez"),
    ("P31", "Seán O'Brien"),
    ("P31", "Sean OBrien"),
    ("P9", "Margaret​ Chen"),  # an invisible character beside a space
]


@pytest.fixture
async def scripts_db(load_dataset):
    return await load_dataset("non_latin_names")


@pytest.fixture
async def scripts_deps(scripts_db):
    return make_deps(scripts_db)


async def _shape(db, name: str):
    async with db.app.conversation(None) as c:
        return await name_shape(c, name)


async def _named(db, name: str) -> set[str]:
    cid = uuid.uuid4()  # a conversation of its own: the database counts wrong values per conversation
    async with db.app.conversation(cid) as c:
        s = StatedIdentity(name=name, dobs=(), phone=None, email=None, id_last4=None, policy=None)
        return {m.party_id for m in await match_identity(c, cid, s) if "name" in m.kinds}


@pytest.mark.parametrize("party,said", SAID)
async def test_each_way_of_typing_a_name_is_whole_and_matches_the_name_on_file(scripts_db, party, said):
    assert (await _shape(scripts_db, said)).shape == "whole"
    assert await _named(scripts_db, said) == {party}


@pytest.mark.parametrize("party,said", SAID)
async def test_the_same_name_typed_another_way_has_the_same_key(scripts_db, party, said):
    assert (await _shape(scripts_db, said)).key == (await _shape(scripts_db, ON_FILE.get(party, ("Margaret Chen",))[0])).key


@pytest.mark.parametrize("said,shape", [
    ("Margaret Chen", "whole"), ("Margaret-Chen", "whole"), ("Mary-Jane O'Neil", "whole"), ("Margaret C. Chen", "whole"),
    ("MargaretChen", "single"), ("Margaret", "single"), ("Oyelaran", "single"), ("कृष्ण", "single"),
    ("Mar​garet", "single"),          # an invisible character inside one word does not split it
    ("Margaret​Chen", "single"),      # nor does it separate two words
    ("M. Chen", "partial"), ("Margaret C", "partial"), ("  ,  ", "partial"),
])
async def test_the_shape_of_a_name(scripts_db, said, shape):
    assert (await _shape(scripts_db, said)).shape == shape


async def test_a_one_token_full_name_never_matches_a_two_word_name_on_file(scripts_db):
    assert await _named(scripts_db, "MargaretChen") == set() and await _named(scripts_db, "Margaret-Chen") == {"P9"}


@pytest.mark.parametrize("party", sorted(ON_FILE))
async def test_a_caller_verifies_with_their_name_as_it_is_on_file(scripts_deps, party):
    name, dob, id_digits = ON_FILE[party]
    cid = await new_conversation(scripts_deps)
    await say(scripts_deps, cid, f"{name}, {dob[0]}, {id_digits}", make_form(**GAVE),
              stated(policyholder_name=name, date_of_birth=dob, id_digits=id_digits))
    assert await verified_party(scripts_deps, cid) == party
    assert await memory(scripts_deps, cid, "identity.wrong_values") == []


@pytest.mark.parametrize("party,said", [p for p in SAID if p[0] != "P9"])
async def test_a_name_typed_another_way_verifies_and_is_never_counted_wrong(scripts_deps, party, said):
    _, dob, id_digits = ON_FILE[party]
    cid = await new_conversation(scripts_deps)
    await say(scripts_deps, cid, f"{said}, {dob[0]}, {id_digits}", make_form(**GAVE),
              stated(policyholder_name=said, date_of_birth=dob, id_digits=id_digits))
    assert await verified_party(scripts_deps, cid) == party
    assert await memory(scripts_deps, cid, "identity.wrong_values") == []


@pytest.fixture
def sent(monkeypatch):
    """Every identity the VERIFY_ID step sends to the database for comparison, in order."""
    seen = []
    real = verify_id.match_identity

    async def recording(conn, conversation_id, s):
        seen.append(s)
        return await real(conn, conversation_id, s)
    monkeypatch.setattr(verify_id, "match_identity", recording)
    return seen


async def test_a_first_name_split_by_an_invisible_character_is_asked_for_in_full_never_compared(scripts_deps, sent):
    cid = await new_conversation(scripts_deps)
    r = await say(scripts_deps, cid, "Mar​garet, 1985-03-15, 4472", make_form(**GAVE),
                  stated(policyholder_name="Mar​garet", date_of_birth=["1985-03-15"], id_digits="4472"))
    assert "ask_full_value" in instructions(r) and all(s.name is None for s in sent)
    assert await verified_party(scripts_deps, cid) is None
    assert await memory(scripts_deps, cid, "identity.wrong_values") == []
    await say(scripts_deps, cid, "Margaret Chen", make_form(**GAVE), stated(policyholder_name="Margaret Chen"))
    assert await verified_party(scripts_deps, cid) == "P9"


async def test_the_name_is_sent_as_the_caller_said_it(scripts_deps, sent):
    cid = await new_conversation(scripts_deps)
    said = "ΝΙΚΟΛΑΟΣ ΠΑΠΑΔΟΠΟΥΛΟΣ"
    await say(scripts_deps, cid, said, make_form(**GAVE), stated(policyholder_name=said))
    assert sent[-1].name == said
    held = (await memory(scripts_deps, cid, "identity.held"))["name"]
    assert held.as_stated == said and held.shape == "whole"


# ---- the cases the code's own copy of the rule was tested with, now held against the database's one rule ----------

@pytest.mark.parametrize("said,key", [
    ("Mary-Jane O'Neil", "jane mary oneil"), ("Margaret.Chen", "chen margaret"), ("Margaret-Chen", "chen margaret"),
    ("margaret, chen", "chen margaret"), ("MargaretChen", "margaretchen"), ("O'Neil, José", "jose oneil"),
    ("Li Ya-Wen", "li wen ya"), ("Li Ya Wen", "li wen ya"),
])
async def test_the_words_of_a_name_whatever_separates_them(scripts_db, said, key):
    """Hyphens, dots and commas separate words; an apostrophe does not; the key ignores the order of the words."""
    assert (await _shape(scripts_db, said)).key == key


@pytest.mark.parametrize("said,shape", [
    ("Margaret.Chen", "whole"), ("Li Ya Wen", "whole"), ("M", "partial"), ("Margaret Chen", "whole"),
])
async def test_more_shapes(scripts_db, said, shape):
    assert (await _shape(scripts_db, said)).shape == shape


async def _matches(db, said: str, stored: str) -> bool:
    import psycopg
    async with await psycopg.AsyncConnection.connect(db.superuser_url) as su:
        return (await (await su.execute("SELECT public.name_matches(%s, %s)", [said, stored])).fetchone())[0]


async def test_names_of_up_to_four_words_match_in_any_order_and_longer_ones_as_said_or_reversed(scripts_db):
    assert await _matches(scripts_db, "Cy Di Ann Bo", "Ann Bo Cy Di")
    assert await _matches(scripts_db, "Ed Di Cy Bo Ann", "Ann Bo Cy Di Ed")
    assert not await _matches(scripts_db, "Bo Ann Cy Di Ed", "Ann Bo Cy Di Ed")
