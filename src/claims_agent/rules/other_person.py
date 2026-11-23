"""Whether a message moves the conversation to a different person.

The fast model is asked whether the caller is asking about a policy that belongs to someone else. For a caller who is
the policyholder that is a different person. A representative, though, always asks about someone else's policy: the
policyholder they act for. So for a representative the reading alone is no switch; the message must also name a
policyholder other than the one the conversation is verified for or being verified against, or be read, by a question
asked only of a representative, as moving to a person other than the one they act for. Reads names only
through the database.
"""

from psycopg import AsyncConnection

from claims_agent.db.identity import name_is_part_of, name_shape
from claims_agent.form.fields import Form, StatedValues
from claims_agent.form.read import confident_yes, picked, unsure
from claims_agent.memory import Memory
from claims_agent.settings import Thresholds


# What a name is, and whether one name is another's, is decided only by the database (db/004_functions.sql, the name
# rule): the words come from its name_shape and the comparison is its name_is_part_of. Nothing here splits or compares
# a name itself.
async def _has_words(conn: AsyncConnection, name: str) -> bool:
    """Whether the name has a word that is not an initial."""
    if not name.strip():
        return False
    return any(len(w) > 1 for w in (await name_shape(conn, name)).key.split(" "))


def acting_for_policyholder(memory: Memory, form: Form, t: Thresholds) -> bool:
    """The caller acts for the policyholder: said so earlier in the conversation, or says so in this message."""
    role = picked(form.caller_role, t) if form.caller_role is not None else None
    return memory.identity.caller_role == "third_party" or role == "third_party"


async def names_someone_else(conn: AsyncConnection, stated: StatedValues, memory: Memory) -> bool:
    """The message names a policyholder who is not the one held: some word of the name said is no word of the held
    name ("Robert Chen" for "Margaret Chen"). Naming her by part of her name ("Margaret") is naming her. With no name
    said, or none held to compare with, nobody else is named."""
    said = stated.policyholder_name or ""
    ident = memory.identity
    held_value = ident.held.get("name") or ident.part_of_name
    held = held_value.as_stated if held_value is not None else ""
    if not await _has_words(conn, said) or not await _has_words(conn, held):
        return False
    return not await name_is_part_of(conn, said, held)


async def names_a_policyholder_not_held(conn: AsyncConnection, stated: StatedValues, memory: Memory) -> bool:
    """A customer has been identified (consent was asked for them) with no policyholder name held, and the message
    names a policyholder: it may or may not be the one the caller acts for. Before then, a name is simply a detail."""
    ident = memory.identity
    return (memory.consent.requested and ident.held.get("name") is None and ident.part_of_name is None
            and await _has_words(conn, stated.policyholder_name or ""))


async def switch_reading(conn: AsyncConnection, memory: Memory, form: Form, stated: StatedValues, t: Thresholds) -> dict:
    """The keyword arguments the every-message rules take about who the message is about: a representative names a
    different policyholder, or is confidently read as moving to a person other than the one they act for; or it is
    unclear (an unsure reading, or a name with none held to compare), and the caller is asked."""
    asked = form.other_than_represented
    other = asked is not None and confident_yes(asked, t)
    maybe = (asked is not None and unsure(asked, t)) or await names_a_policyholder_not_held(conn, stated, memory)
    return {"acting_for_policyholder": acting_for_policyholder(memory, form, t),
            "names_someone_else": await names_someone_else(conn, stated, memory) or other,
            "maybe_someone_else": maybe and not other}
