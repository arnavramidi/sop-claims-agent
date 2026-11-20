"""Whether a message moves the conversation to a different person.

The fast model is asked whether the caller is asking about a policy that belongs to someone else. For a caller who is
the policyholder that is a different person. A representative, though, always asks about someone else's policy: the
policyholder they act for. So for a representative the reading alone is no switch; the message must also name a
policyholder other than the one the conversation is verified for or being verified against, or be read, by a question
asked only of a representative, as moving to a person other than the one they act for. Pure: no database, no
model.
"""

from claims_agent.form.fields import Form, StatedValues
from claims_agent.form.read import confident_yes, picked, unsure
from claims_agent.identity.normalize import name_words
from claims_agent.memory import Memory
from claims_agent.settings import Thresholds

# A word of a name this long or longer may differ by one letter and still be the same word (a typo), as the
# database's own name comparison allows.
TYPO_MIN_LETTERS = 6


def _words(name: str) -> list[str]:
    """The name's words that are not initials, by the one name-word rule (identity/normalize.py)."""
    return [w for w in name_words(name) if len(w) > 1]


def _one_edit(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _same_word(said: str, held: str) -> bool:
    return said == held or (min(len(said), len(held)) >= TYPO_MIN_LETTERS and _one_edit(said, held))


def acting_for_policyholder(memory: Memory, form: Form, t: Thresholds) -> bool:
    """The caller acts for the policyholder: said so earlier in the conversation, or says so in this message."""
    role = picked(form.caller_role, t) if form.caller_role is not None else None
    return memory.identity.caller_role == "third_party" or role == "third_party"


def names_someone_else(stated: StatedValues, memory: Memory) -> bool:
    """The message names a policyholder who is not the one held: some word of the name said is no word of the held
    name ("Robert Chen" for "Margaret Chen"). Naming her by part of her name ("Margaret") is naming her. With no name
    said, or none held to compare with, nobody else is named."""
    said = _words(stated.policyholder_name or "")
    ident = memory.identity
    held_value = ident.held.get("name") or ident.part_of_name
    held = _words(held_value.as_stated) if held_value is not None else []
    if not said or not held:
        return False
    return not all(any(_same_word(w, h) for h in held) for w in said)


def names_a_policyholder_not_held(stated: StatedValues, memory: Memory) -> bool:
    """A customer has been identified (consent was asked for them) with no policyholder name held, and the message
    names a policyholder: it may or may not be the one the caller acts for. Before then, a name is simply a detail."""
    ident = memory.identity
    return (memory.consent.requested and bool(_words(stated.policyholder_name or ""))
            and ident.held.get("name") is None and ident.part_of_name is None)


def switch_reading(memory: Memory, form: Form, stated: StatedValues, t: Thresholds) -> dict:
    """The keyword arguments the every-message rules take about who the message is about: a representative names a
    different policyholder, or is confidently read as moving to a person other than the one they act for; or it is
    unclear (an unsure reading, or a name with none held to compare), and the caller is asked."""
    asked = form.other_than_represented
    other = asked is not None and confident_yes(asked, t)
    maybe = (asked is not None and unsure(asked, t)) or names_a_policyholder_not_held(stated, memory)
    return {"acting_for_policyholder": acting_for_policyholder(memory, form, t),
            "names_someone_else": names_someone_else(stated, memory) or other,
            "maybe_someone_else": maybe and not other}
