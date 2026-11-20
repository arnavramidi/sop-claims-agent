"""Turn stated values into the forms the database compares, and say which ones are malformed.

Only the caller's own words are handled here. The stored side is normalised the same way inside the
database (db/004_functions.sql), which re-checks every rule below and never trusts this module.
"""

import re
import unicodedata
from datetime import date
from itertools import permutations

from claims_agent.form.fields import StatedValues
from claims_agent.identity.types import MAX_DOB_READINGS, FormatFailure, StatedIdentity

MAX_PERMUTED_WORDS = 4
MIN_PHONE_DIGITS = 10
FULL_ID_DIGITS = 9

_ID_SEPARATORS = re.compile(r"[\s.\-]+")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")  # one @ with a dot after it; the loader's own rule
_KEY_DROP = re.compile(r"[^A-Za-z0-9]")


# THE NAME-WORD RULE, one rule in two places: name_words below and public.name_words in db/004_functions.sql. A
# name is lower-cased; apostrophes (' ’ ʼ) are dropped, so "O'Neil" is one word; every other character that is not a
# letter (space, hyphen, dot, comma, digit, ...) separates words, so "Mary-Jane O'Neil" is three words (mary, jane,
# oneil) and "MargaretChen" is one. A one-letter word is an initial. A name is WHOLE when it has two or more words that
# are not initials; it is SINGLE when it is exactly one word that is not an initial. The database compares a stated
# name only when it is whole, or single against a name on file that is itself single; nothing else is ever compared.
_APOSTROPHES = "'\u2019\u02bc"


def name_words(stated: str) -> list[str]:
    """The words of a name by the name-word rule above."""
    words, word = [], []
    for c in stated.lower():
        if c in _APOSTROPHES:
            continue
        if c.isalpha():
            word.append(c)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    return words


def is_whole_name(stated: str) -> bool:
    """A full name, as the verification rule asks for: two or more words that are not initials (the name-word rule)."""
    return sum(1 for w in name_words(stated) if len(w) > 1) >= 2


def is_single_name(stated: str) -> bool:
    """Exactly one word that is not an initial: the whole name only of a customer whose name on file is one word."""
    words = name_words(stated)
    return len(words) == 1 and len(words[0]) > 1


def name_candidates(stated: str) -> tuple[str, ...]:
    """Every ordering of the name's words (the name-word rule), each joined by one space so the database can count
    the words again. Beyond four words: as typed and reversed."""
    words = name_words(stated)
    if not words:
        return ()
    orders = permutations(words) if len(words) <= MAX_PERMUTED_WORDS else (words, words[::-1])
    return tuple(dict.fromkeys(" ".join(o) for o in orders))


def claim_number_key(stated: str) -> str:
    """Upper-case letters and digits only (the same rule runs on the stored side)."""
    return _KEY_DROP.sub("", stated).upper()


def _given(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _digits(text: str) -> str:
    """Every decimal digit in the text, written as ASCII."""
    return "".join(str(unicodedata.decimal(c)) for c in text if c.isdecimal())


def _dobs(readings: list[str], as_stated: str | None, bad: list[FormatFailure]) -> tuple[date, ...]:
    real: list[date] = []
    for reading in readings:
        try:
            d = date.fromisoformat(reading.strip())
        except ValueError:
            continue
        if d not in real:
            real.append(d)
    if readings and not real:
        bad.append(FormatFailure("dob", as_stated or ", ".join(readings), "not a real date"))
    elif as_stated and not readings:  # said, but with no whole day in it ("1985", "March 1985")
        bad.append(FormatFailure("dob", as_stated, "not a whole date", partial=True))
    # A stated date has at most two readings; the database refuses more.
    return tuple(real[:MAX_DOB_READINGS])


def _phone(stated: str, bad: list[FormatFailure]) -> str | None:
    digits = _digits(stated)
    if len(digits) < MIN_PHONE_DIGITS:
        bad.append(FormatFailure("phone", stated, "fewer than ten digits", partial=True))
        return None
    return digits


def _email(stated: str, bad: list[FormatFailure]) -> str | None:
    email = stated.lower()
    if not _EMAIL.fullmatch(email):
        bad.append(FormatFailure("email", stated, "not an email address"))
        return None
    return email


def _id_last4(stated: str, bad: list[FormatFailure]) -> str | None:
    compact = _ID_SEPARATORS.sub("", stated)
    digits = _digits(compact)
    if len(digits) != len(compact):
        bad.append(FormatFailure("id", stated, "not only digits"))
    elif len(digits) == 4 or len(digits) >= FULL_ID_DIGITS:
        return digits[-4:]
    else:
        bad.append(FormatFailure("id", stated, "needs exactly four digits or the full number", partial=True))
    return None


def normalise(v: StatedValues) -> tuple[StatedIdentity, list[FormatFailure]]:
    bad: list[FormatFailure] = []
    name, phone, email = _given(v.policyholder_name), _given(v.phone), _given(v.email)
    id_digits, policy = _given(v.id_digits), _given(v.policy_number)
    if name and not is_whole_name(name):
        bad.append(FormatFailure("name", name, "only part of a name", partial=True))
    identity = StatedIdentity(
        names=name_candidates(name) if name and is_whole_name(name) else (),
        dobs=_dobs([r for r in v.date_of_birth if r and r.strip()], _given(v.date_of_birth_as_stated), bad),
        phone=_phone(phone, bad) if phone else None,
        email=_email(email, bad) if email else None,
        id_last4=_id_last4(id_digits, bad) if id_digits else None,
        policy=(claim_number_key(policy) or None) if policy else None,
    )
    return identity, bad
