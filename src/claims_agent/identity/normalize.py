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

_WORD_BREAK = re.compile(r"[\s,;]+")
_ID_SEPARATORS = re.compile(r"[\s.\-]+")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")  # one @ with a dot after it; the loader's own rule
_KEY_DROP = re.compile(r"[^A-Za-z0-9]")


def name_candidates(stated: str) -> tuple[str, ...]:
    """Lower-cased, letters only, every ordering of the words joined with no spaces.
    Beyond four words: as typed and reversed."""
    words = [w for w in ("".join(c for c in part.lower() if c.isalpha()) for part in _WORD_BREAK.split(stated)) if w]
    if not words:
        return ()
    orders = permutations(words) if len(words) <= MAX_PERMUTED_WORDS else (words, words[::-1])
    return tuple(dict.fromkeys("".join(o) for o in orders))


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
    # A stated date has at most two readings; the database refuses more.
    return tuple(real[:MAX_DOB_READINGS])


def _phone(stated: str, bad: list[FormatFailure]) -> str | None:
    digits = _digits(stated)
    if len(digits) < MIN_PHONE_DIGITS:
        bad.append(FormatFailure("phone", stated, "fewer than ten digits"))
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
        bad.append(FormatFailure("id", stated, "needs exactly four digits or the full number"))
    return None


def normalise(v: StatedValues) -> tuple[StatedIdentity, list[FormatFailure]]:
    bad: list[FormatFailure] = []
    name, phone, email = _given(v.policyholder_name), _given(v.phone), _given(v.email)
    id_digits, policy = _given(v.id_digits), _given(v.policy_number)
    identity = StatedIdentity(
        names=name_candidates(name) if name else (),
        dobs=_dobs([r for r in v.date_of_birth if r and r.strip()], _given(v.date_of_birth_as_stated), bad),
        phone=_phone(phone, bad) if phone else None,
        email=_email(email, bad) if email else None,
        id_last4=_id_last4(id_digits, bad) if id_digits else None,
        policy=(claim_number_key(policy) or None) if policy else None,
    )
    return identity, bad
