"""What a caller stated about who they are, ready to be compared inside the database."""

from dataclasses import dataclass
from datetime import date

# The kinds of detail that count towards verification. The policy number is not one of them.
KINDS = ("name", "dob", "phone", "email", "id")

# Limits the database enforces too, so one call cannot test many guesses at once.
MAX_NAME_CANDIDATES = 24  # every ordering of four words
MAX_DOB_READINGS = 2  # a date whose day and month could swap


@dataclass(frozen=True)
class StatedIdentity:
    names: tuple[str, ...]  # name_candidates of the stated policyholder name
    dobs: tuple[date, ...]  # the real-date readings
    phone: str | None  # digits only, ten or more
    email: str | None  # trimmed, lower-cased
    id_last4: str | None  # four digits
    policy: str | None  # claim_number_key of the stated policy number


@dataclass(frozen=True)
class FormatFailure:
    kind: str
    as_stated: str
    reason: str


@dataclass(frozen=True)
class CustomerMatch:
    party_id: str
    kinds: frozenset[str]
    policy_ok: bool
