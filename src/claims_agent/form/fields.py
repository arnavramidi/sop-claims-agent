from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ListedClaim:
    """A claim the caller was just shown in a list, as the form is told about it: the same things the list said."""
    case_id: str
    case_type: str | None
    status: str
    created_at: str | None  # ISO date, or None when the claim has no date


class YesNo(BaseModel):
    p: float | None = None  # P(yes); None means refused or omitted
    by: str = "fast"


class Pick(BaseModel):
    value: str | None = None
    p: float = 0.0
    options: dict[str, float] = {}
    by: str = "fast"


class StatedValues(BaseModel):
    policyholder_name: str | None = None
    caller_name: str | None = None
    relationship: str | None = None
    date_of_birth_as_stated: str | None = None
    date_of_birth: list[str] = Field(default_factory=list)  # ISO readings; two when day and month could swap
    phone: str | None = None
    email: str | None = None
    id_digits: str | None = None
    policy_number: str | None = None
    claim_number: str | None = None
    question_parts: list[str] = Field(default_factory=list)  # each question the message asks, when asked for


class Form(BaseModel):
    wants_person: YesNo
    about: Pick
    skips_step: YesNo
    asks_why_step: YesNo
    declines: YesNo
    mood: Pick
    pending_reply: Pick
    done: YesNo
    english: YesNo
    other_person_policy: YesNo
    wants_email_summary: YesNo
    shaped_need: list[YesNo]
    caller_role: Pick | None = None
    gave_identity_detail: YesNo | None = None
    correcting_detail: YesNo | None = None
    partial_detail: YesNo | None = None
    consent_recheck: YesNo | None = None
    relationship: Pick | None = None
    mentions_claim: YesNo
    quoted_claim_number: YesNo
    how_many: Pick
    claim_types: dict[str, YesNo]
    claim_statuses: dict[str, YesNo]
    months: dict[int, YesNo]
    years: dict[int, YesNo]
    no_year_said: YesNo
    relative_time: Pick
    search_relation: Pick
    listed_claim: Pick
    need: Pick
    answer_wanted: Pick
    multi_question: YesNo
    which_document: Pick
    cant_get_document: YesNo
    asks_personal_detail: YesNo
    email_offer_reply: Pick | None = None
    # Stage 2 (Ruling 22): asked only after verification; the defaults mean "nothing asked".
    calculation: Pick = Field(default_factory=Pick)
    figures: dict[str, YesNo] = Field(default_factory=dict)
    no_figure_said: YesNo = Field(default_factory=lambda: YesNo(p=0.0))
    other_than_represented: YesNo | None = None  # asked only of a caller acting for a policyholder
