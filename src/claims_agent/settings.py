import os
import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class Limits(BaseModel):
    stalled_turns: int
    wrong_values: int
    consent_checks: int
    claims_in_focus: int
    claims_prepared: int
    unclear_email_replies: int
    model_retries: int
    model_timeout_seconds: int
    message_chars: int = 4000  # at most this much of one caller message goes into any model request
    documents_offered: int = 25  # at most this many outstanding documents are offered to the form as choices


class Today(BaseModel):
    date: date


class Data(BaseModel):
    folder: str
    reset_on_start: bool


class Models(BaseModel):
    fast: str
    text: str
    larger: str
    wording_max_output_tokens: int
    wording_reasoning_effort: str
    wording_verbosity: str
    extractor_max_output_tokens: int
    extractor_reasoning_effort: str


class Claims(BaseModel):
    in_progress_statuses: list[str]


class ReplyCheck(BaseModel):
    wary_status_words: list[str]
    status_forms: dict[str, list[str]] = Field(default_factory=dict)
    max_length: int = 1500


class PreparedReplies(BaseModel):
    enabled: bool


class Braintrust(BaseModel):
    project: str = "sop-claims-agent"


class Freedom(BaseModel):
    claim_facts: bool
    settle_unsure: bool
    prepared_replies: bool


class StateSetting(BaseModel):
    freedom: str


class Need(BaseModel):
    opens_with: list[str]
    opens_with_when_denied: list[str] | None = None


class Special(BaseModel):
    with_deadline: bool = False
    drop_when_deadline_passed: bool = False
    documents_as_requirements: bool = False
    per_document_alternatives: bool = False


class Followups(BaseModel):
    order: list[str] = Field(default_factory=list)
    comes_after: dict[str, list[str]] = Field(default_factory=dict)
    special: dict[str, Special] = Field(default_factory=dict)


class Settings(BaseModel):
    limits: Limits
    today: Today
    data: Data
    models: Models
    claims: Claims
    reply_check: ReplyCheck
    prepared_replies: PreparedReplies
    freedom: dict[str, Freedom]
    states: dict[str, StateSetting]
    needs: dict[str, Need]
    followups: Followups
    braintrust: Braintrust = Field(default_factory=Braintrust)


class Thresholds(BaseModel):
    yes: float
    unsure_low: float
    unsure_high: float
    pick_confident: float
    filter_include_min: float
    send_email_min: float
    prepared_reply_min: float


def _read(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_settings(path: Path | None = None) -> Settings:
    path = path or Path(os.environ.get("SETTINGS_FILE") or ROOT / "settings.toml")
    return Settings.model_validate(_read(path))


def load_thresholds(path: Path | None = None) -> Thresholds:
    return Thresholds.model_validate(_read(path or ROOT / "thresholds.toml"))
