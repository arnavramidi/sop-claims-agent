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
    event_body_chars: int = 6000  # the longest body of one event record, as JSON text
    events_per_turn: int = 40  # the most event records one turn writes
    replies_prepared: int = 4  # the most follow-up replies worded ahead after one turn
    upset_turns_before_offer: int  # upset turns that move nothing forward before a person is offered
    questions_per_message: int  # questions answered from one message; the rest wait for the next one
    form_asked_chars: int = 40000  # the longest one form's questions and answers may be in a turn's record, as JSON


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
    polarity_forms: dict[str, list[str]] = Field(default_factory=dict)


class PreparedReplies(BaseModel):
    enabled: bool


class Braintrust(BaseModel):
    project: str = "sop-claims-agent"


class Examples(BaseModel):
    """Reviewed example phrasings shown to the form-filling model (see claims_agent.examples)."""
    path: str = "evals/approved_examples.json"
    dataset: str = "approved-examples"
    per_option: int = 5
    total: int = 100
    message_chars: int = 120
    remote_timeout_seconds: float = 5.0


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


class Hosted(BaseModel):
    """Limits that apply only in hosted mode (HOSTED=1), where strangers bring their own model key."""
    max_key_chars: int = 512  # the longest model key accepted in the X-Model-Key header
    conversations_per_minute: int = 10  # per visitor address
    messages_per_minute: int = 30  # per visitor address
    tracked_addresses: int = 10000  # the most visitor addresses the rate limiter remembers at once


class Database(BaseModel):
    """The server's connections. A turn holds one connection and its conversation's row lock for the whole turn, model
    calls included (about two minutes at worst), so these keep a few slow turns from stalling everything else."""
    pool_size: int = 20  # the most connections the server holds at once
    pool_wait_seconds: float = 10  # how long a request waits for a free connection before answering "busy"
    lock_wait_seconds: float = 10  # how long a statement waits for a lock (another turn on the same conversation)
    idle_in_transaction_seconds: float = 150  # the longest pause inside a transaction before the database ends it


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
    examples: Examples = Field(default_factory=Examples)
    hosted: Hosted = Field(default_factory=Hosted)
    database: Database = Field(default_factory=Database)


class Thresholds(BaseModel):
    yes: float
    unsure_low: float
    unsure_high: float
    pick_confident: float
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
