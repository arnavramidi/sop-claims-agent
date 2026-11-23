"""The conversation's memory: sections that only their owning state may write.

`section(name, writer)` is the only way the rules reach a section for writing, so a state cannot touch
another state's record by accident. `reset_for_new_identity` handles this: when the caller turns out to be
asking about someone else, everything that holds claim data or the caller's own words is wiped.
"""

from pydantic import BaseModel, Field

from claims_agent.states import State


class SectionOwnershipError(Exception):
    pass


class HeldValue(BaseModel):
    as_stated: str
    key: str  # the normalised form
    turn: int
    shape: str | None = None  # a name's shape as the database read it (whole or single); None for other kinds


class IdentityMemory(BaseModel):
    held: dict[str, HeldValue] = {}
    caller_role: str = "not_stated"
    caller_name: str | None = None
    relationship: str | None = None
    tally: dict[str, list[str]] = {}
    wrong_values: list[str] = []
    # A one-word name that was asked for in full: never held, never sent to the database. The same word given again
    # is taken as the caller's whole name (held, compared, counted when wrong). Dropped once a name is held.
    part_of_name: HeldValue | None = None
    # Kinds of detail the caller said they would rather not give: never asked for again in this conversation.
    refused: list[str] = []
    # The caller said they act for someone else and was told they are not listed for that policyholder. A later claim
    # to be the policyholder then goes to a person; kept across a different-person reset, like wrong_values.
    refused_as_representative: bool = False


class ConsentMemory(BaseModel):
    requested: bool = False
    checks_used: int = 0
    last_status: str | None = None


class HintMemory(BaseModel):
    claim_filter: dict | None = None
    claim_number: str | None = None
    need: str | None = None
    need_turn: int | None = None  # the turn the hinted need was first read
    wants_email: bool = False
    # The caller was told the summary will be sent when the conversation ends: the wrap-up sends it without asking.
    # Cleared, with wants_email, once the email is sent or declined, so only a new request brings it back.
    email_promised: bool = False


class FocusMemory(BaseModel):
    claim_filter: dict | None = None
    match_count: int | None = None
    claim_ids: list[str] = []
    last_list: list[str] = []
    need: str | None = None  # the need RESOLVE_INTENT settled
    need_note: str | None = None  # why it differs from what was asked
    email_list_filter: dict | None = None  # set when the caller asks for an overview's full list by email
    # The claims in focus were put there together with nothing describing one of them (the whole policy, or "all of
    # them" in answer to a list): the reply tells each and asks which to go into, instead of offering a follow-up.
    told_together: bool = False


class RequestMemory(BaseModel):
    need: str | None = None
    answered: dict[str, list[str]] = {}
    alternatives_offered: dict[str, list[str]] = {}
    # Follow-ups the caller said no to, per claim: never offered again for that claim (still given when asked for).
    # Not answers: the summary email reads `answered`, never this.
    declined: dict[str, list[str]] = {}


class WrapupMemory(BaseModel):
    topics: list[str] = []
    claims_discussed: list[str] = []
    offers_made: int = 0
    unclear_replies: int = 0
    outcome: str | None = None
    # What had been answered when the email was last sent or declined: a later wrap-up compares it with what has been
    # answered since, so a decision once made stands until there is something new to send.
    answered_when_settled: dict[str, list[str]] = {}


class Counters(BaseModel):
    turn_no: int = 0
    stalled_in_a_row: int = 0
    mood_by_turn: list[str] = []
    context_floor: int = 0
    upset_unmoved: int = 0  # upset turns that moved nothing forward since a person was last offered for it


class Pending(BaseModel):
    kind: str
    label: str
    yes_no: bool


# Section -> the one state that may write it. None means any state. `pending` is the engine's and is
# deliberately not reachable through `section`.
_OWNERS: dict[str, State | None] = {
    "identity": State.VERIFY_ID,
    "consent": State.AWAIT_CONSENT,
    "focus": State.RESOLVE_INTENT,
    "request": State.PROCESS_CASE,
    "wrapup": State.POST_PROCESS,
    "hints": None,
    "counters": None,
}


# Identity fields a reset keeps (an exception to the wipe): the wrong values already counted are the caller's own
# failed guesses, not anyone's data, and the three-wrong-values limit is per conversation. The count itself is the
# database's (db/012_identity_lockout.sql), which a reset never touches; memory keeps a copy to show, read back from
# the database each time details are compared, so the copy does not drop to zero at a reset either.
KEPT_ACROSS_RESET: tuple[str, ...] = ("wrong_values", "refused_as_representative")


class Memory(BaseModel):
    identity: IdentityMemory = Field(default_factory=IdentityMemory)
    consent: ConsentMemory = Field(default_factory=ConsentMemory)
    hints: HintMemory = Field(default_factory=HintMemory)
    focus: FocusMemory = Field(default_factory=FocusMemory)
    request: RequestMemory = Field(default_factory=RequestMemory)
    wrapup: WrapupMemory = Field(default_factory=WrapupMemory)
    counters: Counters = Field(default_factory=Counters)
    pending: Pending | None = None

    @classmethod
    def fresh(cls) -> "Memory":
        return cls()

    def section(self, name: str, writer: State) -> BaseModel:
        if name not in _OWNERS:
            raise SectionOwnershipError(f"{name!r} is not a memory section a state may write")
        owner = _OWNERS[name]
        if owner is not None and owner != writer:
            raise SectionOwnershipError(f"{writer.value} may not write {name!r} (owned by {owner.value})")
        return getattr(self, name)

    def reset_for_new_identity(self) -> None:
        self.identity = IdentityMemory(**{f: getattr(self.identity, f) for f in KEPT_ACROSS_RESET})
        # The consent checks already spent survive the reset for the same reason as wrong_values: the five-check
        # limit is per conversation, or a caller could ask about someone else and re-request consent without end.
        self.consent = ConsentMemory(checks_used=self.consent.checks_used)
        self.hints = HintMemory()
        self.focus = FocusMemory()
        self.request = RequestMemory()
        self.wrapup = WrapupMemory()
        self.pending = None
        self.counters.stalled_in_a_row = 0
        self.counters.context_floor = self.counters.turn_no

    def snapshot(self) -> dict:
        """A deep copy of every section, to hand back to `assert_only_owned_changed` after a state step."""
        return self.model_dump(mode="json")

    def assert_only_owned_changed(self, before: dict, writer: State) -> None:
        """Raise SectionOwnershipError naming a section `writer` does not own but that differs from `before`.
        Catches every route (plain attribute, a kept reference, a nested edit). `hints` and `counters` are
        open to any state; `pending` is the engine's and is not checked here."""
        now = self.snapshot()
        for name, owner in _OWNERS.items():
            if owner is None or owner == writer:
                continue
            if now[name] != before[name]:
                raise SectionOwnershipError(f"{writer.value} changed {name!r} (owned by {owner.value})")
