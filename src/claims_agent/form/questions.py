from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from claims_agent.form.fields import Form, ListedClaim, Pick, YesNo
from claims_agent.identity.types import KINDS
from claims_agent.loader.read import MONEY_FIELDS
from claims_agent.form.read import PickAnswer, Refused, YesNoAnswer
from claims_agent.states import State

# What a caller can ask code to work out. Kept here so the form needs nothing from the answers package.
CALCULATIONS = ("none", "total", "difference", "highest", "lowest", "count", "days_open", "days_left")

# Each calculation described, so that "my most expensive claim" or "the top half of my claims" reads as a ranking.
CALCULATION_MEANS = {
    "none": "nothing to be worked out",
    "total": "a total of a money figure",
    "difference": "the difference between two money figures",
    "highest": "the highest, largest, most expensive or top claims by a money figure",
    "lowest": "the lowest, smallest, cheapest or bottom claims by a money figure",
    "count": "how many claims",
    "days_open": "how many days a claim has been open",
    "days_left": "how many days are left before a deadline",
}

RECENT_WINDOW = 6  # how many earlier caller messages a request carries (and asks a question about)

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

# Every question is about the message being read. The rest of the data record (earlier messages, the agent's reply,
# the money figures' meanings) is there to resolve what the message refers to; without this, a reading carries over
# from it: "Thanks!" read as asking for the email the caller asked for a turn earlier, a joke read as mentioning the
# claim being discussed, a type read into "all my claims" from the figures' wording. The money figures are there to
# tell which figure the caller's words describe, so the rule says that rather than that they are not in the message:
# said that way, the figure questions stopped matching the caller's words to them ("what you actually paid" read 0.07
# as the net pay).
THIS_MESSAGE = ("Every question is about caller_message: what that message itself says, names or asks. The other fields "
                "are there only so you can tell what caller_message refers to (a bare \"yes\", \"that one\", \"the "
                "other claim\"). Something said only in an earlier message or by the agent is not said in "
                "caller_message; a question about the earlier messages says so. money_figures only tells which money "
                "figure the caller's words describe: its wording is never the caller's.")

# A month or year inside a date of birth describes the caller, never a claim.
NOT_A_CLAIM_DATE = "A date of birth is not the time of a claim."

# A caller who simply gives their own name or another detail is identifying the policyholder unless they say they
# are calling for someone else; without this a bare "my name is ..." reads as no detail at all. A representative who
# gives the policyholder's details gives details too (the question once read as "only the policyholder's own").
OWN_DETAILS_COUNT = ("Details the caller gives about the policyholder they are calling for count. A caller's own "
                     "name, date of birth, phone number, email address, ID digits or policy number count when the caller "
                     "is the policyholder or does not say otherwise; a caller who says they are calling for someone else "
                     "and gives only their own name gives none.")

# A bare value answering the agent's question is a detail ("4472" right after the agent listed the last four digits of
# an ID among the details it needs read 0.17 as a detail and 1.0 as part of one, so it was not kept).
ANSWERING_COUNTS = ("A value given on its own, with nothing said about it (only digits, a date, an email address or a "
                    "name), counts as a detail when it fits one of the details the agent just asked for "
                    "(agent_previous_reply, agent_just_asked).")
ASKED_FORM_IS_WHOLE = ("A detail in the form the agent asked for it (for example the last four digits of an ID, when "
                       "those are what was asked for) is whole, not part of one.")

# Each kind of identifying detail as a refusal question names it (identity.types.KINDS).
REFUSABLE = {"name": "full name", "dob": "date of birth", "phone": "phone number", "email": "email address",
             "id": "ID number (or its last four digits)"}

# A representative always asks about someone else's policy; only a move to a different person's policy is a switch.
NOT_A_SWITCH = ("Answer yes only when the caller moves to a policy of a different person from the one this conversation "
                "has been about so far. A caller acting for a policyholder they have named (for example a parent) and "
                "asking about that same person's policy is not asking about someone else.")

# Asked only of a caller acting for a policyholder: that policyholder's policy is never another person's.
REPRESENTED_IS_NOT_OTHER = ("Asking about the policyholder the caller is acting for (by name, as \"her\" or \"him\", "
                            "or by the relationship given before) is not a different person.")

# What "unsupported" covers: requests about insurance that this claims line does not handle. What "off_topic" and
# "conversation" cover, so that a general-knowledge question asked in the middle of a claim is still off-topic.
UNSUPPORTED_MEANS = ("unsupported_insurance means an insurance request this claims line cannot handle, such as opening "
                     "or filing a new claim, coverage or benefits questions, billing or payments, or changing contact or "
                     "policy details. off_topic means a message with nothing to do with the caller's insurance, such as a "
                     "general-knowledge question, a question about what a word or abbreviation means when it is not an "
                     "insurance term, or a request for a joke; it stays off_topic during a conversation about a claim. "
                     "conversation means greetings, thanks, goodbyes and questions about this conversation itself.")

# The only account question this line answers is what the account holds (its policy number). "I want to change the
# phone number on my account" names the account but asks for a change the line cannot make: read as an account
# question it was answered with the policy number, and the offer of a person went missing whenever the reading of
# what the message is about fell below its line.
ACCOUNT_QUESTION_MEANS = ("a question asking for something the caller's account holds, such as its policy number; a "
                          "request to change account, contact or policy details is not one, and neither is a general "
                          "question about the policy or what is going on with it (that is about its claims)")
CLAIM_OR_ACCOUNT_MEANS = ("the caller's claims (anything about them: status, documents, deadlines, money figures and "
                          "what the words for them mean), or what their account holds, such as its policy number; a "
                          "request to change account, contact or policy details is unsupported_insurance")

# "Between March and June" names April and May too.
MONTH_IN_A_RANGE = "A month inside a range the caller gives counts (April is in \"between March and June\")."

# A claim in focus counts as listed: describing a different claim from the one being discussed is a new search.
BEING_DISCUSSED = ("The claims the agent is already discussing count as listed: a message that describes a different "
                   "claim from those is a new search.")

# A time counted from today, each described: "early this year" is this year, not a question of how recent.
RELATIVE_TIMES = (
    ("none", "no time counted from today"),
    ("most_recent", "the newest claim"),
    ("oldest", "the oldest claim"),
    ("this_year", "claims made this year, including early in it or so far this year"),
    ("last_year", "claims made last year"),
)

# Who the caller is, each described: "Actually, this is Margaret" after calling for Margaret is the policyholder
# speaking, which a bare list of the three words left unsure.
CALLER_ROLES = (
    ("policyholder", "the caller says they are the policyholder, or gives the policyholder's name as their own"),
    ("third_party", "the caller says they are calling for someone else, the policyholder"),
    ("not_stated", "the message does not say who the caller is"),
)

# The search relation's choices, each described, so "no list shown" is not picked while a claim is being discussed.
SEARCH_RELATIONS = (
    ("narrowing", "picks out some of the claims already listed or being discussed"),
    ("correcting", "corrects how the caller described a claim before"),
    ("new_search", "describes a different claim from those listed or being discussed"),
    ("no_list_shown", "no claim has been listed or discussed yet"),
)

# A question about one's own claim is not a request to read out what is held about the caller.
PERSONAL_DETAIL_MEANS = ("A personal detail is something held about the caller, such as their email address, phone "
                         "number, date of birth or ID number; a question about a claim is not one.")

# Each type is its own answer: naming "dental" says nothing about "healthcare", and "healthcare or dental" names both.
# A request about all claims, or claims in general, names none (it would otherwise hide the caller's other claims).
ONE_TYPE_AT_A_TIME = ("Answer for this type alone: yes when the caller names it, also as one of alternatives "
                      "(\"healthcare or dental\" names both); a type the caller does not name does not count, and "
                      "asking about all claims or claims in general names no type.")

# A figure's meaning comes from a data file, so it travels in the data record (figure_meanings, keyed by the figure's
# name in words), never in a question; the question only points at its entry.
# Each figure is its own answer: a caller describing what was paid is not asking about the most that could be paid.
# A figure named in a shortened form, or described as one side of a comparison, is still named.
ONE_FIGURE_AT_A_TIME = ("Answer for this figure alone: yes when the caller's words name it, also in a shortened "
                        "form, or describe what it means, also as one of two figures compared; a different figure the "
                        "caller describes does not count for it.")

# The document asked about is the one the caller cannot get (the only use of the answer): "I have the office note, but
# I can't get the pathology report" names both, and the one they have is not it.
CANNOT_GET_WHICH = ("A document the caller says they have, or can get, is not one they cannot get; with no document "
                    "named, answer none_stated.")

# A claim's type or status counts only when the caller says it: the agent's earlier reply naming a claim is not the
# caller describing one, and a message that names no claim at all describes none.
CALLERS_OWN_WORDS = ("Count only what the caller's own words in this message say, not the agent's earlier reply; "
                     "answer no when the caller names none.")


@dataclass(frozen=True)
class Question:
    name: str
    type: Literal["predicate", "choice"]
    instructions: str
    choices: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FormRequest:
    message: str
    recent_caller_messages: tuple[str, ...] = ()
    previous_reply: str | None = None
    pending: str | None = None
    state: State = State.VERIFY_ID
    verified: bool = False
    claim_types: tuple[str, ...] = ()
    claim_statuses: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    needs: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    answers: tuple[tuple[str, str], ...] = ()
    listed_claims: tuple[ListedClaim, ...] = ()
    documents: tuple[str, ...] = ()
    examples: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    acting_for_policyholder: bool = False  # the caller has said they act for the policyholder
    figure_meanings: Mapping[str, str] = field(default_factory=dict)  # money field -> glossary meaning: data, sent in the record

    def __post_init__(self) -> None:
        if not self.verified and (self.listed_claims or self.documents):
            raise ValueError("claim rows must not appear in a request before verification")


def _yn(name: str, text: str) -> Question:
    return Question(name, "predicate", text)


def _choice(req: FormRequest, name: str, text: str, options: list[tuple[str, str]]) -> Question:
    described = []
    seen: set[str] = set()
    for value, desc in options:
        if value in seen:  # the real API rejects a repeated choice value; keep the first
            continue
        seen.add(value)
        ex = req.examples.get(f"{name}:{value}", ())
        if ex:
            desc = f"{desc} (for example: {'; '.join(repr(e) for e in ex)})" if desc else \
                f"for example: {'; '.join(repr(e) for e in ex)}"
        described.append((value, desc or value.replace("_", " ")))
    return Question(name, "choice", text, tuple(described))


def _plain(values: list[str]) -> list[tuple[str, str]]:
    return [(v, v.replace("_", " ")) for v in values]


def _listed(position: int, c: ListedClaim) -> str:
    """A listed claim worded as the caller saw it: its place in the list, number, type, status and date."""
    return (f"number {position} in the list: {c.case_id}, {c.case_type or 'claim'}, status {c.status}, "
            f"created {c.created_at or 'date not on file'}")


def build_questions(req: FormRequest) -> list[Question]:
    c = lambda name, text, values: _choice(req, name, text, _plain(values))  # noqa: E731
    qs: list[Question] = [
        _yn("wants_person", "Does the caller ask to speak to a person?"),
        _choice(req, "about", "What is the caller's message about? " + UNSUPPORTED_MEANS,
                [("claim_or_account", CLAIM_OR_ACCOUNT_MEANS)] + _plain(["unsupported_insurance", "off_topic",
                                                                         "conversation"])),
        _yn("skips_step", "Does the caller want to skip the step they were just asked for?"),
        _yn("asks_why_step", "Does the caller ask why the agent needs what it just asked for?"),
        _yn("declines", "Does the caller refuse to give what was asked?"),
        c("mood", "How does the caller sound?", ["calm", "uneasy", "upset", "angry"]),
        c("pending_reply", "How does the caller's message answer the question the agent just asked?",
          ["yes", "no", "unclear", "not_answering", "nothing_pending"]),
        _yn("done", "Is the caller finished and ready to end the conversation?"),
        _yn("english", "Is the caller's message written in English?"),
        _yn("other_person_policy", "Is the caller asking about a policy that belongs to someone else? "
            + NOT_A_SWITCH),
        _yn("wants_email_summary", "Does the caller ask for a summary by email?"),
    ]
    qs += [_yn(f"shaped_need:{i}", f"Earlier caller message number {i} is in the data record. Did that message help "
                                   "establish what the caller needs now?")
           for i in range(len(req.recent_caller_messages[-RECENT_WINDOW:]))]
    if not req.verified:
        qs += [
            _choice(req, "caller_role", "Who is the caller in relation to the policy?", list(CALLER_ROLES)),
            _yn("gave_identity_detail", "Does the message state a detail that identifies the policyholder? "
                + OWN_DETAILS_COUNT + " " + ANSWERING_COUNTS),
            _yn("correcting_detail", "Is the caller correcting a detail they gave earlier?"),
            _yn("partial_detail", "Does the caller give only part of a detail, such as some digits? " + ASKED_FORM_IS_WHOLE),
            c("relationship", "How does the caller say they are related to the policyholder?",
              list(req.relationships) + ["other", "not_stated"]),
        ]
        qs += [_yn(f"refuses:{k}", f"Does the caller say they will not, or would rather not, give their {REFUSABLE[k]}?")
               for k in KINDS]
    if req.acting_for_policyholder:
        qs.append(_yn("other_than_represented", "The caller is acting for a policyholder. Does the caller now ask about "
                      "the policy of a different person from that policyholder (for example another relative)? "
                      + REPRESENTED_IS_NOT_OTHER))
    if req.state == State.AWAIT_CONSENT:
        qs.append(_yn("consent_recheck", "Is the caller asking to check the consent answer again, or saying the policyholder has approved it?"))
    qs += [
        _yn("mentions_claim", "Does the caller mention a particular claim or kind of claim?"),
        _yn("quoted_claim_number", "Does the caller quote a claim number?"),
        c("how_many", "How many claims does the caller mean?",
          ["one", "several", "all", "none", "same_as_before"]),
    ]
    qs += [_yn(f"claim_type:{v}", f"Does the caller mean a claim of type {v}? {ONE_TYPE_AT_A_TIME} {CALLERS_OWN_WORDS}")
           for v in dict.fromkeys(req.claim_types)]
    qs += [_yn(f"claim_status:{v}", f"Does the caller mean a claim whose status is {v}? {CALLERS_OWN_WORDS}")
           for v in dict.fromkeys(req.claim_statuses)]
    qs += [_yn(f"month:{m}", f"Does the caller mention {MONTH_NAMES[m - 1]} as the time of a claim? {NOT_A_CLAIM_DATE} "
                             f"{MONTH_IN_A_RANGE} {CALLERS_OWN_WORDS}")
           for m in range(1, 13)]
    qs += [_yn(f"year:{y}", f"Does the caller mention the year {y} as the time of a claim? {NOT_A_CLAIM_DATE} "
                            f"{CALLERS_OWN_WORDS}")
           for y in dict.fromkeys(req.years)]
    qs += [
        _yn("no_year_said", "Does the caller mention a month or date without saying a year?"),
        _choice(req, "relative_time", "Does the caller describe a claim by when it was made, counted from today?",
                list(RELATIVE_TIMES)),
        _choice(req, "search_relation", "How does this message relate to the claims already listed? " + BEING_DISCUSSED,
                list(SEARCH_RELATIONS)),
    ]
    if req.listed_claims:
        qs.append(_choice(req, "listed_claim", "Which listed claim does the caller point at?",
                          [(c.case_id, _listed(i, c)) for i, c in enumerate(req.listed_claims, 1)]
                          + [("none", "none of them")]))
    qs.append(_choice(req, "need", "What does the caller need?",
                      _plain(list(req.needs)) + [("account_question", ACCOUNT_QUESTION_MEANS), ("none", "none")]))
    qs.append(_choice(req, "answer_wanted", "Which supplied answer does the caller want?",
                      [(topic, desc) for topic, desc in req.answers]
                      + [("something_else", "something not listed"), ("none", "no particular answer")]))
    qs.append(_yn("multi_question", "Does the message ask more than one question?"))
    if req.documents:
        # "not_listed" lets the form say the caller named a document, but one that is not on this list; that is
        # never the same as naming none.
        qs.append(_choice(req, "which_document", "Which required document does the caller say they cannot get? "
                          + CANNOT_GET_WHICH,
                          _plain(list(req.documents)) + [
                              ("all", "all of them"),
                              ("not_listed", "a document that is not in this list"),
                              ("none_stated", "no document named")]))
    if req.verified:
        qs.append(_choice(req, "calculation", "Does the caller ask for a total, a difference, the highest or lowest, a "
                          "count or a number of days to be worked out?",
                          [(v, CALCULATION_MEANS[v]) for v in CALCULATIONS]))
        qs += [_yn(f"figure:{m}", f"The data record's money_figures lists each money figure with what it means on "
                   f"file. Is the caller asking about the figure named \"{m.replace('_', ' ')}\" there, by its name "
                   f"or by describing what it means? " + ONE_FIGURE_AT_A_TIME)
               for m in MONEY_FIELDS]
        qs.append(_yn("no_figure_said", "Does the caller ask for a total, difference, highest or lowest without "
                      "saying which money figure?"))
    qs += [
        _yn("cant_get_document", "Does the caller say they cannot get a required document?"),
        _yn("asks_personal_detail", "Does the caller ask the agent to tell them a personal detail? "
            + PERSONAL_DETAIL_MEANS),
    ]
    if req.state == State.POST_PROCESS:
        qs.append(c("email_offer_reply", "How does the caller answer the offer of an email summary?",
                    ["send", "dont_send", "whats_in_it", "which_address", "different_address",
                     "unclear", "something_else"]))
    return qs


def _yn_from(ans, asked: bool) -> YesNo:
    if isinstance(ans, YesNoAnswer):
        return YesNo(p=ans.probability)
    return YesNo(p=None)


def form_from_answers(req: FormRequest, answers: Mapping[str, YesNoAnswer | PickAnswer | Refused]) -> Form:
    questions = {q.name: q for q in build_questions(req)}

    def yn(name: str) -> YesNo:
        return _yn_from(answers.get(name), name in questions)

    def pick(name: str) -> Pick:
        ans = answers.get(name)
        q = questions[name]
        if not isinstance(ans, PickAnswer) or ans.choice not in {v for v, _ in q.choices}:
            return Pick(value=None, p=0.0, options={})
        offered = {v for v, _ in q.choices}
        return Pick(value=ans.choice, p=ans.confidence,
                    options={k: p for k, p in ans.probabilities.items() if k in offered})

    def opt_yn(name: str) -> YesNo | None:
        return yn(name) if name in questions else None

    def opt_pick(name: str) -> Pick | None:
        return pick(name) if name in questions else None

    def unasked_yn(name: str) -> YesNo:
        return yn(name) if name in questions else YesNo(p=0.0)

    def unasked_pick(name: str) -> Pick:
        return pick(name) if name in questions else Pick()

    return Form(
        wants_person=yn("wants_person"), about=pick("about"), skips_step=yn("skips_step"),
        asks_why_step=yn("asks_why_step"), declines=yn("declines"), mood=pick("mood"),
        pending_reply=pick("pending_reply"), done=yn("done"), english=yn("english"),
        other_person_policy=yn("other_person_policy"), wants_email_summary=yn("wants_email_summary"),
        shaped_need=[yn(f"shaped_need:{i}") for i in range(len(req.recent_caller_messages[-RECENT_WINDOW:]))],
        caller_role=opt_pick("caller_role"), gave_identity_detail=opt_yn("gave_identity_detail"),
        correcting_detail=opt_yn("correcting_detail"), partial_detail=opt_yn("partial_detail"),
        consent_recheck=opt_yn("consent_recheck"), relationship=opt_pick("relationship"),
        mentions_claim=yn("mentions_claim"), quoted_claim_number=yn("quoted_claim_number"),
        how_many=pick("how_many"),
        claim_types={v: yn(f"claim_type:{v}") for v in req.claim_types},
        claim_statuses={v: yn(f"claim_status:{v}") for v in req.claim_statuses},
        months={m: yn(f"month:{m}") for m in range(1, 13)},
        years={y: yn(f"year:{y}") for y in req.years},
        no_year_said=yn("no_year_said"), relative_time=pick("relative_time"),
        search_relation=pick("search_relation"), listed_claim=unasked_pick("listed_claim"),
        need=pick("need"), answer_wanted=pick("answer_wanted"), multi_question=yn("multi_question"),
        which_document=unasked_pick("which_document"), cant_get_document=yn("cant_get_document"),
        asks_personal_detail=yn("asks_personal_detail"), email_offer_reply=opt_pick("email_offer_reply"),
        calculation=unasked_pick("calculation"), figures={m: unasked_yn(f"figure:{m}") for m in MONEY_FIELDS},
        no_figure_said=unasked_yn("no_figure_said"),
        other_than_represented=opt_yn("other_than_represented"),
        refuses={k: yn(f"refuses:{k}") for k in KINDS if f"refuses:{k}" in questions},
    )


def figure_meanings(req: FormRequest) -> list[dict[str, str]]:
    """The money figures for the data record: each one's name in words, as the questions name it, and its glossary
    meaning (data from a file, so it travels here and never in a question)."""
    return [{"name": m.replace("_", " "), "meaning": text} for m, text in req.figure_meanings.items() if text]
