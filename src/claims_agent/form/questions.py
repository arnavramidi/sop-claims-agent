from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from claims_agent.form.fields import Form, ListedClaim, Pick, YesNo
from claims_agent.loader.read import MONEY_FIELDS
from claims_agent.form.read import PickAnswer, Refused, YesNoAnswer
from claims_agent.states import State

# What a caller can ask code to work out (Ruling 22). Kept here so the form needs nothing from the answers package.
CALCULATIONS = ("none", "total", "difference", "highest", "lowest", "count", "days_open", "days_left")

RECENT_WINDOW = 6  # how many earlier caller messages a request carries (and asks a question about)

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

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
        c("about", "What is the caller's message about?",
          ["claim_or_account", "unsupported_insurance", "off_topic", "conversation"]),
        _yn("skips_step", "Does the caller want to skip the step they were just asked for?"),
        _yn("asks_why_step", "Does the caller ask why the agent needs what it just asked for?"),
        _yn("declines", "Does the caller refuse to give what was asked?"),
        c("mood", "How does the caller sound?", ["calm", "uneasy", "upset", "angry"]),
        c("pending_reply", "How does the caller's message answer the question the agent just asked?",
          ["yes", "no", "unclear", "not_answering", "nothing_pending"]),
        _yn("done", "Is the caller finished and ready to end the conversation?"),
        _yn("english", "Is the caller's message written in English?"),
        _yn("other_person_policy", "Is the caller asking about a policy that belongs to someone else?"),
        _yn("wants_email_summary", "Does the caller ask for a summary by email?"),
    ]
    qs += [_yn(f"shaped_need:{i}", f"Earlier caller message number {i} is in the data record. Did that message help "
                                   "establish what the caller needs now?")
           for i in range(len(req.recent_caller_messages[-RECENT_WINDOW:]))]
    if not req.verified:
        qs += [
            c("caller_role", "Who is the caller in relation to the policy?",
              ["policyholder", "third_party", "not_stated"]),
            _yn("gave_identity_detail", "Does the message state a detail that identifies the policyholder?"),
            _yn("correcting_detail", "Is the caller correcting a detail they gave earlier?"),
            _yn("partial_detail", "Does the caller give only part of a detail, such as some digits?"),
            c("relationship", "How does the caller say they are related to the policyholder?",
              list(req.relationships) + ["other", "not_stated"]),
        ]
    if req.state == State.AWAIT_CONSENT:
        qs.append(_yn("consent_recheck", "Is the caller asking to check the consent answer again, or saying the policyholder has approved it?"))
    qs += [
        _yn("mentions_claim", "Does the caller mention a particular claim or kind of claim?"),
        _yn("quoted_claim_number", "Does the caller quote a claim number?"),
        c("how_many", "How many claims does the caller mean?",
          ["one", "several", "all", "none", "same_as_before"]),
    ]
    qs += [_yn(f"claim_type:{v}", f"Does the caller mean a claim of type {v}?") for v in dict.fromkeys(req.claim_types)]
    qs += [_yn(f"claim_status:{v}", f"Does the caller mean a claim whose status is {v}?")
           for v in dict.fromkeys(req.claim_statuses)]
    qs += [_yn(f"month:{m}", f"Does the caller mention {MONTH_NAMES[m - 1]}?") for m in range(1, 13)]
    qs += [_yn(f"year:{y}", f"Does the caller mention the year {y}?") for y in dict.fromkeys(req.years)]
    qs += [
        _yn("no_year_said", "Does the caller mention a month or date without saying a year?"),
        c("relative_time", "Does the caller describe a claim by how recent it is?",
          ["none", "most_recent", "oldest", "this_year", "last_year"]),
        c("search_relation", "How does this message relate to the claims already listed?",
          ["narrowing", "correcting", "new_search", "no_list_shown"]),
    ]
    if req.listed_claims:
        qs.append(_choice(req, "listed_claim", "Which listed claim does the caller point at?",
                          [(c.case_id, _listed(i, c)) for i, c in enumerate(req.listed_claims, 1)]
                          + [("none", "none of them")]))
    qs.append(c("need", "What does the caller need?", list(req.needs) + ["account_question", "none"]))
    qs.append(_choice(req, "answer_wanted", "Which supplied answer does the caller want?",
                      [(topic, desc) for topic, desc in req.answers]
                      + [("something_else", "something not listed"), ("none", "no particular answer")]))
    qs.append(_yn("multi_question", "Does the message ask more than one question?"))
    if req.documents:
        # "not_listed" lets the form say the caller named a document, but one that is not on this list; that is
        # never the same as naming none.
        qs.append(_choice(req, "which_document", "Which required document does the caller mean?",
                          _plain(list(req.documents)) + [
                              ("all", "all of them"),
                              ("not_listed", "a document that is not in this list"),
                              ("none_stated", "no document named")]))
    if req.verified:
        qs.append(c("calculation", "Does the caller ask for a total, a difference, the highest or lowest, a count or a "
                    "number of days to be worked out?", list(CALCULATIONS)))
        qs += [_yn(f"figure:{m}", f"Is the caller asking about the {m.replace('_', ' ')} figure?") for m in MONEY_FIELDS]
        qs.append(_yn("no_figure_said", "Does the caller ask for a total, difference, highest or lowest without "
                      "saying which money figure?"))
    qs += [
        _yn("cant_get_document", "Does the caller say they cannot get a required document?"),
        _yn("asks_personal_detail", "Does the caller ask the agent to tell them a personal detail?"),
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
    )
