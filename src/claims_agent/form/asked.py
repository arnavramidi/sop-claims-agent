"""The form as the fast model was asked it, for the inspector: every question in the words it was sent, with the
answer it got and how sure that answer was by the thresholds.

The questions come from `build_questions`, the same function the fillers send, so a question added to the form is
listed here without any other change. Its answer is read from the filled form by the question's name: a plain name
is the form field of that name, and a name "family:key" is entry `key` of the form field the family maps to below
(a test checks that every question the form can ask finds its answer). Each question is put under a topic by its
name; a name no topic lists goes under "Other".
"""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field

from claims_agent.form.fields import Form, Pick, YesNo
from claims_agent.form.questions import FormRequest, Question, build_questions
from claims_agent.settings import Thresholds

# Question family (the name before ":") -> the form field that holds its answers.
FAMILIES = {"claim_type": "claim_types", "claim_status": "claim_statuses", "month": "months", "year": "years",
            "figure": "figures", "refuses": "refuses", "shaped_need": "shaped_need"}

# Topic -> the question names (or families) it groups, in the order the inspector shows the topics.
TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The conversation", ("wants_person", "about", "mood", "english", "pending_reply", "skips_step", "asks_why_step",
                          "declines", "done", "multi_question", "wants_email_summary", "email_offer_reply")),
    ("Who is calling", ("caller_role", "gave_identity_detail", "correcting_detail", "partial_detail", "relationship",
                        "refuses", "other_person_policy", "other_than_represented", "consent_recheck")),
    ("Which claim", ("mentions_claim", "quoted_claim_number", "how_many", "claim_type", "claim_status", "month",
                     "year", "no_year_said", "relative_time", "search_relation", "listed_claim")),
    ("What they need", ("need", "answer_wanted", "which_document", "cant_get_document", "asks_personal_detail")),
    ("Figures and sums", ("calculation", "figure", "no_figure_said")),
    ("Earlier messages", ("shaped_need",)),
)
OTHER = "Other"
_TOPIC_OF = {name: topic for topic, names in TOPICS for name in names}

# Choices that mean the caller said nothing of that kind: a confident pick of one is not counted as answered.
SAID_NOTHING = frozenset({"none", "none_stated", "not_stated", "nothing_pending", "no_list_shown", "not_answering"})


@dataclass
class Reading:
    """One filled form of this turn: the whole message's, or one part's when a message was split into questions.
    `settled` gets, by question name, how the turn settled an answer that was not sure enough."""
    request: FormRequest
    form: Form
    part: int | None = None  # the part's number; None for the whole message
    settled: MutableMapping[str, dict] = field(default_factory=dict)


def topic(name: str) -> str:
    return _TOPIC_OF.get(name.partition(":")[0], OTHER)


def answer_of(form: Form, name: str) -> YesNo | Pick | None:
    """The form's answer to the question called `name`, or None when the form holds none for it."""
    base, _, key = name.partition(":")
    if not key:
        found = getattr(form, base, None)
        return found if isinstance(found, (YesNo, Pick)) else None
    holder = getattr(form, FAMILIES.get(base, base), None)
    if isinstance(holder, list):
        return holder[int(key)] if key.isdigit() and int(key) < len(holder) else None
    if isinstance(holder, Mapping):
        found = holder.get(key)
        if found is None and key.lstrip("-").isdigit():
            found = holder.get(int(key))
        return found if isinstance(found, (YesNo, Pick)) else None
    return None


def _yes_no(a: YesNo | None, t: Thresholds) -> dict:
    if a is None or a.p is None:
        return {"answer": None if a is None else {"p": None, "by": a.by}, "reading": None, "mark": "no_answer",
                "answered": False}
    reading = "yes" if a.p >= t.yes else "no"
    mark = "unsure" if t.unsure_low < a.p < t.unsure_high else "confident"
    return {"answer": {"p": a.p, "by": a.by}, "reading": reading, "mark": mark, "answered": reading == "yes"}


def _pick(a: Pick | None, t: Thresholds) -> dict:
    if a is None or a.value is None:
        return {"answer": None if a is None else {"value": None, "p": a.p, "options": dict(a.options), "by": a.by},
                "reading": None, "mark": "no_answer", "answered": False}
    mark = "confident" if a.p >= t.pick_confident else "below"
    return {"answer": {"value": a.value, "p": a.p, "options": dict(a.options), "by": a.by}, "reading": a.value,
            "mark": mark, "answered": mark == "confident" and a.value not in SAID_NOTHING}


def question_entry(q: Question, form: Form, t: Thresholds, settled: Mapping[str, dict]) -> dict:
    a = answer_of(form, q.name)
    out: dict = {"id": q.name, "topic": topic(q.name), "type": "yes_no" if q.type == "predicate" else "pick",
                 "text": q.instructions}
    if q.type == "choice":
        out["options"] = [{"value": v, "text": d} for v, d in q.choices]
        out.update(_pick(a if isinstance(a, Pick) else None, t))
    else:
        out.update(_yes_no(a if isinstance(a, YesNo) else None, t))
    if q.name in settled:
        out["settled"] = dict(settled[q.name])
    return out


def what_it_saw(req: FormRequest) -> list[str]:
    """What shaped this request's question set and data record, in plain words, read from the request itself."""
    state = req.state.value if hasattr(req.state, "value") else str(req.state)
    out = [f"State {state}."]
    if req.verified:
        out.append("The caller is verified: claim types, statuses and years are the caller's own claims'.")
    else:
        out.append("The caller is not verified: no claim of theirs is in the form; claim types, statuses and years "
                   "are the lists for all the data, and identity questions are asked.")
    if req.listed_claims:
        out.append(f"{len(req.listed_claims)} claims just listed to the caller are offered to pick from.")
    if req.documents:
        out.append(f"{len(req.documents)} outstanding documents of the claims in focus are offered to pick from.")
    if req.acting_for_policyholder:
        out.append("The caller acts for the policyholder, so the form asks whether a different person is meant.")
    if req.figure_meanings:
        out.append("The data record carries what each money figure means.")
    n = len(req.recent_caller_messages)
    out.append(f"{n} earlier caller message{'' if n == 1 else 's'} {'is' if n == 1 else 'are'} in the data record"
               + ("; the agent's previous reply is too." if req.previous_reply else "; no previous reply is."))
    if req.pending:
        out.append(f"The agent had just asked: {req.pending}.")
    return out


def reading_entry(r: Reading, t: Thresholds) -> dict:
    qs = build_questions(r.request)
    return {"part": r.part, "message": r.request.message if r.part is not None else None,
            "state": r.request.state.value if hasattr(r.request.state, "value") else str(r.request.state),
            "verified": r.request.verified, "count": len(qs), "what_it_saw": what_it_saw(r.request),
            "questions": [question_entry(q, r.form, t, r.settled) for q in qs]}


def form_asked(readings: list[Reading], t: Thresholds, *, cap: int) -> dict | None:
    """The record's `form_asked`: the thresholds the marks were read with, and each reading of this turn (the whole
    message's form, then one per part of a split message), masked like a trace and each cut to `cap` characters of
    JSON (texts are shortened first; a reading that was cut says `truncated`). None when no form was filled."""
    from claims_agent.observability.turn_log import mask_for_trace  # late: that module imports the inspector's
    from claims_agent.records import fit
    if not readings:
        return None
    return {"thresholds": {"yes": t.yes, "unsure_low": t.unsure_low, "unsure_high": t.unsure_high,
                           "pick_confident": t.pick_confident},
            "readings": [fit(mask_for_trace(reading_entry(r, t)), cap) for r in readings]}
