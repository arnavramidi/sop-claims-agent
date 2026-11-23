"""A turn's record laid out as a trace: the steps, their inputs and outputs, masking, scores, and the SDK calls."""

import json
import sys
import types
from uuid import UUID

import pytest

from claims_agent.observability import BraintrustLog, make_turn_log
from claims_agent.observability.turn_spans import DRAFT_NOTE, turn_spans
from claims_agent.settings import load_settings


def _q(name, mark, answered, answer=None, settled=None, options=None):
    q = {"id": name, "topic": "x", "type": "pick" if options else "yes_no", "text": f"Question {name}?",
         "answer": answer, "reading": None, "mark": mark, "answered": answered}
    if options:
        q["options"] = options
    if settled:
        q["settled"] = settled
    return q


def record(**over) -> dict:
    base = {
        "state_before": "VERIFY_ID", "state_after": "RESOLVE_INTENT", "path": ["VERIFY_ID", "RESOLVE_INTENT"],
        "filler": "decisions", "flags": ["x"], "models": {"fast": "m-fast", "text": "m-text", "larger": "m-large"},
        "stated": {"id_digits": "123454472", "email": "a@b.com"},
        "fired": [{"rule": "r1"}, {"rule": "r2"}],
        "form_asked": {"thresholds": {"yes": 0.5}, "readings": [{
            "part": None, "message": None, "state": "VERIFY_ID", "verified": False, "count": 3, "what_it_saw": ["s"],
            "questions": [
                _q("done", "confident", False, {"p": 0.01, "by": "m"}),
                _q("about", "confident", True, {"value": "claims", "p": 0.9, "options": {}, "by": "m"},
                   options=[{"value": "claims", "text": "Claims"}]),
                _q("mood", "below", False, {"value": "calm", "p": 0.4, "options": {}, "by": "m"},
                   settled={"value": "calm", "steps": ["code"], "larger": "calm"}, options=[{"value": "calm", "text": "Calm"}]),
                _q("skips", "no_answer", False)]}]},
        "parts": [{"instruction": "answer", "facts": [{"kind": "k", "text": "Claim CL-1 is denied", "claim": "CL-1"}],
                   "plain": None, "pending": None}],
        "reply_source": "rewrite",
        "checks": [{"passed": False, "failures": ["status word"], "leak": [], "invented": [], "other": []},
                   {"passed": True, "failures": [], "leak": [], "invented": [], "other": []}],
        "usage": {"calls": [
            {"job": "extract", "model": "m-text", "input_tokens": 5, "output_tokens": 1, "total_tokens": 6,
             "started": 10.0, "ended": 11.0},
            {"job": "form", "model": "m-fast", "input_tokens": 50, "output_tokens": 9, "total_tokens": 59,
             "started": 11.0, "ended": 12.0},
            {"job": "settle", "model": "m-large", "input_tokens": 7, "output_tokens": 2, "total_tokens": 9,
             "started": 12.5, "ended": 13.0},
            {"job": "word", "model": "m-text", "input_tokens": 8, "output_tokens": 3, "total_tokens": 11,
             "started": 14.0, "ended": 15.0},
            {"job": "rewrite", "model": "m-large", "input_tokens": 9, "output_tokens": 4, "total_tokens": 13,
             "started": 15.5, "ended": 16.0}],
            "totals": {"total_tokens": 98}},
        "db_actions": {"actions": [{"action": "lookup_claims", "outcome": {"rows": 2}, "started": 12.2, "ended": 12.3}]},
    }
    base.update(over)
    return base


def spans(rec=None, **kw):
    return turn_spans(message="my id is 123-45-4472", reply="Claim CL-1 is denied.", record=rec or record(), **kw)


def test_the_turn_comes_first_and_its_steps_follow_in_the_order_they_happened():
    out = spans(conversation_id="c1", turn_no=2, tags=["needs_review"])
    assert out[0].name == "turn" and out[0].type == "task" and out[0].tags == ["needs_review"]
    assert [s.name for s in out[1:]] == ["extract", "form", "lookup_claims", "settle", "rules", "word", "reply check",
                                         "rewrite"]
    assert [s.type for s in out[1:]] == ["llm", "llm", "function", "llm", "function", "llm", "function", "llm"]
    starts = [s.start for s in out[1:] if s.start is not None]
    assert starts == sorted(starts)


def test_the_turn_has_a_small_summary_and_keeps_the_whole_record():
    root = spans(conversation_id="c1", turn_no=2)[0]
    md = root.metadata
    assert (md["conversation_id"], md["turn_no"], md["state_before"], md["state_after"]) == ("c1", 2, "VERIFY_ID",
                                                                                          "RESOLVE_INTENT")
    assert md["path"] == ["VERIFY_ID", "RESOLVE_INTENT"] and md["form_filler"] == "decisions"
    assert md["reply_source"] == "rewrite" and md["flags"] == ["x"] and md["models"]["text"] == "m-text"
    assert md["record"]["fired"] == [{"rule": "r1"}, {"rule": "r2"}] and md["record"]["usage"]["totals"]
    assert root.metrics == {}  # token counts only on the model calls


def test_the_form_as_asked_is_sent_once_not_also_inside_the_kept_record():
    out = spans()
    form = next(s for s in out if s.name == "form")
    assert [q["id"] for q in form.input["questions"]] == ["done", "about", "mood", "skips"]
    assert form.input["questions"][1]["options"] == [{"value": "claims", "text": "Claims"}]
    assert form.output["answers"][1] == {"id": "about", "value": "claims", "probability": 0.9, "mark": "confident"}
    kept = out[0].metadata["record"]["form_asked"]["readings"][0]
    assert isinstance(kept["questions"], str) and kept["what_it_saw"] == ["s"]
    assert "Question about?" not in json.dumps(out[0].metadata)


def test_a_form_with_no_form_step_stays_whole_in_the_kept_record():
    rec = record()
    rec["usage"] = None
    out = spans(rec)
    assert isinstance(out[0].metadata["record"]["form_asked"]["readings"][0]["questions"], list)
    assert "form" not in [s.name for s in out]


def test_each_model_call_carries_what_the_record_says_it_was_given_and_returned():
    by = {s.name: s for s in spans()}
    assert by["extract"].output == {"id_digits": "*****4472", "email": "[email]"}
    assert by["extract"].input == "my id is ***-**-4472"
    assert by["settle"].input == {"question": {"id": "mood", "text": "Question mood?", "options": [{"value": "calm", "text": "Calm"}]},
                                  "form_answer": {"id": "mood", "value": "calm", "probability": 0.4, "mark": "below"}}
    assert by["settle"].output == {"value": "calm", "steps": ["code"], "larger": "calm"}
    assert by["word"].input == {"parts": [{"instruction": "answer", "allowed_facts": ["Claim CL-1 is denied"]}]}
    assert by["rewrite"].input["rejected_by_check"] == {"failures": ["status word"]}
    assert by["rewrite"].output == "Claim CL-1 is denied."
    assert by["extract"].metrics == {"prompt_tokens": 5, "completion_tokens": 1, "tokens": 6}


def test_a_first_draft_that_the_record_does_not_hold_is_not_invented():
    word = next(s for s in spans() if s.name == "word")
    assert word.output == {"rejected_by_check": {"failures": ["status word"]}, "note": DRAFT_NOTE}
    live = record(reply_source="live", checks=[record()["checks"][1]])
    live["usage"]["calls"] = [c for c in live["usage"]["calls"] if c["job"] != "rewrite"]
    assert next(s for s in spans(live) if s.name == "word").output == "Claim CL-1 is denied."


def test_a_split_step_shows_the_parts_it_made():
    rec = record()
    rec["usage"]["calls"][0]["job"] = "split"
    rec["form_asked"]["readings"].append({"part": 1, "message": "what about CL-1?", "questions": []})
    split = next(s for s in spans(rec) if s.name == "split")
    assert split.output["questions"] == ["what about CL-1?"]


def test_database_actions_give_the_action_and_a_coarse_outcome_only():
    act = next(s for s in spans() if s.name == "lookup_claims")
    assert act.type == "function" and act.output == {"rows": 2} and act.input is None
    assert (act.start, act.end) == (12.2, 12.3)


def test_the_rules_and_the_reply_check_are_steps_of_their_own():
    by = {s.name: s for s in spans()}
    assert by["rules"].input == {"state_before": "VERIFY_ID", "form": {"answered": ["about"], "unsure": ["mood"]}}
    assert by["rules"].output == {"state_after": "RESOLVE_INTENT", "path": ["VERIFY_ID", "RESOLVE_INTENT"],
                                  "fired": [{"rule": "r1"}, {"rule": "r2"}]}
    assert (by["rules"].start, by["rules"].end) == (13.0, 14.0)
    assert by["reply check"].output["attempts"] == [{"attempt": 1, "passed": False, "failures": ["status word"]},
                                                    {"attempt": 2, "passed": True}]
    assert (by["reply check"].start, by["reply check"].end) == (15.0, 15.5)


def test_scores_say_whether_the_first_wording_passed_and_how_sure_the_form_was():
    assert spans()[0].scores == {"reply_check_first_pass": 0, "form_confident": round(2 / 3, 4)}
    rec = record(checks=[record()["checks"][1]])
    assert spans(rec)[0].scores["reply_check_first_pass"] == 1


@pytest.mark.parametrize("rec", [{}, {"usage": None, "db_actions": None, "form_asked": None, "checks": None},
                                 {"usage": {"calls": ["junk", 3]}, "checks": ["junk"], "parts": "junk",
                                  "form_asked": {"readings": ["junk"]}, "db_actions": {"actions": [None]}}])
def test_a_missing_or_odd_record_gives_the_turn_alone_or_what_can_be_built(rec):
    out = turn_spans(message="hi", reply="hello", record=rec)
    assert out[0].name == "turn" and out[0].input == "hi" and out[0].scores == {}
    assert all(s.name in ("turn", "reply check") or s.type == "llm" for s in out)


def test_a_turn_that_made_no_model_call_has_no_model_steps():
    rec = record(usage={"calls": [], "totals": {}}, reply_source="fixed", checks=[])
    names = [s.name for s in spans(rec)]
    assert "extract" not in names and "reply check" not in names and "rules" in names


def test_nothing_unmasked_is_in_any_span():
    text = json.dumps([s.__dict__ for s in spans()])
    for secret in ("123454472", "123-45-4472", "a@b.com"):
        assert secret not in text


def test_a_broken_piece_costs_only_its_own_step(monkeypatch):
    from claims_agent.observability import turn_spans as mod
    monkeypatch.setattr(mod, "_rules_step", lambda *a: 1 / 0)
    names = [s.name for s in spans()]
    assert "rules" not in names and "form" in names and "reply check" in names


# ---- the calls made to the library ----------------------------------------------------------------------------------

class _Span:
    def __init__(self, calls, kw, fail=None): self.calls, self.kw, self.fail = calls, kw, fail
    def log(self, **kw):
        if self.fail and self.kw.get("name") in self.fail:
            raise RuntimeError("rejected")
        self.calls.append(("log", self.kw.get("name"), kw))
    def end(self, **kw): self.calls.append(("end", self.kw.get("name"), kw))
    def start_span(self, **kw):
        self.calls.append(("child", kw.get("name"), kw))
        return _Span(self.calls, kw, self.fail)


class _Logger:
    def __init__(self, calls, fail=None): self.calls, self.fail = calls, fail
    def start_span(self, **kw):
        self.calls.append(("root", kw.get("name"), kw))
        return _Span(self.calls, kw, self.fail)
    def flush(self): pass


def _log(monkeypatch, calls, fail=None):
    fake = types.ModuleType("braintrust")
    fake.set_masking_function = lambda f: None
    fake.init_logger = lambda **kw: _Logger(calls, fail)
    monkeypatch.setitem(sys.modules, "braintrust", fake)
    return make_turn_log(load_settings(), "k")


def test_the_steps_are_sent_as_children_of_the_turn_with_their_types_and_times(monkeypatch):
    calls: list = []
    log = _log(monkeypatch, calls)
    assert isinstance(log, BraintrustLog)
    span_id = log.start(UUID(int=1), 4)
    log.finish(span_id, message="my id is 123-45-4472", reply="Claim CL-1 is denied.", record=record(), tags=["needs_review"])
    assert calls[0][0] == "root" and calls[0][2]["type"] == "task" and calls[0][2]["id"] == span_id
    kids = [c for c in calls if c[0] == "child"]
    assert [(c[1], c[2]["type"]) for c in kids] == [("extract", "llm"), ("form", "llm"), ("lookup_claims", "function"),
                                                    ("settle", "llm"), ("rules", "function"), ("word", "llm"),
                                                    ("reply check", "function"), ("rewrite", "llm")]
    assert next(c for c in kids if c[1] == "form")[2]["start_time"] == 11.0
    root_log = next(c for c in calls if c[0] == "log" and c[1] == "turn")[2]
    assert root_log["scores"]["reply_check_first_pass"] == 0 and root_log["tags"] == ["needs_review"]
    assert "metrics" not in root_log and root_log["metadata"]["turn_no"] == 4
    form_log = next(c for c in calls if c[0] == "log" and c[1] == "form")[2]
    assert form_log["metrics"]["tokens"] == 59 and form_log["input"]["questions"]
    assert ("end", "form", {"end_time": 12.0}) in calls and calls[-1][:2] == ("end", "turn")


def test_a_step_the_library_refuses_does_not_stop_the_others_or_the_turn(monkeypatch):
    calls: list = []
    log = _log(monkeypatch, calls, fail={"form"})
    log.finish(log.start(UUID(int=1), 1), message="hi", reply="hello", record=record())
    assert ("child", "rewrite") in [c[:2] for c in calls] and calls[-1][:2] == ("end", "turn")


def test_an_error_while_building_spans_never_reaches_the_turn(monkeypatch):
    from claims_agent.observability import safe_finish
    from claims_agent.observability import turn_log as mod
    calls: list = []
    log = _log(monkeypatch, calls)
    import claims_agent.observability.turn_spans as spans_mod
    monkeypatch.setattr(spans_mod, "turn_spans", lambda **kw: 1 / 0)
    safe_finish(log, log.start(UUID(int=1), 1), message="hi", reply="hello", record=record())  # does not raise
    monkeypatch.setattr(spans_mod, "turn_spans", lambda **kw: (_ for _ in ()).throw(ValueError("x")))
    safe_finish(log, "unknown", message="hi", reply="hello", record={})
    assert mod is not None


def test_the_conversation_id_is_kept_whole_although_it_holds_a_long_run_of_digits():
    cid = "f1340947-b41f-4123-9048-6b06c0ef220a"
    root = spans(conversation_id=cid, turn_no=1)[0]
    assert root.metadata["conversation_id"] == cid
