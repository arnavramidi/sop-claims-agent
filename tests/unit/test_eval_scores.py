"""The Braintrust eval's plain scores, its rows, and what it does without keys (no network, no models)."""

import importlib.util
import json
import pathlib

import pytest

from claims_agent.settings import ROOT

_spec = importlib.util.spec_from_file_location("run_braintrust", ROOT / "evals" / "run_braintrust.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def out(failures=(), transcript=()):
    return {"passed": not failures, "failures": list(failures), "transcript": list(transcript)}


def test_expectations_met_counts_plain_expectations_only():
    assert rb.expectations_met({}, out())["score"] == 1.0
    assert rb.expectations_met({}, out(["turn 1: state is VERIFY_ID, expected PROCESS_CASE"]))["score"] == 0.0
    leak_and_judge = ["turn 1: reply said claim number CL-2048 before verification",
                      "turn 2: no judge available for: The agent's last reply is calm."]
    assert rb.expectations_met({}, out(leak_and_judge))["score"] == 1.0


def test_no_leak_scores_any_claim_detail_before_verification():
    assert rb.no_leak({}, out(["turn 1: state is VERIFY_ID, expected PROCESS_CASE"]))["score"] == 1.0
    assert rb.no_leak({}, out(["turn 1: model request 2 carried denial reason of CL-2048 before verification"]))["score"] == 0.0


def test_grounded_is_the_share_of_replies_whose_numbers_and_dates_are_in_the_facts():
    turns = [
        {"turn": 1, "caller": "hi", "reply": "Your appeal deadline is 2026-04-30.",
         "facts": [{"kind": "deadline", "text": "appeal deadline 2026-04-30", "claim": "CL-2048"}]},
        {"turn": 2, "caller": "and the amount?", "reply": "You will get 950 dollars by March 3.",
         "facts": [{"kind": "amount", "text": "expected 780.00", "claim": "CL-2048"}]},
    ]
    got = rb.grounded({}, out(transcript=turns))
    assert got["score"] == 0.5 and [u["turn"] for u in got["metadata"]["ungrounded"]] == [2]


def test_grounded_reads_numbers_the_way_the_reply_check_does():
    # a claim number's digits are not a stray number, and a claim number must itself be in the facts
    in_facts = [{"turn": 1, "caller": "about CL-2048", "reply": "Your claim CL-2048 is being looked at.",
                 "facts": [{"kind": "status", "text": "status denied", "claim": "CL-2048"}]}]
    not_in_facts = [{"turn": 1, "caller": "about CL-2048", "reply": "Your claim CL-2049 is being looked at.",
                     "facts": [{"kind": "status", "text": "status denied", "claim": "CL-2048"}]}]
    assert rb.grounded({}, out(transcript=in_facts))["score"] == 1.0
    assert rb.grounded({}, out(transcript=not_in_facts))["score"] == 0.0


def test_the_rows_are_the_live_scenarios_by_their_own_names_without_the_waiting_ones():
    rows = rb.scenario_rows()
    ids = {r["id"] for r in rows}
    files = list(pathlib.Path("tests/live/scenarios").glob("*.yaml"))
    waiting = {f.stem for f in files if "waiting_for:" in f.read_text()}
    assert "verified_in_one_message" in ids and ids == {f.stem for f in files} - waiting
    assert all(r["input"]["scenario"] == r["id"] and r["input"]["caller_turns"] for r in rows)


def test_labelled_rows_have_stable_ids_and_pass_the_upload_masking():
    first, again = rb.labelled_rows(), rb.labelled_rows()
    assert [r["id"] for r in first] == [r["id"] for r in again] and len({r["id"] for r in first}) == len(first)
    from claims_agent.examples import for_upload
    assert all(for_upload(r) is not None for r in first)


def test_an_uploaded_row_with_an_id_number_or_email_is_masked(monkeypatch):
    class Dataset:
        def __init__(self):
            self.rows = []

        def insert(self, **row):
            self.rows.append(row)

        def flush(self):
            pass

    ds = Dataset()
    fake = type("bt", (), {"init_dataset": staticmethod(lambda project, name: ds)})
    rows = [{"id": "a", "input": {"message": "SSN 123-45-6789, mail me at jo@example.com"}},
            {"id": "b", "input": {"message": "write to jo@localhost"}}]
    assert rb.upload(fake, "p", "d", rows) == (1, 1)
    text = json.dumps(ds.rows)
    assert "123-45-6789" not in text and "jo@example.com" not in text and "localhost" not in text


async def test_without_keys_the_eval_says_so_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert await rb.main() == 2
    assert "BRAINTRUST_API_KEY not set" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["expectations_met", "no_leak", "grounded"])
def test_every_plain_score_is_named_for_itself(name):
    assert getattr(rb, name)({}, out())["name"] == name
