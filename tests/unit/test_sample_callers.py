"""The sample callers: the committed file and page are what the script makes from the data, the list is served from
that file only while it matches the folder's data files, and the page shows it as sample data."""

import json
import shutil
from datetime import date

import pytest
from fastapi.testclient import TestClient

from claims_agent import api, hosted, sample_callers
from claims_agent.engine import Deps
from claims_agent.settings import ROOT, load_settings, load_thresholds

STARTER = ROOT / "data" / "starter"
TODAY = load_settings().today.date


def deps_for(folder):
    s = load_settings()
    s = s.model_copy(update={"data": s.data.model_copy(update={"folder": str(folder)})})
    return Deps(settings=s, thresholds=load_thresholds(), db=None, guidance=None, filler=None, extractor=None,
                wording=None)


def listed(folder, hosted_mode=False):
    with TestClient(api.create_app(deps_for(folder), hosted_mode=hosted_mode)) as c:
        r = c.get("/api/sample-callers")
    assert r.status_code == 200
    return r.json()["callers"]


@pytest.fixture
def copy_of_starter(tmp_path):
    shutil.copytree(STARTER, tmp_path / "d")
    return tmp_path / "d"


def test_the_committed_file_is_what_the_script_makes_from_the_data():
    assert (STARTER / sample_callers.FILE).read_text(encoding="utf-8") == sample_callers.to_json(
        sample_callers.build(STARTER, TODAY))


def test_the_committed_page_is_what_the_script_makes_from_the_data():
    doc = sample_callers.build(STARTER, TODAY)
    assert (ROOT / "docs" / "sample-callers.md").read_text(encoding="utf-8") == sample_callers.to_markdown(doc)


def test_every_policyholder_is_listed_with_messages_that_carry_their_own_details():
    doc = json.loads((STARTER / sample_callers.FILE).read_text(encoding="utf-8"))
    people = json.loads((STARTER / "policyholders.json").read_text(encoding="utf-8"))
    assert [c["party_id"] for c in doc["callers"]] == [p["party_id"] for p in people]
    for c in doc["callers"]:
        assert c["good_for"] and c["opening_messages"]
        for m in c["opening_messages"]:
            assert m["label"] and m["text"] and "{" not in m["text"]


def test_a_policyholder_without_an_entry_stops_the_build(copy_of_starter):
    people = json.loads((copy_of_starter / "policyholders.json").read_text(encoding="utf-8"))
    people.append({**people[0], "party_id": "P999"})
    (copy_of_starter / "policyholders.json").write_text(json.dumps(people), encoding="utf-8")
    with pytest.raises(ValueError):
        sample_callers.build(copy_of_starter, TODAY)


def test_a_past_deadline_is_worded_as_passed_and_today_as_the_last_day():
    claims = {c["case_id"]: c for p in sample_callers.build(STARTER, TODAY)["callers"] for c in p["claims"]}
    assert "passed 13 days" in claims["CL-4201"]["appeal_deadline_note"]
    assert "last day" in claims["CL-4401"]["appeal_deadline_note"]
    assert sample_callers._deadline_note("2026-03-06", date(2026, 3, 5)) == "1 day after the agent's date"


def test_the_file_is_listed_for_the_bundled_data():
    callers = listed(STARTER)
    assert len(callers) == 9 and callers[0]["name"] == "Margaret Chen"


def test_no_file_means_an_empty_list(copy_of_starter):
    (copy_of_starter / sample_callers.FILE).unlink()
    assert listed(copy_of_starter) == []


def test_a_file_for_other_data_is_not_trusted(copy_of_starter):
    people = json.loads((copy_of_starter / "policyholders.json").read_text(encoding="utf-8"))
    people[0]["name"] = "Someone Real"
    (copy_of_starter / "policyholders.json").write_text(json.dumps(people), encoding="utf-8")
    assert listed(copy_of_starter) == []


def test_a_damaged_file_is_not_trusted(copy_of_starter):
    (copy_of_starter / sample_callers.FILE).write_text("[1, 2", encoding="utf-8")
    assert listed(copy_of_starter) == []
    (copy_of_starter / sample_callers.FILE).write_text("[]", encoding="utf-8")
    assert listed(copy_of_starter) == []


def test_hosted_lists_the_callers_for_the_bundled_data(monkeypatch):
    monkeypatch.delenv(hosted.ALLOW_OTHER_DATA, raising=False)
    assert len(listed(STARTER, hosted_mode=True)) == 9


def test_hosted_never_lists_them_when_other_data_is_allowed(monkeypatch, copy_of_starter):
    monkeypatch.setenv(hosted.ALLOW_OTHER_DATA, "1")
    assert listed(copy_of_starter, hosted_mode=True) == []
    assert listed(STARTER, hosted_mode=True) == []


def test_the_page_has_a_closed_labelled_drawer_and_fills_the_box_without_sending():
    html = (ROOT / "src/claims_agent/web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "src/claims_agent/web/app.js").read_text(encoding="utf-8")
    assert 'id="sample-callers"' in html and "sample data" in html.lower()
    assert '<details id="sample-callers" class="static" hidden>' in html  # collapsed (no `open`) until it has callers
    assert "/api/sample-callers" in js
