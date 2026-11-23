"""The database actions a turn takes, as its record lists them: name, coarse outcome and times, never a value."""

import json

import pytest

from claims_agent.db import actions
from claims_agent.db.actions import action, rows, turn_db_actions
from claims_agent.http_view import for_http
from claims_agent.models.usage import collecting


@action("look up", rows)
async def look_up(secret: str) -> list[str]:
    return [secret, secret]


@action("outer")
async def outer() -> int:
    await look_up("inner")
    return 1


@action("fails")
async def fails() -> None:
    raise LookupError("no such row")


@action("odd outcome", lambda r: 1 / 0)
async def odd() -> str:
    return "kept"


async def test_an_action_is_listed_with_its_outcome_and_times_and_no_values():
    with collecting():
        assert await look_up("Margaret Chen 1985-03-15") == ["Margaret Chen 1985-03-15"] * 2
        got = turn_db_actions()
    [entry] = got["actions"]
    assert entry["action"] == "look up" and entry["outcome"] == {"rows": 2} and got["dropped"] == 0
    assert entry["ended"] >= entry["started"] > 0
    assert "Margaret" not in json.dumps(got)


async def test_outside_a_turn_nothing_is_kept():
    assert await look_up("x") == ["x", "x"]
    assert turn_db_actions() is None


async def test_an_action_inside_another_is_part_of_it():
    with collecting():
        await outer()
        assert [a["action"] for a in turn_db_actions()["actions"]] == ["outer"]


async def test_an_error_is_listed_by_its_kind_and_raised_unchanged():
    with collecting():
        with pytest.raises(LookupError, match="no such row"):
            await fails()
        assert turn_db_actions()["actions"][0]["outcome"] == {"error": "LookupError"}


async def test_a_broken_outcome_reader_never_changes_the_result():
    with collecting():
        assert await odd() == "kept"
        assert turn_db_actions()["actions"][0]["outcome"] == {}


async def test_past_the_cap_actions_are_only_counted(monkeypatch):
    monkeypatch.setattr(actions, "MAX_ACTIONS", 2)
    with collecting():
        for _ in range(5):
            await look_up("x")
        got = turn_db_actions()
    assert len(got["actions"]) == 2 and got["dropped"] == 3


def test_hosted_responses_carry_no_panel_so_no_database_actions_or_form():
    body = {"reply": "hi", "inspector": {"db_actions": {"actions": [], "dropped": 0}, "form_asked": {"readings": []}}}
    assert "inspector" not in for_http(body, hosted=True)
    assert "inspector" not in for_http({"turns": [body]}, hosted=True)["turns"][0]
