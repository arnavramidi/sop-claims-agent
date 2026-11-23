"""What a turn's record adds for the inspector: the form as the fast model was asked it, the database actions the turn
took, the states it passed through and the models of each role."""

import json

from claims_agent.form.fields import Pick
from claims_agent.loader.read import read_folder
from claims_agent.states import State
from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, STARTER, make_form, say, stated, turn_record


def _questions(record: dict, reading: int = 0) -> dict:
    return {q["id"]: q for q in record["form_asked"]["readings"][reading]["questions"]}


def _actions(record: dict) -> list[str]:
    return [a["action"] for a in record["db_actions"]["actions"]]


async def test_the_form_as_asked_is_kept_and_changes_once_the_caller_is_verified(deps, conv):
    first = await say(deps, conv, "hello", make_form())
    second = await say(deps, conv, BRIEF_MSG, make_form(**GAVE), BRIEF_SAID)
    third = await say(deps, conv, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                      stated(claim_number="CL-2048"))
    before, after = first.inspector["form_asked"]["readings"][0], third.inspector["form_asked"]["readings"][0]
    assert before["verified"] is False and after["verified"] is True
    assert "gave_identity_detail" in _questions(first.inspector) and "gave_identity_detail" not in _questions(third.inspector)
    assert any(i.startswith("figure:") for i in _questions(third.inspector))
    assert before["count"] == len(before["questions"])
    assert second.inspector["form_asked"]["readings"][0]["state"] == "VERIFY_ID"
    assert (await turn_record(deps, third.turn_id)) == json.loads(json.dumps(third.inspector))


async def test_the_verifying_turn_lists_its_database_actions_without_any_value_or_party(deps, conv):
    r = await say(deps, conv, BRIEF_MSG, make_form(**GAVE), BRIEF_SAID)
    names = _actions(r.inspector)
    assert {"match identity", "bind policyholder"} <= set(names)
    by_name = {a["action"]: a for a in r.inspector["db_actions"]["actions"]}
    assert by_name["match identity"]["outcome"] == {} and by_name["bind policyholder"]["outcome"] == {"verified": True}
    said = json.dumps([[a["action"], a["outcome"]] for a in r.inspector["db_actions"]["actions"]])  # times aside
    for value in ("Margaret", "1985-03-15", "4472", "POL-9921", *{c.party_id for c in read_folder(STARTER).customers}):
        assert value not in said, value


async def test_details_that_match_no_one_say_nothing_about_how_many_matched(deps, conv):
    r = await say(deps, conv, "I'm Ann Smith born 1990-01-01", make_form(**GAVE),
                  stated(policyholder_name="Ann Smith", date_of_birth=["1990-01-01"]))
    assert all(a["outcome"] == {} for a in r.inspector["db_actions"]["actions"] if a["action"] == "match identity")


async def test_the_states_a_turn_passed_through_are_listed(deps, conv):
    r = await say(deps, conv, BRIEF_MSG, make_form(**GAVE, need="denial_question", mentions_claim=True,
                                                   claim_types={"healthcare": 0.95}, claim_statuses={"denied": 0.95},
                                                   months={1: 0.95}, how_many="one"), BRIEF_SAID)
    assert r.inspector["path"][0] == "VERIFY_ID" and len(r.inspector["path"]) >= 2
    assert r.inspector["path"][-1] == r.state.value or r.state == State.ESCALATED


async def test_an_answer_the_larger_model_was_asked_to_settle_says_so(deps, verified_conv):
    await say(deps, verified_conv, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
              stated(claim_number="CL-2048"))
    deps.settler.queue(Pick(value=None, by="larger"))
    r = await say(deps, verified_conv, "what about the timing?", make_form(answer_wanted=Pick(value=None)))
    settled = _questions(r.inspector)["answer_wanted"]["settled"]
    assert "larger" in settled["steps"] and settled["value"] is None and settled["larger"] == "cannot_tell"


async def test_the_record_names_the_model_of_each_role(deps, conv):
    r = await say(deps, conv, "hello", make_form())
    s = deps.settings.models
    assert r.inspector["models"] == {"fast": s.fast, "text": s.text, "larger": s.larger}
