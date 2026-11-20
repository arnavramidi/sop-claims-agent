"""Whole turns where one of a few matching claims is picked by its summary, and a list is pointed at."""

import pytest
from helpers import (GAVE, facts, focus, instructions, make_deps, make_form, memory, new_conversation, say,
                     stated)

from claims_agent.form.fields import Pick
from claims_agent.models.errors import ModelRequestRejected, ModelUnavailable
from claims_agent.states import State

TOOTH = make_form(how_many="one", months={8: 0.95}, no_year_said=True, mentions_claim=True)
SAID = "the one where I broke my tooth and went to the ER in August"


async def _dana(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "Dana Whitfield, born 1983-11-30, ID ends 9042", make_form(**GAVE),
              stated(policyholder_name="Dana Whitfield", date_of_birth=["1983-11-30"], id_digits="9042"))
    return cid


@pytest.fixture
async def descriptive_deps(load_dataset):
    return make_deps(await load_dataset("august_descriptive"))


@pytest.fixture
async def generic_deps(load_dataset):
    return make_deps(await load_dataset("august_generic"))


@pytest.fixture
async def single_deps(load_dataset):
    return make_deps(await load_dataset("august_single"))


@pytest.fixture
async def dana(descriptive_deps):
    return await _dana(descriptive_deps)


async def test_r13_descriptive_summaries_let_the_fast_model_pick(descriptive_deps, dana):
    descriptive_deps.filler.queue_one(Pick(value="CL-6001", p=0.9))
    r = await say(descriptive_deps, dana, SAID, TOOTH)
    assert await focus(descriptive_deps, dana) == ["CL-6001"]
    assert facts(r)[0].kind == "found_claim"
    offered = dict(descriptive_deps.filler.one_questions[0].choices)
    assert offered["CL-6001"].startswith("Emergency room") and set(offered) == {"CL-6001", "CL-6002", "cannot_tell"}


async def test_r14_generic_summaries_show_both_and_say_it_cannot_tell(load_dataset):
    deps = make_deps(await load_dataset("august_generic"))
    cid = await _dana(deps)
    deps.filler.queue_one(Pick(value="cannot_tell", p=0.8))
    r = await say(deps, cid, SAID, TOOTH)
    assert "which_claim" in instructions(r) and await memory(deps, cid, "focus.last_list") == ["CL-6002", "CL-6001"]
    assert "cannot_tell" in [f.kind for f in facts(r)]


async def test_r12_one_august_claim_names_its_type_and_quotes_its_summary(load_dataset):
    deps = make_deps(await load_dataset("august_single"))
    cid = await _dana(deps)
    r = await say(deps, cid, SAID, TOOTH)
    first = facts(r)[0]
    assert first.kind == "found_claim" and "dental" in first.text and "Dental claim completed" in first.text \
        and "emergency" not in first.text.lower()


async def test_x14_a_summary_pick_outside_the_candidates_is_rejected(descriptive_deps, dana):
    descriptive_deps.filler.queue_one(Pick(value="CL-2048", p=0.95))
    r = await say(descriptive_deps, dana, "the tooth one in August", TOOTH)
    assert await focus(descriptive_deps, dana) == [] and "which_claim" in instructions(r)


async def test_an_unsure_summary_pick_asks_again_without_guessing(descriptive_deps, dana):
    descriptive_deps.filler.queue_one(Pick(value="CL-6001", p=0.3))
    r = await say(descriptive_deps, dana, SAID, TOOTH)
    assert await focus(descriptive_deps, dana) == [] and "which_claim" in instructions(r)
    assert "cannot_tell" not in [f.kind for f in facts(r)]


async def test_describing_a_claim_in_answer_to_the_list_picks_it(descriptive_deps, dana):
    await say(descriptive_deps, dana, "my August claim", TOOTH)
    assert await memory(descriptive_deps, dana, "focus.last_list") == ["CL-6002", "CL-6001"]
    descriptive_deps.filler.queue_one(Pick(value="CL-6002", p=0.9))
    await say(descriptive_deps, dana, "the cleaning one", make_form())
    assert await focus(descriptive_deps, dana) == ["CL-6002"]
    asked = descriptive_deps.filler.one_questions[-1]
    assert {v for v, _ in asked.choices} == {"CL-6001", "CL-6002", "cannot_tell"}


@pytest.mark.parametrize("error", [ModelUnavailable("m"), ModelRequestRejected("m")])
async def test_a_model_failure_asks_the_caller_with_the_same_list(descriptive_deps, dana, error):
    descriptive_deps.filler.queue_one(error)
    r = await say(descriptive_deps, dana, SAID, TOOTH)
    assert await focus(descriptive_deps, dana) == [] and "which_claim" in instructions(r)
    assert r.state == State.RESOLVE_INTENT


async def test_r10_listed_claims_carry_dates_so_the_newer_one_can_be_picked(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    await say(deps, cid, "my healthcare claim from January",
              make_form(how_many="one", claim_types={"healthcare": 0.95}, months={1: 0.95}, no_year_said=True))
    await say(deps, cid, "the newer one", make_form(listed_claim="CL-2048"))
    labels = dict(deps.filler.requests[-1].listed_claims and
                  [(c.case_id, c.created_at) for c in deps.filler.requests[-1].listed_claims])
    assert "2026-01-12" in labels["CL-2048"] and "2025-01-28" in labels["CL-2011"]
    assert await focus(deps, cid) == ["CL-2048"]
