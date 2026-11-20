"""The assignment's own test case, whole turns over the starter data: the policyholder's message: verified and answered in
one turn, with no claim data reaching a model before verification."""

import json
from dataclasses import asdict

from helpers import BRIEF_MSG, BRIEF_SAID, GAVE, assert_no_claim_data, facts, focus, make_form, say, verified_party

from claims_agent.form.fields import Pick
from claims_agent.states import State

# The brief's message as the fast model should read it, with the need read as a denial question.
FORM = make_form(**GAVE, need=Pick(value="denial_question", p=0.7), how_many="one",
                 claim_types={"healthcare": 0.97}, claim_statuses={"denied": 0.96}, months={1: 0.95}, no_year_said=True)


async def test_the_briefs_message_is_verified_and_answered_in_one_turn(deps, conv):
    r = await say(deps, conv, BRIEF_MSG, FORM, BRIEF_SAID)
    assert await verified_party(deps, conv) == "P9" and await focus(deps, conv) == ["CL-2048"]
    assert r.state == State.PROCESS_CASE
    t = " ".join(f.text for f in facts(r))
    assert "denied" in t and "pathology report" in t and "office note" in t and "2026-03-18" in t and "13 days" in t
    assert "CL-2011" not in t and r.inspector["reply_source"] in ("live", "rewrite", "template")
    kinds = [f.kind for f in facts(r) if f.claim == "CL-2048"]
    assert {"status", "denial_reason", "documents", "deadline"} <= set(kinds)


async def test_no_claim_data_reaches_a_model_before_verification(deps, conv):
    await say(deps, conv, BRIEF_MSG, FORM, BRIEF_SAID)
    assert_no_claim_data(json.dumps(asdict(deps.filler.requests[0]), default=str))
    assert_no_claim_data(json.dumps(asdict(deps.extractor.requests[0]), default=str))
