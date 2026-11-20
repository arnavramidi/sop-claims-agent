"""Each part of a message that asks several things is settled on its own words: when its answer pick is unsure, the
larger model is asked about that part, not about the whole message (where two parts ask for two answers and it can
only say it cannot tell)."""

from helpers import GAVE, facts, focus, make_form, new_conversation, say, stated
from test_multi_question import say_parts

from claims_agent.form.fields import Pick

WHOLE = "What does allowed amount mean, and what if the clinic won't reissue the report?"
FIRST = "What does allowed amount mean?"


async def test_an_unsure_part_is_settled_on_that_part_alone(deps):
    cid = await new_conversation(deps)
    await say(deps, cid, "Margaret Chen, born 1985-03-15, ID ends 4472", make_form(**GAVE),
              stated(policyholder_name="Margaret Chen", date_of_birth=["1985-03-15"], id_digits="4472"))
    await say(deps, cid, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
              stated(claim_number="CL-2048"))
    assert await focus(deps, cid) == ["CL-2048"]
    deps.settler.queue(Pick(value="amounts", p=0.9, by="larger"))
    r = await say_parts(deps, cid, WHOLE, [
        (FIRST, make_form(answer_wanted=Pick(value="amounts", p=0.5, options={"amounts": 0.5, "documents_needed": 0.4}))),
        ("What if the clinic won't reissue the report?", make_form(cant_get_document=True, which_document="pathology report"))])
    asked, question = deps.settler.calls[0]
    assert question.name == "answer_wanted" and asked.message == FIRST
    assert "amount" in [f.kind for f in facts(r)]
