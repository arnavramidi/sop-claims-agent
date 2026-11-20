"""After verification the form is told what the figures and the answers mean, so a caller who describes a figure by
its meaning ("what you actually paid") or asks what a figure means is understood."""

from helpers import make_form, say

from claims_agent.form.fields import Pick


async def test_the_figure_questions_carry_the_glossary_meanings(deps, verified_conv):
    await say(deps, verified_conv, "what did you actually pay?", make_form())
    req = deps.filler.requests[-1]
    assert req.figure_meanings["net_pay"] == "The finalized amount the insurer paid for the claim."


async def test_the_amounts_answer_says_it_gives_what_each_figure_means(deps, verified_conv):
    await say(deps, verified_conv, "what does allowed amount mean?", make_form())
    answers = dict(deps.filler.requests[-1].answers)
    assert "mean" in answers["amounts"] and "status" in answers["status"]


async def test_nothing_of_the_glossary_goes_in_before_verification(deps, conv):
    await say(deps, conv, "hello there", make_form())
    assert deps.filler.requests[-1].figure_meanings == {}


async def test_a_question_about_one_figure_with_no_sum_asked_gets_the_figures_and_their_meanings(deps, verified_conv):
    from helpers import facts, stated
    await say(deps, verified_conv, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
              stated(claim_number="CL-2048"))
    r = await say(deps, verified_conv, "What does allowed amount mean?",
                  make_form(answer_wanted=Pick(value="something_else", p=0.8, options={"something_else": 0.8}),
                            figures={"allowed_max_amount": 0.94}))
    told = " ".join(f.text for f in facts(r) if f.kind == "amount")
    assert "Meaning of allowed max amount" in told and "not_on_file" not in [p["instruction"] for p in r.inspector["parts"]]
