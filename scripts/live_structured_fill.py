"""One real StructuredFiller call (the text model). Run: docker compose --profile test run --rm tests python scripts/live_structured_fill.py"""
import asyncio
import os

from claims_agent.form.questions import FormRequest, build_questions
from claims_agent.form.structured_filler import StructuredFiller
from claims_agent.models.client import make_client
from claims_agent.settings import load_settings

MESSAGE = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare "
           "claim from January. DOB is 1985-03-15.")


async def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set")
        return 1
    s = load_settings()
    req = FormRequest(message=MESSAGE, claim_types=("healthcare", "dental"), claim_statuses=("denied", "open"),
                      needs=("appeal",), relationships=("spouse",))
    form = await StructuredFiller(make_client(), s.models.text, retries=0, timeout=90).fill(req)
    print("questions:", len(build_questions(req)))
    print({"filler": form.wants_person.by, "caller_role": form.caller_role.value, "gave_identity_detail": form.gave_identity_detail.p,
           "mentions_claim": form.mentions_claim.p, "healthcare": form.claim_types["healthcare"].p, "dental": form.claim_types["dental"].p,
           "denied": form.claim_statuses["denied"].p, "month1": form.months[1].p, "month2": form.months[2].p,
           "no_year_said": form.no_year_said.p, "wants_person": form.wants_person.p, "english": form.english.p})
    return 0


raise SystemExit(asyncio.run(main()))
