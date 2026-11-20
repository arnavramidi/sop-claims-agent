"""The assignment's test case against the real models. Needs OPENAI_API_KEY and a loaded database. Run:
docker compose run --rm setup
docker compose run --rm server python scripts/live_brief_case.py
Imports nothing from tests/ (the server image does not contain it). Prints no key."""

import asyncio
import os
import sys
import time

from claims_agent.db import conversations as store
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps, run_turn
from claims_agent.form.decisions_filler import DecisionsFiller
from claims_agent.form.extractor import TextExtractor
from claims_agent.models.client import make_client
from claims_agent.reply.wording import ModelWording
from claims_agent.settings import load_settings, load_thresholds

BRIEF_MSG = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare "
             "claim from January. DOB is 1985-03-15, SSN last four is 4472.")
ANGRY = "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."


async def one_conversation(deps: Deps, label: str, message: str) -> None:
    async with deps.db.conversation(None) as c:
        cid = await store.create(c, consent_script="default")
    t = time.monotonic()
    r = await run_turn(deps, cid, message)
    seconds = time.monotonic() - t
    async with deps.db.conversation(cid) as c:
        verified = (await (await c.execute("SELECT verified_party_id FROM conversations WHERE id = %s",
                                           [cid])).fetchone())[0]
    print(f"--- {label} ({seconds:.1f} s)")
    print("caller:", message)
    print("state:", r.state.value)
    print("verified customer:", verified)
    print("claims in focus:", r.inspector.get("memory", {}).get("focus", {}).get("claim_ids"))
    print("reply source:", r.inspector.get("reply_source"))
    print("reply checks:", r.inspector.get("checks"))
    print("reply:", r.reply)


async def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set")
        return 1
    s, client = load_settings(), make_client()
    kw = {"retries": s.limits.model_retries, "timeout": s.limits.model_timeout_seconds}
    db = AppDb(os.environ["DATABASE_URL"])
    deps = Deps(settings=s, thresholds=load_thresholds(), db=db, guidance=None,
                filler=DecisionsFiller(client, s.models.fast, **kw),
                extractor=TextExtractor(client, s.models.text, **kw),
                wording=ModelWording(client, s.models.text, s.models.larger, **kw))
    try:
        await one_conversation(deps, "the brief's test message", BRIEF_MSG)
        await one_conversation(deps, "an angry caller, not verified", ANGRY)
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
