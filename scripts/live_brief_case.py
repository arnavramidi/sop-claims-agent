"""The assignment's test case against the real models, and the upset-caller conversations from its bonus part.
Needs OPENAI_API_KEY and a loaded database. Run:
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
# An upset caller who will not give details at first, repeats the demand, then gives them.
FRUSTRATED = (
    ANGRY,
    "I'm not doing this again. Why was my claim denied? Just answer the question.",
    "Fine. Margaret Chen, born 1985-03-15, SSN ends in 4472.",
)
# An upset caller who takes the person offered instead.
TAKES_A_PERSON = (FRUSTRATED[0], FRUSTRATED[1], "Yes. Put me through to someone.")
# A caller who is verified first, then asks about the denial upset.
UPSET_AFTER_VERIFYING = (
    "Margaret Chen, date of birth 1985-03-15, last four of my SSN 4472.",
    "I'm really upset about this. Why on earth was my healthcare claim denied?",
)


def _form_line(r) -> str:
    form = r.inspector.get("form") or {}
    mood = form.get("mood") or {}
    person = (form.get("wants_person") or {}).get("p")
    return f"mood {mood.get('value')} ({mood.get('p')}), wants person {person}"


async def conversation(deps: Deps, label: str, messages) -> None:
    async with deps.db.conversation(None) as c:
        cid = await store.create(c, consent_script="default")
    print(f"=== {label}")
    for message in messages:
        t = time.monotonic()
        r = await run_turn(deps, cid, message)
        seconds = time.monotonic() - t
        async with deps.db.conversation(cid) as c:
            verified = (await (await c.execute("SELECT verified_party_id FROM conversations WHERE id = %s",
                                               [cid])).fetchone())[0]
        print(f"--- turn ({seconds:.1f} s)")
        print("caller:", message)
        print("form:", _form_line(r))
        print("state:", r.state.value, "| verified customer:", verified,
              "| claims in focus:", r.inspector.get("memory", {}).get("focus", {}).get("claim_ids"),
              "| stalled:", r.inspector.get("counts", {}).get("stalled"))
        print("parts:", [p["instruction"] for p in r.inspector.get("parts", [])])
        print("facts:", [f["kind"] for p in r.inspector.get("parts", []) for f in p["facts"]])
        print("reply source:", r.inspector.get("reply_source"), "| checks:",
              [c["failures"] for c in r.inspector.get("checks", [])])
        print("reply:", r.reply)
        if os.environ.get("LIVE_DEBUG"):
            print("hints:", r.inspector.get("memory", {}).get("hints"))
            print("fired:", r.inspector.get("fired"))
            f = r.inspector.get("form") or {}
            print("form picks:", {k: v for k, v in f.items() if isinstance(v, dict) and "value" in v})
            print("months/statuses/types:", {k: {a: b.get("p") for a, b in f.get(k, {}).items() if (b.get("p") or 0) > 0.2}
                                             for k in ("months", "claim_statuses", "claim_types", "years")})


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
        await conversation(deps, "the brief's test message", (BRIEF_MSG,))
        await conversation(deps, "an upset caller who is not verified, then gives the details", FRUSTRATED)
        await conversation(deps, "an upset caller who takes the person offered", TAKES_A_PERSON)
        await conversation(deps, "a verified caller who is upset about the denial", UPSET_AFTER_VERIFYING)
    finally:
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
