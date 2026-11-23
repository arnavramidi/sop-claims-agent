"""Send one real conversation of two turns through the real engine with Braintrust tracing on, then read the traces
back from Braintrust and print each turn as a tree (names, types, times, input/output sizes, scores).

Run: docker compose --profile test run --rm -T tests python scripts/trace_check.py
Needs OPENAI_API_KEY and BRAINTRUST_API_KEY in the container's environment (Compose passes them); the test database
must be migrated. Loads the starter data into it. Prints identifiers and trace shape, never a key. Makes real model calls.
"""

import asyncio
import json
import os
import sys
import time
import urllib.request

from claims_agent.api import model_clients, model_settler
from claims_agent.db import conversations as store
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps, run_turn
from claims_agent.loader.read import read_folder
from claims_agent.loader.write import write
from claims_agent.observability import make_turn_log
from claims_agent.settings import ROOT, load_settings, load_thresholds

MESSAGES = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare "
            "claim from January. DOB is 1985-03-15, SSN last four is 4472.",
            "Why was that claim denied?")


def fetch(key: str, project_id: str, limit: int) -> list[dict]:
    req = urllib.request.Request(f"https://api.braintrust.dev/v1/project_logs/{project_id}/fetch?limit={limit}",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp).get("events", [])


def size(v) -> str:
    return "-" if v is None else f"{len(json.dumps(v))}b"


async def main() -> int:
    key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not key or not os.environ.get("OPENAI_API_KEY"):
        print("needs BRAINTRUST_API_KEY and OPENAI_API_KEY")
        return 1
    s = load_settings()
    await write(read_folder(ROOT / "data" / "starter"), os.environ["LOADER_DATABASE_URL"])
    log = make_turn_log(s, key)
    filler, extractor, wording = model_clients(s)
    db = AppDb(os.environ["DATABASE_URL"], limits=s.database)
    deps = Deps(settings=s, thresholds=load_thresholds(), db=db, guidance=None, filler=filler, extractor=extractor,
                wording=wording, settler=model_settler(s), turn_log=log)
    spans = []
    async with db.conversation(None) as c:
        cid = await store.create(c, consent_script="default")
    for m in MESSAGES:
        r = await run_turn(deps, cid, m)
        print("turn:", r.state.value, "|", r.reply[:80])
    await db.close()
    log._logger.flush()
    for _ in range(15):
        time.sleep(4)
        events = fetch(key, log._logger.project.id, 200)
        roots = [e for e in events if e.get("is_root") and (e.get("metadata") or {}).get("conversation_id") == str(cid)]
        if len(roots) == 2 and all(any(e.get("span_parents") and e["root_span_id"] == r["root_span_id"] for e in events)
                                   for r in roots):
            break
    for root in sorted(roots, key=lambda e: e["metadata"]["turn_no"]):
        kids = sorted((e for e in events if e["root_span_id"] == root["root_span_id"] and not e.get("is_root")),
                      key=lambda e: e["metrics"].get("start") or 0)
        spans.append(root)
        print(f"\nTURN {root['metadata']['turn_no']}  scores={root.get('scores')}  tags={root.get('tags')}"
              f"  in={size(root['input'])} out={size(root['output'])} metrics={sorted(root.get('metrics') or {})}")
        print("  metadata keys:", sorted(root["metadata"]))
        t0 = root["metrics"].get("start") or 0
        for e in kids:
            a, m = e["span_attributes"], e.get("metrics") or {}
            print(f"  +{(m.get('start') or t0) - t0:6.2f}s {a['type']:<8} {a['name']:<22} in={size(e.get('input')):<7}"
                  f" out={size(e.get('output')):<7} tokens={m.get('tokens')}")
    return 0 if len(spans) == 2 else 2


sys.exit(asyncio.run(main()))
