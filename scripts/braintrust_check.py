"""Send one sample turn and one rating to Braintrust and read them back, to show the connection works.

Run: docker compose --profile test run --rm tests python scripts/braintrust_check.py
Needs BRAINTRUST_API_KEY in the container's environment (Compose passes it). Prints identifiers, never the key.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4

from claims_agent.observability import BraintrustLog, make_turn_log
from claims_agent.settings import load_settings

SAMPLE_MESSAGE = "Hi, my SSN is 123-45-4472 and my email is margaret@email.com. Why was my claim denied?"
SAMPLE_REPLY = "I can see claim CL-2048 was denied. Would you like to know which documents are outstanding?"


def main() -> int:
    key = os.environ.get("BRAINTRUST_API_KEY", "")
    log = make_turn_log(load_settings(), key)
    if not isinstance(log, BraintrustLog):
        print("tracing is off: no key, or the library could not start")
        return 1
    span_id = log.start(uuid4(), 1)
    log.finish(span_id, message=SAMPLE_MESSAGE, reply=SAMPLE_REPLY,
               record={"state_after": "RESOLVE_INTENT", "note": "connection check"}, tags=["connection_check"])
    log.feedback(span_id, rating=0, comment="connection check: thumbs down with a comment")
    log._logger.flush()
    project_id = log._logger.project.id
    print("project id:", project_id)
    print("span id:", span_id)
    query = {"query": f"SELECT id, input, output, scores, tags, metadata.turn_no AS turn_no FROM project_logs('{project_id}') "
                      f"WHERE id = '{span_id}' LIMIT 5", "fmt": "json"}
    for _ in range(12):  # the log API takes a few seconds to index a new row
        req = urllib.request.Request("https://api.braintrust.dev/btql", json.dumps(query).encode(),
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                rows = json.load(resp).get("data", [])
        except urllib.error.HTTPError as e:
            print("read-back refused:", e.code, e.read().decode()[:300])
            return 2
        if rows:
            for row in rows:
                print("row:", json.dumps(row))
            return 0
        time.sleep(5)
    print("the span was sent but not yet readable back")
    return 2


if __name__ == "__main__":
    sys.exit(main())
