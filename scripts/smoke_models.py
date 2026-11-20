"""One real call to each model client. Needs OPENAI_API_KEY. Run:
docker compose --profile test run --rm tests python scripts/smoke_models.py
Imports nothing from tests/ (the server image does not contain it). Prints no key and no request headers."""

import asyncio
import os
import sys
import time

from claims_agent.form.decisions_filler import DecisionsFiller
from claims_agent.form.extractor import TextExtractor
from claims_agent.form.questions import FormRequest
from claims_agent.models.client import make_client
from claims_agent.reply.brief import Fact, ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.reply.wording import ModelWording
from claims_agent.settings import load_settings
from claims_agent.states import State

MESSAGE = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
           "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")


class _Spy:
    """Wraps one SDK method: times each call and prints its token usage."""

    def __init__(self, label, fn):
        self._label, self._fn = label, fn

    async def __call__(self, **kw):
        t = time.monotonic()
        extra = f" questions={len(kw['questions'])}" if "questions" in kw else ""
        try:
            r = await self._fn(**kw)
        except Exception as e:  # report the kind, status and the API's error code/message; never headers or key
            body = getattr(e, "body", None)
            print(f"[{self._label}] REFUSED {type(e).__name__} status={getattr(e, 'status_code', None)} "
                  f"code={getattr(e, 'code', None)} body={body!r}"[:600])
            raise
        u = getattr(r, "usage", None)
        print(f"[{self._label}] ok model={kw['model']}{extra} seconds={time.monotonic() - t:.2f} usage={u.model_dump() if u else None}")
        return r


class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def spied(client):
    return _Ns(decisions=_Ns(create=_Spy("decisions", client.decisions.create)),
               responses=_Ns(parse=_Spy("extract", client.responses.parse),
                             create=_Spy("wording", client.responses.create)))


async def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set")
        return 1
    s = load_settings()
    client = spied(make_client())
    kw = {"retries": 0, "timeout": 60}
    req = FormRequest(message=MESSAGE, claim_types=("healthcare", "dental"), claim_statuses=("denied", "open"))
    form = await DecisionsFiller(client, s.models.fast, **kw).fill(req)
    print("form:", {"claim_type:healthcare": form.claim_types["healthcare"].p, "claim_status:denied": form.claim_statuses["denied"].p,
                    "month:1": form.months[1].p, "wants_person": form.wants_person.p,
                    "caller_role": (form.caller_role.value, form.caller_role.p),
                    "gave_identity_detail": form.gave_identity_detail.p})
    print("stated:", await TextExtractor(client, s.models.text, **kw).extract(
        MESSAGE, want_identity=True, want_claim_number=False))
    brief = ReplyBrief(
        state=State.PROCESS_CASE, parts=(ReplyPart("answer", facts=(
            Fact("amounts", "Allowed maximum for claim CL-2048: 1450.00", "CL-2048"),
            Fact("appeal_deadline", "Appeal deadline for claim CL-2048: 2026-03-18", "CL-2048"))),),
        mood="calm", caller_message="What is the limit and the appeal deadline?", previous_reply=None,
        english=True, verified=True)
    wording = ModelWording(client, s.models.text, s.models.larger, **kw)
    check = dict(wary_status_words=s.reply_check.wary_status_words, data_statuses=["denied", "open"])
    for label, reply in (("word", await wording.word(brief)),
                         ("rewrite", await wording.rewrite(brief, "It is approved.", ["status_word"]))):
        print(f"{label}: {reply!r} check={check_reply(reply, brief, **check)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
