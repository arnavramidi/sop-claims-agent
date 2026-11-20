"""A consent fact carries no claim-status word, so it cannot unlock one in a reply."""

import re

import pytest

from claims_agent.reply.brief import ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.rules.await_consent import consent_fact
from claims_agent.settings import load_settings
from claims_agent.states import State

_RC = load_settings().reply_check
OUTCOMES = ("approved", "pending", "denied", "maybe", "no_request", "no_script")


def _brief(fact):
    return ReplyBrief(state=State.RESOLVE_INTENT, parts=(ReplyPart("verified", (fact,), "Plain."),), mood="calm",
                      caller_message="", previous_reply=None, english=True, verified=True)


@pytest.mark.parametrize("status", OUTCOMES)
def test_no_consent_fact_holds_a_status_word(status):
    fact = consent_fact(status)
    assert fact.kind == "consent"
    words = [w for forms in _RC.status_forms.values() for w in forms] + list(_RC.wary_status_words)
    for w in words:
        assert not re.search(rf"\b{re.escape(w)}\b", fact.text, re.I), (status, w)


def test_a_reply_saying_the_claim_was_approved_fails_on_the_approval_turn():
    r = check_reply("Thanks. Her claim was approved.", _brief(consent_fact("approved")),
                    wary_status_words=_RC.wary_status_words, data_statuses=("denied", "closed", "open"),
                    status_forms=_RC.status_forms)
    assert not r.passed
