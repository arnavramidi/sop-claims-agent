"""What a reply part says the reply must mention (every option still open, the rule) or must not (a thank-you for a
detail that was not kept), checked by code like the rest of the reply check; and the extractor's context."""

import json

from claims_agent.form.extractor import TextExtractor
from claims_agent.form.fields import StatedValues
from claims_agent.memory import IdentityMemory, HeldValue
from claims_agent.reply.brief import ReplyBrief, ReplyPart
from claims_agent.reply.check import check_reply
from claims_agent.rules import verify_id
from claims_agent.settings import load_settings
from claims_agent.states import State

SETTINGS = load_settings()


def _check(text: str, *parts: ReplyPart):
    brief = ReplyBrief(state=State.VERIFY_ID, parts=parts, mood="calm", caller_message="x", previous_reply=None,
                       english=True, verified=False)
    rc = SETTINGS.reply_check
    return check_reply(text, brief, wary_status_words=rc.wary_status_words, data_statuses=("denied",),
                       status_forms=rc.status_forms)


def _held(*kinds: str) -> IdentityMemory:
    return IdentityMemory(held={k: HeldValue(as_stated="x", key="x", turn=1) for k in kinds})


def test_a_reply_that_asks_for_one_detail_when_several_are_open_fails():
    # The wording model was told to "ask for one more identifying detail" and asked for the phone number alone.
    part = verify_id._ask(_held("name", "dob"), ["dob"])
    r = _check("Thanks for that. Could you please provide your phone number to help verify your identity?", part)
    assert not r.passed and r.failures == ("required_mention_missing",)


def test_a_reply_that_says_the_rule_and_every_open_option_passes():
    part = verify_id._ask(_held("name", "dob"), ["dob"])
    r = _check("Thank you. To verify who you are, 3 details must match the policy. You can give your phone number, "
               "email address, or the last four digits of your SSN or national ID, or several at once.", part)
    assert r.passed, r.failures


def test_every_ask_says_the_rule_and_names_every_open_kind_and_its_plain_text_passes_its_own_check():
    for held, kept in (((), []), (("name",), ["name"]), (("name", "dob"), []), (("name", "dob", "phone"), [])):
        ident = _held(*held)
        part = verify_id._ask(ident, kept, lost=not kept)
        open_kinds = [k for k in verify_id.KINDS if k not in held]
        assert part.must_mention == (verify_id.THE_RULE, *(verify_id.MENTION[k] for k in open_kinds))
        assert _check(part.plain, part).passed, part.plain


def test_a_detail_that_was_not_kept_is_not_thanked_for():
    part = verify_id._ask(_held("name", "dob"), [], lost=True)
    assert "Thank" not in part.plain
    r = _check("Thank you. To verify who you are, 3 details must match the policy. Could you give your phone number, "
               "email address or the last four digits of your SSN or national ID?", part)
    assert r.failures == ("forbidden_mention",)
    kept = verify_id._ask(_held("name", "dob"), ["dob"], lost=True)
    assert kept.plain.startswith("Thank you.") and kept.must_not_mention == ()
    assert any(f.kind == "detail_kept" and "date of birth" in f.text for f in kept.facts)


def test_a_message_with_no_detail_is_not_barred_from_a_greeting_thanks():
    assert verify_id._ask(_held(), [], lost=False).must_not_mention == ()


def test_a_refusal_names_every_other_kind_and_the_rule():
    ident = _held("name")
    ident.refused = ["phone"]
    part = verify_id._refusal(ident, ["phone"])[0]
    assert part.must_mention == (verify_id.THE_RULE, *(verify_id.MENTION[k] for k in ("dob", "email", "id")))
    assert _check(part.plain, part).passed, part.plain


def test_the_could_not_read_reply_names_the_kind_the_extractor_found_and_never_the_value():
    part = verify_id._not_read(StatedValues(phone="650"))
    assert part.instruction == "restate_value" and [f.text for f in part.facts] == ["your phone number"]
    assert "650" not in part.plain and "Thank" not in part.plain
    unknown = verify_id._not_read(StatedValues())
    assert unknown.instruction == "unreadable_detail" and "couldn't read" in unknown.plain


class _Client:
    class responses:
        calls: list = []

        @staticmethod
        async def parse(**kw):
            _Client.responses.calls.append(kw)

            class R:
                status, output, output_parsed = "completed", [], StatedValues()
            return R()


async def test_the_extractor_gets_the_previous_reply_and_what_was_asked_as_data_only():
    # "16505212836" alone came back with every field empty: the extractor was never told the agent had just asked
    # for a phone number.
    _Client.responses.calls.clear()
    await TextExtractor(_Client, "m", retries=0, timeout=5).extract(
        "16505212836", want_identity=True, want_claim_number=True,
        previous_reply="Could you please provide your phone number?", just_asked="your phone number",
        asked_for=("phone number", "email address"))
    call = _Client.responses.calls[0]
    record = json.loads(call["input"][1]["content"])
    assert record == {"caller_message": "16505212836",
                      "agent_previous_reply": "Could you please provide your phone number?",
                      "agent_just_asked": "your phone number", "agent_asked_for": ["phone number", "email address"]}
    developer = call["input"][0]["content"]
    assert "caller_message alone" in developer and "agent_previous_reply" in developer
    assert "Could you please provide" not in developer  # the context never reaches the instructions


async def test_the_extractor_record_holds_only_the_message_when_there_is_no_context():
    _Client.responses.calls.clear()
    await TextExtractor(_Client, "m", retries=0, timeout=5).extract("x", want_identity=True, want_claim_number=True)
    assert json.loads(_Client.responses.calls[0]["input"][1]["content"]) == {"caller_message": "x"}
