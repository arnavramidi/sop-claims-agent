"""Replies worded ahead in the background: when one is used, when it is not, and that it is claim data the database
keeps to its own conversation and customer."""

import asyncio
import logging
import uuid
from dataclasses import replace

import psycopg
import pytest
from fastapi import BackgroundTasks
from helpers import GAVE, facts, focus, make_deps, make_form, memory, new_conversation, say, stated

from claims_agent.form.fields import Pick
from claims_agent.models.errors import ModelKeyRejected, ModelUnavailable
from claims_agent.prepared import prepare_ahead, run_in_background, schedule
from claims_agent.reply.wording import ScriptedWording
from claims_agent.settings import PreparedReplies
from claims_agent.states import State

pytestmark = pytest.mark.db

ASK_TIMING = make_form(answer_wanted=Pick(value="submission_timing", p=0.92))


# Generous: a loaded machine may be slow, and nothing here waits this long unless something is wrong.
WAIT = 30


async def _held_job(deps, conversation_id):
    """Prepare replies in the background with the first wording call held at its gate; returns once the job is
    parked there (the gate, and the job)."""
    gate = deps.wording.hold()
    job = asyncio.create_task(prepare_ahead(deps, conversation_id))
    entered = asyncio.create_task(deps.wording.entered.wait())
    done, _ = await asyncio.wait({entered, job}, timeout=WAIT, return_when=asyncio.FIRST_COMPLETED)
    if entered not in done:
        entered.cancel()
        raise AssertionError("the background job never reached its wording call")
    return gate, job


def switched_on(deps):
    return replace(deps, settings=deps.settings.model_copy(update={"prepared_replies": PreparedReplies(enabled=True)}))


async def _verified(deps, name, dob, last4):
    cid = await new_conversation(deps)
    r = await say(deps, cid, f"{name}, born {dob}, ID ends {last4}", make_form(**GAVE),
                  stated(policyholder_name=name, date_of_birth=[dob], id_digits=last4))
    assert r.state == State.RESOLVE_INTENT
    return cid


@pytest.fixture
def on(deps):
    """The same dependencies (same database, same scripted models) with prepared replies switched on."""
    return switched_on(deps)


@pytest.fixture
async def denial_case(on):
    """P9 asking why CL-2048 was denied: in PROCESS_CASE, the opening given, the first follow-up offered."""
    cid = await _verified(on, "Margaret Chen", "1985-03-15", "4472")
    r = await say(on, cid, "about CL-2048", make_form(quoted_claim_number=True, need="denial_question"),
                  stated(claim_number="CL-2048"))
    assert r.state == State.PROCESS_CASE and await focus(on, cid) == ["CL-2048"]
    return cid


@pytest.fixture
async def on_many(load_dataset):
    return switched_on(make_deps(await load_dataset("many_claims")))


@pytest.fixture
async def omar_five_denied_in_focus(on_many):
    cid = await _verified(on_many, "Omar Reyes", "1978-04-09", "2210")
    r = await say(on_many, cid, "why were my denied claims denied?",
                  make_form(need="denial_question", how_many="all", claim_statuses={"denied": 0.95}))
    assert r.state == State.PROCESS_CASE and len(await focus(on_many, cid)) == 5
    return cid


async def _rows(deps, cid, query, params):
    async with deps.db.conversation(cid) as c:
        return await (await c.execute(query, params)).fetchall()


async def prepared_rows(deps, cid, answer=None) -> int:
    """Prepared replies the conversation can see (the server's own account, bound to it)."""
    if answer is None:
        return (await _rows(deps, cid, "SELECT count(*) FROM prepared_replies", []))[0][0]
    return (await _rows(deps, cid, "SELECT count(*) FROM prepared_replies WHERE answer = %s", [answer]))[0][0]


async def prepared_claim_count(deps, cid) -> int:
    return (await _rows(deps, cid, "SELECT count(DISTINCT case_id) FROM prepared_replies", []))[0][0]


# ---- when a prepared reply is used ----------------------------------------------------------------------------------------

async def test_a_confident_calm_single_question_uses_the_prepared_reply(on, denial_case):
    assert await prepare_ahead(on, denial_case) == 2          # alternatives and submission timing are linked to denial_question
    worded = len(on.wording.briefs); r = await say(on, denial_case, "when do I need to send it?", ASK_TIMING)
    assert r.inspector["reply_source"] == "prepared" and len(on.wording.briefs) == worded and "within a week" in r.reply


async def test_an_upset_caller_gets_a_live_reply(on, denial_case):
    await prepare_ahead(on, denial_case)
    r = await say(on, denial_case, "this is ridiculous, when do I send it?", ASK_TIMING.model_copy(update={"mood": Pick(value="upset", p=0.9)}))
    assert r.inspector["reply_source"] == "live"


async def test_a_pick_below_the_threshold_gets_a_live_reply(on, denial_case):
    await prepare_ahead(on, denial_case)
    assert (await say(on, denial_case, "timing?", make_form(answer_wanted=Pick(value="submission_timing", p=0.7)))).inspector["reply_source"] == "live"


async def test_a_claim_that_changed_discards_its_prepared_reply(on, denial_case, loader_conn):
    await prepare_ahead(on, denial_case)
    await loader_conn.execute("UPDATE claims SET appeal_deadline = '2026-03-25' WHERE case_id = 'CL-2048'"); await loader_conn.commit()
    r = await say(on, denial_case, "when do I need to send it?", ASK_TIMING)
    assert r.inspector["reply_source"] == "live" and "2026-03-25" in " ".join(f.text for f in facts(r)) and await prepared_rows(on, denial_case, "submission_timing") == 0


async def test_switching_claims_discards_prepared_replies(on, denial_case):
    await prepare_ahead(on, denial_case)
    await say(on, denial_case, "and my auto claim?", make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one"))
    assert await prepared_rows(on, denial_case) == 0


async def test_nothing_is_prepared_with_the_switch_off_or_before_verification(deps, on, denial_case, conv):
    assert await prepare_ahead(deps, denial_case) == 0 and await prepare_ahead(on, conv) == 0


async def test_at_most_three_claims_are_prepared(on_many, omar_five_denied_in_focus):
    await prepare_ahead(on_many, omar_five_denied_in_focus); assert await prepared_claim_count(on_many, omar_five_denied_in_focus) == 3


async def test_a_prepared_reply_that_fails_the_check_is_not_stored(on, denial_case):
    on.wording.queue(["See https://x.example", "See https://x.example"] * 2); assert await prepare_ahead(on, denial_case) == 0


async def test_a_message_during_preparation_is_answered_live(on, denial_case):
    gate, job = await _held_job(on, denial_case)
    r = await asyncio.wait_for(say(on, denial_case, "when do I need to send it?", ASK_TIMING), timeout=WAIT)
    assert r.inspector["reply_source"] == "live"; gate.set(); await job


# ---- more -----------------------------------------------------------------------------------------------------

async def test_a_yes_to_the_offer_uses_the_prepared_reply_and_the_next_offer_is_plain_text(on, denial_case):
    await prepare_ahead(on, denial_case)
    worded = len(on.wording.briefs)
    r = await say(on, denial_case, "yes please", make_form(pending_reply=Pick(value="yes", p=0.95)))
    assert r.inspector["reply_source"] == "prepared" and len(on.wording.briefs) == worded
    assert r.reply.endswith("Would you like to hear about submission timing for claim CL-2048?")
    assert {"rule": "prepared_reply", "claim": "CL-2048", "answer": "missing_required_material_alternatives",
            "used": True} in r.inspector["fired"]


async def test_a_message_asking_several_things_is_worded_live(on, denial_case):
    await prepare_ahead(on, denial_case)
    r = await say(on, denial_case, "when do I need to send it?", ASK_TIMING.model_copy(update={"multi_question": make_form(multi_question=True).multi_question}))
    assert r.inspector["reply_source"] == "live"


async def test_a_new_today_changes_the_facts_and_the_reply_is_worded_live(on, denial_case):
    await prepare_ahead(on, denial_case)
    async with on.db.conversation(denial_case) as c:
        await c.execute("UPDATE conversations SET today = '2026-03-10' WHERE id = %s", [denial_case])
    r = await say(on, denial_case, "when do I need to send it?", ASK_TIMING)
    assert r.inspector["reply_source"] == "live" and await prepared_rows(on, denial_case, "submission_timing") == 0


async def test_a_state_whose_freedom_level_forbids_it_prepares_nothing_and_uses_nothing(on, denial_case):
    await prepare_ahead(on, denial_case)
    strict = replace(on, settings=on.settings.model_copy(update={"freedom": {
        **on.settings.freedom, "free": on.settings.freedom["free"].model_copy(update={"prepared_replies": False})}}))
    assert await prepare_ahead(strict, denial_case) == 0
    assert (await say(strict, denial_case, "when do I need to send it?", ASK_TIMING)).inspector["reply_source"] == "live"


async def test_the_inspector_lists_the_prepared_replies_without_their_text(on, denial_case):
    await prepare_ahead(on, denial_case)
    r = await say(on, denial_case, "timing?", make_form(answer_wanted=Pick(value="submission_timing", p=0.7)))
    assert r.inspector["prepared"] == [{"claim": "CL-2048", "answer": "missing_required_material_alternatives"},
                                       {"claim": "CL-2048", "answer": "submission_timing"}]


async def test_a_different_person_reset_discards_prepared_replies(on, denial_case, loader_conn):
    await prepare_ahead(on, denial_case)
    assert await prepared_rows(on, denial_case) == 2
    r = await say(on, denial_case, "what about my husband's policy?", make_form(other_person_policy=True))
    assert r.state == State.VERIFY_ID and await memory(on, denial_case, "focus.claim_ids") == []
    # Deleted, not only hidden: the loader account sees past the row rule.
    assert (await (await loader_conn.execute("SELECT count(*) FROM prepared_replies")).fetchone())[0] == 0


async def test_a_focus_change_while_replies_are_worded_drops_them(on, denial_case):
    gate, job = await _held_job(on, denial_case)
    await asyncio.wait_for(say(on, denial_case, "and my auto claim?", make_form(answer_wanted="something_else", mentions_claim=True, claim_types={"auto": 0.95}, search_relation="new_search", how_many="one")), timeout=WAIT)
    gate.set()
    assert await job == 0 and await prepared_rows(on, denial_case) == 0


async def test_a_customer_cleared_while_replies_are_worded_gets_none_stored(on, denial_case, loader_conn):
    gate, job = await _held_job(on, denial_case)
    async with on.db.conversation(denial_case) as c:
        await c.execute("SELECT clear_verified(%s)", [denial_case])
    gate.set()
    assert await job == 0
    assert (await (await loader_conn.execute("SELECT count(*) FROM prepared_replies")).fetchone())[0] == 0


class _Failing(ScriptedWording):
    def __init__(self, error):
        super().__init__()
        self.error, self.calls = error, 0

    async def word(self, brief):
        self.calls += 1
        raise self.error


async def test_a_model_outage_in_the_job_is_logged_and_dropped(on, denial_case, caplog):
    failing = replace(on, wording=_Failing(ModelUnavailable("scripted")))
    with caplog.at_level(logging.WARNING, logger="claims_agent.prepared"):
        await run_in_background(failing, denial_case)
    assert await prepared_rows(on, denial_case) == 0 and failing.wording.calls == 1
    assert any("model call failed" in m for m in caplog.messages)


async def test_a_rejected_key_in_the_job_is_logged_once_and_never_retried(on, denial_case, caplog, monkeypatch):
    from claims_agent import prepared
    monkeypatch.setattr(prepared, "_setup_logged", set())
    failing = replace(on, wording=_Failing(ModelKeyRejected("scripted", "rejected")))
    with caplog.at_level(logging.ERROR, logger="claims_agent.prepared"):
        await run_in_background(failing, denial_case)
        await run_in_background(failing, denial_case)
    assert failing.wording.calls == 2  # one call per job: no retry within a job
    assert sum("ModelKeyRejected" in m for m in caplog.messages) == 1


async def test_the_job_is_scheduled_after_a_reply_only_with_the_switch_on_and_in_process_case(deps, on, denial_case):
    off, wrong_state, right = BackgroundTasks(), BackgroundTasks(), BackgroundTasks()
    schedule(off, deps, denial_case, State.PROCESS_CASE)
    schedule(wrong_state, on, denial_case, State.RESOLVE_INTENT)
    schedule(right, on, denial_case, State.PROCESS_CASE)
    assert (len(off.tasks), len(wrong_state.tasks), len(right.tasks)) == (0, 0, 1)
    await right()  # what the framework runs after the response is sent
    assert await prepared_rows(on, denial_case) == 2


# ---- the table is claim data at rest: its own conversation and its verified customer only ----------------------

async def _app_insert(app_db, conversation_id, party_id, case_id="CL-3001"):
    async with app_db.conversation(conversation_id) as c:
        await c.execute("INSERT INTO prepared_replies (conversation_id, case_id, answer, text, claim_updated_at, "
                        "focus_key, party_id, facts) VALUES (%s, %s, 'submission_timing', 'x', now(), '[]', %s, '[]')",
                        [conversation_id, case_id, party_id])


async def _app_count(app_db, conversation_id) -> int:
    async with app_db.conversation(conversation_id) as c:
        return (await (await c.execute("SELECT count(*) FROM prepared_replies")).fetchone())[0]


@pytest.fixture
async def second_p12(seed, loader_conn, owner_conn):
    """Another conversation verified as the same customer, P12."""
    cid = uuid.uuid4()
    await loader_conn.execute("INSERT INTO conversations (id, state, today) VALUES (%s, 'PROCESS_CASE', '2026-03-05')", [cid])
    await loader_conn.commit()
    await owner_conn.execute("UPDATE conversations SET verified_party_id = 'P12' WHERE id = %s", [cid])
    await owner_conn.commit()
    return cid


async def test_one_conversation_cannot_read_anothers_prepared_replies(app_db, seed, second_p12):
    await _app_insert(app_db, seed.verified_p12, "P12")
    assert await _app_count(app_db, seed.verified_p12) == 1
    assert await _app_count(app_db, second_p12) == 0       # the same customer, another conversation
    assert await _app_count(app_db, seed.unverified) == 0
    assert await _app_count(app_db, None) == 0


async def test_after_the_verified_customer_is_cleared_old_prepared_replies_are_invisible(app_db, seed, loader_conn):
    await _app_insert(app_db, seed.verified_p12, "P12")
    async with app_db.conversation(seed.verified_p12) as c:
        await c.execute("SELECT clear_verified(%s)", [seed.verified_p12])  # no delete of prepared rows: the rule alone
    assert await _app_count(app_db, seed.verified_p12) == 0
    assert (await (await loader_conn.execute("SELECT count(*) FROM prepared_replies")).fetchone())[0] == 1


async def test_a_row_for_another_customer_is_invisible_and_cannot_be_written(app_db, seed, loader_conn):
    await loader_conn.execute("INSERT INTO prepared_replies (conversation_id, case_id, answer, text, claim_updated_at, "
                              "focus_key, party_id, facts) VALUES (%s, 'CL-2048', 'a', 'x', now(), '[]', 'P9', '[]')",
                              [seed.verified_p12])
    await loader_conn.commit()
    assert await _app_count(app_db, seed.verified_p12) == 0
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        await _app_insert(app_db, seed.verified_p12, "P9", "CL-2048")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # nor into another conversation
        await _app_insert(app_db, seed.unverified, "P12")


async def test_the_app_account_cannot_change_a_prepared_reply(app_db, seed):
    await _app_insert(app_db, seed.verified_p12, "P12")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with app_db.conversation(seed.verified_p12) as c:
            await c.execute("UPDATE prepared_replies SET text = 'y'")
