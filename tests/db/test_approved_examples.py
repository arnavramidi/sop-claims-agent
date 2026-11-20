"""Approved example phrasings at app start and in every turn's form request."""

import json
import logging

from helpers import make_deps, new_conversation, say

from claims_agent import examples
from claims_agent.examples import startup_examples
from claims_agent.settings import Examples, load_settings


def settings_with(path, **limits):
    return load_settings().model_copy(update={"examples": Examples(path=str(path), **limits)})


def write_json(tmp_path, rows):
    path = tmp_path / "approved_examples.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


async def test_approved_examples_follow_the_datas_own_in_a_turns_form_request(load_dataset):
    deps = make_deps(await load_dataset("starter"),
                     examples={"answer_wanted:submission_timing": ("what's my cutoff for sending it",)})
    await say(deps, await new_conversation(deps), "hello")
    shown = deps.filler.requests[-1].examples["answer_wanted:submission_timing"]
    assert shown[-1] == "what's my cutoff for sending it" and "when do i need to send" in shown[:-1]


async def test_with_no_approved_examples_a_turns_request_carries_only_the_datas_own(load_dataset):
    deps = make_deps(await load_dataset("starter"))
    await say(deps, await new_conversation(deps), "hello")
    shown = deps.filler.requests[-1].examples
    assert shown["answer_wanted:submission_timing"][-1] == "when should i send"
    assert all(k.startswith("answer_wanted:") for k in shown)


async def test_at_start_only_fields_and_options_the_form_asks_are_kept(load_dataset, tmp_path):
    db = await load_dataset("starter")
    path = write_json(tmp_path, [
        {"field": "answer_wanted", "option": "submission_timing", "message": "what's my cutoff for sending it"},
        {"field": "answer_wanted", "option": "no_such_topic", "message": "what now"},
        {"field": "need", "option": "next_steps", "message": "so what happens now"},
        {"field": "mood", "option": "furious", "message": "ugh"},
        {"field": "relationship", "option": "son", "message": "I'm her son"},
        {"field": "email_offer_reply", "option": "send", "message": "go ahead and mail it"},
        {"field": "listed_claim", "option": "none", "message": "neither of those"},
        {"field": "wants_person", "option": "yes", "message": "a human please"},
    ])
    got = await startup_examples(settings_with(path), None, db.app)
    assert got == {"answer_wanted:submission_timing": ("what's my cutoff for sending it",),
                   "need:next_steps": ("so what happens now",), "relationship:son": ("I'm her son",),
                   "email_offer_reply:send": ("go ahead and mail it",)}


async def test_a_failing_remote_at_start_leaves_the_files_examples(load_dataset, tmp_path, monkeypatch):
    def down(project, dataset="approved-examples"):
        raise TimeoutError("no answer")
    monkeypatch.setattr(examples, "load_remote", down)
    db = await load_dataset("starter")
    path = write_json(tmp_path, [{"field": "need", "option": "next_steps", "message": "so what happens now"}])
    assert await startup_examples(settings_with(path), "key", db.app) == {"need:next_steps": ("so what happens now",)}


async def test_a_database_problem_at_start_means_no_examples_and_the_app_still_starts(tmp_path, caplog):
    class Down:
        def conversation(self, _):
            raise OSError("database unreachable")

    path = write_json(tmp_path, [{"field": "need", "option": "next_steps", "message": "so what happens now"}])
    with caplog.at_level(logging.WARNING, logger="claims_agent.examples"):
        assert await startup_examples(settings_with(path), None, Down()) == {}
    assert "app starts without them" in caplog.text
