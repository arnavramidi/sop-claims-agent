from types import UnionType
from typing import Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from claims_agent.memory import (
    ConsentMemory, Counters, FocusMemory, HeldValue, HintMemory, IdentityMemory, Memory, Pending, RequestMemory,
    SectionOwnershipError, WrapupMemory,
)
from claims_agent.states import State


def _sample(annotation):
    """A non-default value for a field of this type, so every field of every section is filled."""
    if annotation is bool:
        return True
    if annotation is int:
        return 5
    if annotation is str:
        return "x"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _filled(annotation)
    if annotation is dict:
        return {"k": "v"}
    args = get_args(annotation)
    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        return _sample(next(a for a in args if a is not type(None)))
    if origin is list:
        return [_sample(args[0])]
    if origin is dict:
        return {"k": _sample(args[1])}
    raise AssertionError(f"full_memory cannot fill {annotation}")


def _filled(model_cls):
    return model_cls(**{n: _sample(f.annotation) for n, f in model_cls.model_fields.items()})


def full_memory(turn_no: int = 7) -> Memory:
    """Every field of every section holds a non-default value (walked from the models, so a field added
    later is filled automatically)."""
    m = Memory(**{n: _filled(f.annotation) for n, f in Memory.model_fields.items() if n != "pending"},
               pending=_filled(Pending))
    m.counters.turn_no = turn_no
    m.counters.stalled_in_a_row = 2
    m.counters.context_floor = 1
    return m


def test_a_state_cannot_write_a_section_it_does_not_own():
    m = Memory.fresh()
    with pytest.raises(SectionOwnershipError):
        m.section("focus", State.VERIFY_ID)
    with pytest.raises(SectionOwnershipError):
        m.section("identity", State.PROCESS_CASE)
    m.section("hints", State.VERIFY_ID).need = "denial_question"
    m.section("focus", State.RESOLVE_INTENT).claim_ids = ["CL-2048"]
    assert m.hints.need == "denial_question" and m.focus.claim_ids == ["CL-2048"]


def test_every_section_has_exactly_its_owner():
    m = Memory.fresh()
    owners = {"identity": State.VERIFY_ID, "consent": State.AWAIT_CONSENT, "focus": State.RESOLVE_INTENT,
              "request": State.PROCESS_CASE, "wrapup": State.POST_PROCESS}
    for name, owner in owners.items():
        assert m.section(name, owner) is getattr(m, name)
        for other in State:
            if other != owner:
                with pytest.raises(SectionOwnershipError):
                    m.section(name, other)
    for name in ("hints", "counters"):
        for st in State:
            assert m.section(name, st) is getattr(m, name)


def test_the_pending_question_and_unknown_names_are_not_sections():
    m = Memory.fresh()
    for name in ("pending", "nothing"):
        with pytest.raises(SectionOwnershipError):
            m.section(name, State.VERIFY_ID)


def test_ruling_4_reset_wipes_everything_that_holds_claim_data():
    m = full_memory(turn_no=7)
    fresh = Memory.fresh()
    for name in ("identity", "consent", "hints", "focus", "request", "wrapup"):
        assert getattr(m, name) != getattr(fresh, name), f"full_memory left {name} at its default"
    assert m.pending is not None
    kept_checks = m.consent.checks_used
    assert kept_checks > 0  # a reset keeps the count: the five-check limit is per conversation
    m.reset_for_new_identity()
    kept = m.identity.wrong_values
    assert kept == ["x"]  # the one thing a reset keeps (KEPT_ACROSS_RESET)
    assert m.identity == IdentityMemory(wrong_values=kept) and m.consent == ConsentMemory(checks_used=kept_checks) and m.hints == HintMemory()
    assert m.focus == FocusMemory() and m.request == RequestMemory() and m.wrapup == WrapupMemory()
    assert m.pending is None
    assert m.counters.context_floor == 7 and m.counters.stalled_in_a_row == 0
    assert m.counters.turn_no == 7


def test_wrong_values_already_counted_survive_a_reset_so_the_three_wrong_values_limit_cannot_be_dodged():
    m = Memory.fresh()
    m.identity.wrong_values = ["dob", "phone"]
    m.reset_for_new_identity()
    m.identity.wrong_values.append("policy")
    assert len(m.identity.wrong_values) == 3
    assert m.identity.held == {} and m.identity.tally == {}


def test_a_step_that_wrote_a_section_it_does_not_own_is_caught_afterwards():
    m = Memory.fresh()
    before = m.snapshot()
    m.focus.claim_ids = ["CL-2048"]  # plain attribute, no section() call
    with pytest.raises(SectionOwnershipError, match="focus"):
        m.assert_only_owned_changed(before, State.VERIFY_ID)
    m.assert_only_owned_changed(before, State.RESOLVE_INTENT)


def test_a_nested_change_inside_a_foreign_section_is_caught():
    m = full_memory()
    before = m.snapshot()
    m.request.answered["k"].append("more")
    with pytest.raises(SectionOwnershipError, match="request"):
        m.assert_only_owned_changed(before, State.POST_PROCESS)


def test_hints_and_counters_may_change_from_any_state_and_pending_is_the_engines():
    for st in State:
        m = Memory.fresh()
        before = m.snapshot()
        m.hints.need = "x"
        m.counters.turn_no += 1
        m.assert_only_owned_changed(before, st)
    m = Memory.fresh()
    before = m.snapshot()
    m.pending = Pending(kind="k", label="l", yes_no=True)
    m.assert_only_owned_changed(before, State.VERIFY_ID)


def test_memory_survives_a_json_round_trip():
    m = full_memory()
    assert Memory.model_validate_json(m.model_dump_json()) == m
    assert Memory.model_validate({}) == Memory.fresh()
