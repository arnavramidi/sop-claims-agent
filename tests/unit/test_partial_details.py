"""A value that is only part of a detail is told apart from one that is malformed, by its shape alone: some digits of
a phone number or ID, a date with no whole day. A part is asked for in full and never held, so it can never count as a
wrong value. (A name's shape is read by the database: tests/db/test_name_rule_scripts.py.)"""

from claims_agent.identity.normalize import normalise
from helpers import stated


def _failure(**values):
    _, bad = normalise(stated(**values))
    assert len(bad) == 1
    return bad[0]


def test_some_digits_of_a_phone_number_are_part_of_it():
    assert _failure(phone="650").partial


def test_too_few_or_too_many_id_digits_are_part_of_the_number():
    assert _failure(id_digits="44").partial and _failure(id_digits="12345").partial


def test_a_date_with_no_whole_day_is_part_of_a_date():
    assert _failure(date_of_birth_as_stated="1985").partial


def test_a_malformed_value_is_not_a_part():
    assert not _failure(email="margaret at email").partial
    assert not _failure(id_digits="44A2").partial
    assert not _failure(date_of_birth=["1985-02-30"]).partial


def test_a_complete_value_is_no_failure_at_all():
    _, bad = normalise(stated(id_digits="4472", phone="+1 650 521 2836", date_of_birth=["1985-03-15"]))
    assert bad == []
