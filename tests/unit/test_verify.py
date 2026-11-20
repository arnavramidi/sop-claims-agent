from claims_agent.identity.types import CustomerMatch
from claims_agent.identity.verify import judge


def m(party, kinds="", policy=False):
    return CustomerMatch(party, frozenset(kinds.split()), policy)


def h(s):
    return frozenset(s.split())


def test_brief_case_verifies():
    d = judge(h("name dob id policy"), [m("P9", "name dob id", True)])
    assert d.verified_party == "P9" and d.wrong_now == frozenset()


def test_v2_three_people_one_kind_each():
    d = judge(h("name dob id"), [m("A", "name"), m("B", "dob"), m("C", "id")])
    assert (d.candidate, d.verified_party, d.wrong_now) == (None, None, frozenset())


def test_v4_policy_plus_two_kinds_is_not_enough():
    assert judge(h("name dob policy"), [m("P9", "name dob", True)]).verified_party is None
    assert judge(h("name dob id policy"), [m("P9", "name dob id", True)]).verified_party == "P9"


def test_v5_someone_elses_policy_number_is_the_one_wrong_value():
    d = judge(h("name dob id policy"), [m("A", "", True), m("B", "name dob id")])
    assert (d.candidate, d.verified_party, d.wrong_now) == ("B", None, frozenset({"policy"}))


def test_v21_verified_on_three_even_if_a_fourth_is_wrong():
    d = judge(h("name dob id phone"), [m("P9", "name dob id"), m("P13", "phone")])
    assert d.verified_party == "P9" and d.wrong_now == frozenset({"phone"})


def test_v24_all_five_given_nobody_at_three():
    d = judge(h("name dob phone email id"), [m("A", "name dob"), m("B", "phone email"), m("C", "id")])
    assert d.verified_party is None and d.all_kinds_no_match


def test_v25_the_odd_value_is_wrong_whatever_the_order():
    first = judge(h("phone"), [m("C", "phone")])
    assert first.wrong_now == frozenset()
    tie = judge(h("phone name"), [m("C", "phone"), m("M", "name")])
    assert tie.candidate is None and tie.wrong_now == frozenset()
    later = judge(h("phone name dob"), [m("C", "phone"), m("M", "name dob")])
    assert later.wrong_now == frozenset({"phone"})


def test_a_value_matching_nobody_is_wrong_with_no_candidate():
    assert judge(h("email"), []).wrong_now == frozenset({"email"})


def test_policy_alone_names_the_candidate_without_verifying():
    d = judge(h("policy"), [m("P9", "", True)])
    assert (d.candidate, d.verified_party, d.wrong_now) == ("P9", None, frozenset())


def test_a_tie_is_broken_by_the_policy_number():
    assert judge(h("name phone policy"), [m("A", "name", True), m("B", "phone")]).candidate == "A"


def test_a_policy_number_matching_nobody_is_wrong_with_no_candidate():
    assert judge(h("policy"), []).wrong_now == frozenset({"policy"})


def test_all_kinds_no_match_needs_all_five_held():
    assert not judge(h("name dob phone email"), [m("A", "name dob")]).all_kinds_no_match


def test_policy_never_counts_as_a_kind():
    d = judge(h("name dob policy"), [m("P9", "name dob policy", True)])
    assert d.verified_party is None and d.wrong_now == frozenset()


def test_an_unknown_kind_never_counts():
    d = judge(h("name dob nickname"), [m("P9", "name dob nickname")])
    assert d.verified_party is None


def test_a_repeated_party_is_refused():
    import pytest

    with pytest.raises(ValueError):
        judge(h("name"), [m("A", "name"), m("A", "dob")])


def test_all_five_held_and_an_unbreakable_tie_is_no_match():
    d = judge(h("name dob phone email id"), [m("A", "name phone email dob"), m("B", "name phone email id")])
    assert (d.candidate, d.verified_party, d.wrong_now) == (None, None, frozenset())
    assert d.all_kinds_no_match
