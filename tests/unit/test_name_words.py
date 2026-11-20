"""The words of a name: one rule, the same in the code that decides whole or part and in the database that compares
(db/004_functions.sql, name_words). Lower-cased; apostrophes dropped, so O'Neil is one word; every other character
that is not a letter (space, hyphen, dot, comma, digit) separates words."""

from claims_agent.identity.normalize import is_single_name, is_whole_name, name_candidates, name_words


def test_hyphens_and_dots_separate_words_and_apostrophes_do_not():
    assert name_words("Mary-Jane O'Neil") == ["mary", "jane", "oneil"]
    assert name_words("Margaret.Chen") == name_words("Margaret-Chen") == name_words("margaret, chen") == ["margaret", "chen"]
    assert name_words("MargaretChen") == ["margaretchen"]


def test_a_whole_name_has_two_words_that_are_more_than_an_initial():
    assert is_whole_name("Margaret Chen") and is_whole_name("Margaret-Chen") and is_whole_name("Margaret C. Chen")
    assert not is_whole_name("MargaretChen") and not is_whole_name("Margaret") and not is_whole_name("M. Chen")


def test_a_single_name_is_exactly_one_word_that_is_more_than_an_initial():
    assert is_single_name("Oyelaran") and is_single_name("MargaretChen")
    assert not is_single_name("Margaret Chen") and not is_single_name("M") and not is_single_name("M. Chen")


def test_candidates_keep_the_words_apart_so_the_database_can_count_them():
    assert "ya wen li" in name_candidates("Li Ya-Wen") and name_candidates("O'Neil, José") == ("oneil josé", "josé oneil")
