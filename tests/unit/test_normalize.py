from datetime import date

from claims_agent.identity.normalize import claim_number_key, normalise
from helpers import stated


def test_both_date_readings_are_kept():
    s, bad = normalise(stated(date_of_birth=["1985-03-04", "1985-04-03"])); assert len(s.dobs) == 2 and bad == []


def test_impossible_date_and_five_id_digits_fail_the_format_check():
    s, bad = normalise(stated(date_of_birth=["1985-02-30"], id_digits="12345"))
    assert s.dobs == () and s.id_last4 is None and {f.kind for f in bad} == {"dob", "id"}


def test_full_nine_digit_id_keeps_the_last_four():
    assert normalise(stated(id_digits="123-45-4472"))[0].id_last4 == "4472"


def test_partials_do_not_count():
    s, bad = normalise(stated(phone="650")); assert s.phone is None and bad[0].kind == "phone"


def test_claim_number_key(): assert claim_number_key("cl 2048") == claim_number_key("CL-2048") == "CL2048"


# Extra checks beyond the core cases.


def test_a_name_is_passed_on_as_stated_for_the_database_to_read():
    s, bad = normalise(stated(policyholder_name="  O'Neil, José  "))
    assert s.name == "O'Neil, José" and bad == []


def test_a_real_reading_beside_an_impossible_one_is_no_format_failure():
    s, bad = normalise(stated(date_of_birth=["1985-02-03", "1985-03-02x"]))
    assert s.dobs == (date(1985, 2, 3),) and bad == []


def test_phone_email_policy_and_id_are_normalised():
    s, bad = normalise(stated(phone="+1 (650) 521-2830", email="  Yawen.Li@Example.COM ",
                              policy_number="pol 9921", id_digits="4472"))
    assert (s.phone, s.email, s.policy, s.id_last4, bad) == ("16505212830", "yawen.li@example.com", "POL9921", "4472", [])


def test_bad_email_and_lettered_id_fail_the_format_check():
    s, bad = normalise(stated(email="margaret at email", id_digits="44A2"))
    assert s.email is None and s.id_last4 is None and {f.kind for f in bad} == {"email", "id"}
    assert all(f.reason for f in bad) and {f.as_stated for f in bad} == {"margaret at email", "44A2"}


def test_nothing_stated_is_nothing_matched_and_no_failure():
    s, bad = normalise(stated(policyholder_name=" ", phone=""))
    assert s.name is None and s.phone is None and bad == []
