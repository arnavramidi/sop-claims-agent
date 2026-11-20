import pytest
from helpers import make_form

from claims_agent.form.fields import Pick, YesNo
from claims_agent.memory import Pending
from claims_agent.rules.every_message import _count, global_outcome, usable
from claims_agent.reply.instructions import INSTRUCTIONS
from claims_agent.settings import load_thresholds

T = load_thresholds()


def test_b8_empty_and_noise_are_unreadable():
    assert not usable("", None) and not usable("???", None) and not usable("a", None)


def test_l6_emoji_only_is_unreadable():
    assert not usable("🙂🙂🙂", None)


def test_ruling_12_one_letter_answers_a_pending_offer():
    assert usable("y", Pending(kind="offer_email", label="send the summary?", yes_no=True))
    assert usable("y", Pending(kind="ask_which_claim", label="which claim?", yes_no=False))
    assert not usable("y", None)
    assert not usable("?", Pending(kind="offer_email", label="send the summary?", yes_no=True))


@pytest.mark.parametrize("verified", [False, True])
def test_b7_asking_for_a_person_hands_over_in_every_state(verified):
    assert global_outcome(make_form(wants_person=True), verified=verified, pending=None, t=T).terminal == "handover"


def test_accepting_the_offered_person_hands_over():
    p = Pending(kind="offer_person", label="speak to a person?", yes_no=True)
    assert global_outcome(make_form(pending_reply="yes"), verified=False, pending=p, t=T).terminal == "handover"
    assert global_outcome(make_form(pending_reply="no"), verified=False, pending=p, t=T).terminal is None


def test_yes_to_a_different_offer_does_not_hand_over():
    p = Pending(kind="offer_email", label="send the summary?", yes_no=True)
    assert global_outcome(make_form(pending_reply="yes"), verified=False, pending=p, t=T).terminal is None


def test_done_wraps_up_when_verified_and_closes_when_not():
    assert global_outcome(make_form(done=True), verified=True, pending=None, t=T).terminal == "wrap_up"
    assert global_outcome(make_form(done=True), verified=False, pending=None, t=T).terminal == "close"


def test_v17_another_persons_policy_resets_identity():
    assert global_outcome(make_form(other_person_policy=True), verified=True, pending=None, t=T).terminal == "reset_identity"


def test_off_topic_and_skipping_add_parts_but_do_not_end_the_turn():
    o = global_outcome(make_form(about="off_topic", skips_step=True), verified=False, pending=None, t=T)
    assert o.terminal is None and [p.instruction for p in o.parts] == ["explain_step", "decline_off_topic"]


def test_skipping_and_asking_why_give_one_explanation():
    o = global_outcome(make_form(skips_step=True, asks_why_step=True), verified=False, pending=None, t=T)
    assert [p.instruction for p in o.parts] == ["explain_step"]


def test_asking_why_a_step_is_needed_gets_the_explanation():
    o = global_outcome(make_form(asks_why_step=True), verified=False, pending=None, t=T)
    assert [p.instruction for p in o.parts] == ["explain_step"]


def test_unsupported_insurance_offers_a_person():
    o = global_outcome(make_form(about="unsupported_insurance"), verified=False, pending=None, t=T)
    assert [p.instruction for p in o.parts] == ["unsupported_offer_person"]


def test_a_message_not_in_english_adds_the_offer_in_the_callers_language():
    o = global_outcome(make_form(english=False), verified=False, pending=None, t=T)
    assert "offer_person_in_language" in [p.instruction for p in o.parts]


def test_v16_a_personal_detail_is_never_read_out_before_or_after_verification():
    for verified in (False, True):
        o = global_outcome(make_form(asks_personal_detail=True), verified=verified, pending=None, t=T)
        assert "cannot_read_out_personal_detail" in [p.instruction for p in o.parts]


def test_part_order_is_fixed_and_every_instruction_exists():
    o = global_outcome(make_form(asks_personal_detail=True, english=False, about="off_topic", skips_step=True),
                       verified=False, pending=None, t=T)
    assert [p.instruction for p in o.parts] == ["explain_step", "decline_off_topic",
                                                "cannot_read_out_personal_detail", "offer_person_in_language"]
    assert all(p.instruction in INSTRUCTIONS for p in o.parts)


def test_a_missing_or_refused_field_is_never_read_as_yes():
    r = YesNo(p=None)
    f = make_form(wants_person=r, done=r, other_person_policy=r, skips_step=r, asks_why_step=r,
                  asks_personal_detail=r)
    for verified in (False, True):
        o = global_outcome(f, verified=verified, pending=None, t=T)
        # a refused wants_person is not dropped: it becomes one offer of a person (F1)
        assert o.terminal is None and [x.instruction for x in o.parts] == ["offer_person"]
        assert o.parts[0].pending.kind == "offer_person"
    # N3: with an offer of a person already pending, no new one is added
    p = Pending(kind="offer_person", label="speak to a person?", yes_no=True)
    assert global_outcome(f, verified=False, pending=p, t=T).parts == ()
    # an unconfident pick of pending_reply and of about is not read as yes or as off topic
    f2 = make_form(pending_reply=Pick(value="yes", p=0.3), about=Pick(value="off_topic", p=0.3))
    o = global_outcome(f2, verified=False, pending=p, t=T)
    assert o.terminal is None and o.parts == ()


def test_n1_emoji_only_messages_are_unreadable_even_with_selectors_and_keycaps():
    for m in ("❤️❤️", "☺️☺️", "🙂🙂🙂", "👍🏽👍🏽"):
        assert not usable(m, None)
    assert not usable("1️⃣", None)  # a lone keycap digit is one counted character, so unreadable alone
    assert usable("1️⃣", Pending(kind="ask_which_claim", label="which?", yes_no=False))
    assert usable("1️⃣2️⃣", None)


def test_n2_accepting_the_offered_person_is_a_confident_pick_of_yes():
    p = Pending(kind="offer_person", label="speak to a person?", yes_no=True)
    at_cutoff = global_outcome(make_form(pending_reply=Pick(value="yes", p=0.60)), verified=False, pending=p, t=T)
    assert at_cutoff.terminal == "handover"
    below = global_outcome(make_form(pending_reply=Pick(value="yes", p=0.59)), verified=False, pending=p, t=T)
    assert below.terminal is None and below.parts == ()
    no = global_outcome(make_form(pending_reply=Pick(value="no", p=0.9)), verified=False, pending=p, t=T)
    assert no.terminal is None and no.parts == ()


def test_marks_count_only_after_a_letter_or_digit_not_after_another_mark():
    assert _count("e\u0301\u0302\u0303") == 2  # the letter and its first mark, not the stacked ones
    assert _count("जी") == 2
    assert _count("1\U000E0100\U000E0101") == 1  # second block of variation selectors never counts
    assert _count("\u0301e") == 1  # a mark with no base counts nothing


def test_n3_at_most_one_person_offer_per_turn_and_the_specific_one_wins():
    o = global_outcome(make_form(wants_person=YesNo(p=0.55), about="unsupported_insurance"),
                       verified=False, pending=None, t=T)
    assert [x.instruction for x in o.parts] == ["unsupported_offer_person"]
    assert o.parts[0].pending.kind == "offer_person"
    p = Pending(kind="offer_person", label="speak to a person?", yes_no=True)
    # already pending from an earlier turn: an unsupported request is still explained
    o = global_outcome(make_form(about="unsupported_insurance"), verified=False, pending=p, t=T)
    assert [x.instruction for x in o.parts] == ["unsupported_offer_person"]
    # a weak yes plus an unsupported request: one offer, the specific one
    o = global_outcome(make_form(about="unsupported_insurance", pending_reply=Pick(value="yes", p=0.59)),
                       verified=False, pending=p, t=T)
    assert [x.instruction for x in o.parts] == ["unsupported_offer_person"]
    # an unsure wants_person while an offer is pending gives none, whatever the reply was
    o = global_outcome(make_form(wants_person=YesNo(p=0.55), pending_reply="no"), verified=False, pending=p, t=T)
    assert o.parts == ()


def test_a_refused_english_answer_is_not_treated_as_english():
    o = global_outcome(make_form(english=YesNo(p=None)), verified=False, pending=None, t=T)
    assert [p.instruction for p in o.parts] == ["offer_person_in_language"]


def test_devanagari_two_letter_words_count_as_two():
    assert usable("जी", None) and usable("दो", None)
    assert not usable("ी", None)


@pytest.mark.parametrize("field,terminal,verified", [
    ("wants_person", "handover", False), ("other_person_policy", "reset_identity", True),
    ("done", "wrap_up", True), ("done", "close", False)])
def test_f1_an_ending_outcome_needs_a_confident_yes(field, terminal, verified):
    def run(p):
        return global_outcome(make_form(**{field: YesNo(p=p)}), verified=verified, pending=None, t=T)
    assert run(0.55).terminal is None
    assert run(0.65).terminal == terminal
    assert run(None).terminal is None


def test_f1_an_unsure_request_for_a_person_is_offered_one_with_a_pending():
    for p in (None, 0.55, 0.4):
        o = global_outcome(make_form(wants_person=YesNo(p=p)), verified=False, pending=None, t=T)
        assert o.terminal is None
        offer = [x for x in o.parts if x.instruction == "offer_person"]
        assert len(offer) == 1 and offer[0].pending.kind == "offer_person" and offer[0].pending.yes_no
    for p in (0.0, 0.1, 0.9):
        o = global_outcome(make_form(wants_person=YesNo(p=p)), verified=False, pending=None, t=T)
        assert "offer_person" not in [x.instruction for x in o.parts]


def test_f2_unsupported_offer_is_acceptable_and_the_language_offer_is_not():
    o = global_outcome(make_form(about="unsupported_insurance", english=False), verified=False, pending=None, t=T)
    by = {p.instruction: p for p in o.parts}
    assert by["unsupported_offer_person"].pending.kind == "offer_person"
    assert by["unsupported_offer_person"].pending.yes_no
    assert by["offer_person_in_language"].pending is None
