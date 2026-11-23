"""One instruction per kind of reply part: what the wording model is asked to do with the facts it is given."""

# Where a summary goes is named only by the phrase a fact gives; no instruction has a destination phrase of its own.
DESTINATION_RULE = "refer to where it goes only by the phrase given in the facts, and never state or guess an email address"

MISSING = ("addresses", "links", "receipt status", "appeal steps", "amounts owed")

# Before verification: why claim details wait, said without naming or confirming any claim.
VERIFY_FIRST = (" If the caller asked about a claim, say first, in one sentence, that claim details are protected and "
                "can only be shared once the caller is verified. Never mention a particular claim, and never say whether "
                "a claim or a policy exists.")

# Every reply that asks for identity details says the rule and offers every kind still open, so the caller can choose
# and give several at once (code checks both: reply.check.required_mentions).
ASK_DETAILS = ("Say the verification rule from the facts (how many details must match the policy), then list EVERY "
               "detail the facts name as not given yet, as options, in the words the facts use; say the caller may give "
               "any of them, several at once. Never say how many details matched or how many more are needed. If a "
               "fact says a detail was kept from this message, thank the caller for it; otherwise thank them for no "
               "detail.")

_USE_ONLY = "Use only the facts given; never add a number, date, claim number or status of your own."

INSTRUCTIONS: dict[str, str] = {
    "ask_again": "Ask the caller again, in other words, for the detail that was just asked for. Keep it short.",
    "ask_identity": "Ask the caller for details to verify who they are. " + ASK_DETAILS + VERIFY_FIRST,
    "ask_another_detail": "Ask the caller for more details to verify who they are. " + ASK_DETAILS + VERIFY_FIRST,
    "unreadable_detail": "Say plainly that the detail in the caller's message could not be read, and ask the caller to give it again. Do not thank the caller for it.",
    "ask_full_value": "The caller gave only part of a detail. Ask for the whole value.",
    "restate_value": "Say plainly that the detail could not be read or used, naming only its KIND as the facts do (for example: phone number, date of birth), and ask the caller to give it again. Never repeat, read back or quote the value itself, and do not thank the caller for it.",
    "explain_step": "Explain briefly, in your own words, why this step is needed, giving the reason the facts give (it protects the caller's information), then ask for the detail again.",
    "decline_off_topic": "Politely say you can only help with claims questions here. Ask no question in this part: the rest of the reply brings the caller back.",
    "unsupported_offer_person": "Say you cannot help with that request here, then ask one clear yes-or-no question: would the caller like you to pass them to a person?",
    "offer_person": "Ask one clear yes-or-no question: would the caller like you to pass them to a person?",
    "handover": "Tell the caller you are handing them to a person and that they will help from here. Do not add details.",
    "closed": "Tell the caller this conversation has ended, politely and briefly.",
    "final_notice": "Tell the caller this conversation has already ended or been passed to a person, so nothing more can be done in it, and that they can start a new conversation.",
    "cannot_read_out_personal_detail": "Say you cannot read out personal details held on file. Offer another way to help.",
    "verified": "Tell the caller they are verified. Do not ask a question.",
    "consent_requested": "Say the policyholder's consent is needed before you can discuss their claims and, in one sentence, why (the facts say), that it has been requested, and ask the caller to say when the policyholder has replied.",
    "consent_pending": "Say consent from the policyholder has not come through yet, and ask the caller to check back.",
    "consent_denied": "Say the policyholder has not given consent, so you cannot discuss their claims.",
    "not_listed": "Say the name the caller gave is not listed as someone who can act for the policyholder.",
    "no_claims_on_file": "Say there are no claims on file for this policy.",
    "claim_not_on_policy": "Say the claim the caller named is not on their policy. Do not repeat the claim number or any other value the caller gave.",
    "ask_same_policyholder": "Ask the caller, as one yes or no question, whether this is still about the policyholder they are calling for. Name no one and say nothing about any claim.",
    "found_claim": "Tell the caller which claim you found, using only the fact given. Add nothing about the claim.",
    "which_claim": "Ask which claim the caller means, listing the claims given in the facts with every detail each fact gives: those details are what tells the claims apart.",
    "nearest_matches": "Say you found no claim matching the description, then list the claims given in the facts with every detail each fact gives. Never repeat what the caller described.",
    "overview": "If a fact says no claim matched the description, say that first. Give a short overview of the claims in the facts. Offer to narrow the list down, or to include the full list in the summary email at the end; if you say where it would go, " + DESTINATION_RULE + ".",
    "narrow_down": "If a fact says no claim matched the description, say that first. There are too many claims to list. Ask the caller to narrow it down by the detail the facts name as most useful.",
    "ask_need": "Ask what the caller would like help with. If the facts name possible needs, ask which of those the caller means.",
    "ask_which_answer": "Ask which of the answers the caller wants to hear, listing the options given.",
    "ask_which_document": "Ask which of the requested documents in the facts the caller cannot get, listing them by name.",
    "ask_which_figure": "Ask which of the money figures in the facts the caller means, naming each in the words the facts use.",
    "answer": "Answer the caller's question using the facts given, and say every one of them: each was chosen as part "
              "of the answer, so leave none out, even when the caller asked only part of it. " + _USE_ONLY,
    "not_on_file": "Say that information is not on file for this claim. Do not guess.",
    "offer_followup": "Offer the follow-up topic given in the facts, in one sentence.",
    "anything_else": "Ask whether there is anything else you can help with.",
    "ask_which_of_these": "Ask, as one question, which of the claims just described the caller would like to go into, or what they would like to know about them.",
    "offer_email": "Offer to email the caller a summary of what was discussed; if you say where it would go, " + DESTINATION_RULE + ".",
    "email_sent": "Tell the caller the summary email has been sent; " + DESTINATION_RULE + ".",
    "email_skipped": "Say you will not send an email, then say goodbye briefly; ask nothing. If you mention where it would have gone, " + DESTINATION_RULE + ".",
    "email_contents": "Tell the caller, in plain words, what the email will contain, using the facts given; if you mention where it goes, " + DESTINATION_RULE + ".",
    "email_address_not_read_out": "Say the email address cannot be read out, but the summary will reach the caller; " + DESTINATION_RULE + ".",
    "different_address_not_possible": "Say the summary can only be sent to the place the facts name; " + DESTINATION_RULE + ".",
    "goodbye": "Say goodbye briefly and politely.",
    "model_trouble": "Apologise for a technical problem and ask the caller to try again.",
    "offer_person_in_language": "Keep the whole reply in English, and add ONE extra line in the caller's own language saying that the caller can ask for a person at any time. State it as a plain fact; do not ask a question and do not offer a choice.",
    "acknowledge_refusal": "The caller would rather not give one or more details. Thank them for saying so and, if the facts name what they would rather not give, say you will not ask for it again (never ask for it). Say which of the other details in the facts would do instead, naming every one of them, and say plainly that without the number of matching details the facts give, nothing about a claim can be shared. If a fact says a person can help instead, say so as a plain statement, not a question.",
    "ask_caller_name": "Ask the caller for their own full name, saying in one short clause why (the facts say). Ask nothing else in this part.",
    "offer_person_before_closing": "Say you understand if the caller wants to stop here, then ask one clear yes-or-no question: would the caller like you to pass them to a person before the conversation ends?",
    "more_questions_later": "Say you have answered the first of the caller's questions, and ask them to send the rest in their next message. Do not repeat or guess the questions left.",
    "email_already_sent": "Say the summary email was already sent and nothing new has been discussed since, then say goodbye briefly; " + DESTINATION_RULE + ".",
    "email_at_wrap_up": "Acknowledge the request for a summary email in one sentence, saying what the facts say will happen when the conversation ends; ask nothing in this part. If you say where it would go, " + DESTINATION_RULE + ".",
}
