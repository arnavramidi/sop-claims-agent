# Live known failures

Scenarios that did not pass in the latest live run, with what failed, the conversation, and the likely cause. A scenario is never weakened to pass.

## `chinese_message_with_details`

A Chinese message with identity details; the details count, and the reply is English with one Chinese line saying a person is available.

Likely cause: a fault in how requests were written, fixed since and not yet run again. Requests to the models carried non-English text as escape codes, and one escape code for a Chinese character happened to contain the digits of another customer's claim amount, so the test that scans requests for claim data before verification counted it as a leak. No claim data was in the request. Requests now carry text as written (`as_data` in `src/claims_agent/models/call.py`).

What failed:

- turn 1: model request 1 carried allowed_max_amount of CL-4302 before verification
- turn 1: model request 1 carried net_fee of CL-4302 before verification
- turn 1: model request 2 carried allowed_max_amount of CL-4302 before verification
- turn 1: model request 2 carried net_fee of CL-4302 before verification

Conversation:

1. Caller: 你好，我是Margaret Chen，我的出生日期是1985年3月15日，社保号码后四位是4472。
   Agent: You’re verified. What would you like help with?  您随时可以要求与人工客服交谈。
   (state RESOLVE_INTENT, verified P9, claims in focus [], reply parts ['verified', 'ask_need', 'offer_person_in_language'], reply live)

