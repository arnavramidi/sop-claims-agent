# Insurance claims support agent with an SOP harness

A chat agent for insurance claim support that follows a fixed operating procedure:
verify the caller, work out what they need, answer from the claim record, then wrap up.
Language models read the caller's messages and word the replies; plain code makes every
decision about state, access and what a reply is allowed to say.

## Run it

You need Docker and an OpenAI API key.

```bash
cp .env.example .env        # then set OPENAI_API_KEY in .env
docker compose up --build
# open http://localhost:8000
```

The first start builds the image, creates the database, loads the sample data in
`data/starter/` and starts the server. The page has the chat on the left and an
operator panel on the right that shows what was decided on each turn and by what
(code, the fast model, the text model, or the larger model). A caller would never see
that panel.

## Try this

1. The full flow in one message:

   > I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about
   > my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

   The agent verifies her (three or more of five identity details match one customer),
   finds the one claim that fits, and answers from the record: status, denial reason,
   documents needed, appeal deadline. Then ask "what if I can't get the pathology
   report?", then "that's all", and accept the summary email.

2. A frustrated caller who has not verified:

   > I already told you who I am. This is ridiculous. Just tell me why my claim was denied.

   The agent acknowledges the frustration, explains that claim details are protected,
   says which details it can accept, and discloses nothing.

3. Out of scope: "what is reinforcement learning?" is declined politely. Ask for a
   person at any point and the agent offers a hand-over.

## How a turn works

```
caller message
   -> a fast model fills in a fixed form about the message   (yes/no and pick-one questions, each with a confidence)
   -> a text model pulls out stated values                   (name, date of birth, policy number, ...)
   -> code reads the form and decides                        (state changes, which claim, which facts may be said)
   -> a text model words a reply from those facts only
   -> code checks the reply against the facts before sending (a failed reply is rewritten once, then replaced by a fixed text)
```

No model ever decides to move the conversation forward, and no model is given anything
about a claim until the database has verified the caller.

### The states

| State | What happens | Freedom |
|---|---|---|
| `VERIFY_ID` | The caller must give at least three of: full name, date of birth, policy number, phone, last four ID digits. The database does the matching. Three wrong values hand over to a person. | Strict: fixed facts only, nothing about any claim. |
| `AWAIT_CONSENT` | Someone calling for a policyholder waits here until the policyholder's consent is recorded. | Strict. |
| `RESOLVE_INTENT` | Works out which claim the caller means (type, status, month, year, a claim number, or a pick from a short list) and what they need. | Free: the models interpret messy language; code turns it into a filter. |
| `PROCESS_CASE` | Answers from the claim's own row and the guidance data, offers one follow-up at a time, gives alternatives when a document can't be obtained, and hands over when the alternatives run out. | Guided: answers are built by code; wording is free. |
| `POST_PROCESS` | "Anything else?", an optional summary email to the address on file, close. | Guided. |

Rules that apply to every message, in any state: a request for a person, abuse, an
unreadable message, an off-topic question, and a caller who is going in circles.

### How the database guards claims

The server connects with an account that cannot read claims at all unless the
conversation has been verified. Verification is a database function: it compares the
details the caller gave with the customer table and, only on a match, records the
verified customer on the conversation. A row-level rule on the claims tables then
returns only that customer's rows to that conversation. So "show me someone else's
claim" fails in the database, not just in the prompt, and a claim number that belongs
to another customer looks exactly like one that does not exist.

The email address used for the summary never leaves the database. The agent refers to
it only as "the email on your policy".

### Models

| Job | Model |
|---|---|
| Fill the form (OpenAI Decisions API, about 40 questions in one request) | `gpt-6-luna` |
| Pull out stated values; word the reply | `gpt-6.1-sol` |
| Rewrite a reply that failed the check | `gpt-6-astra` |

Measured on the test message above: about 0.4 cents and four to six seconds per caller
message. Model names and thresholds are in `settings.toml` and `thresholds.toml`.

### Why no agent framework

The procedure is a small state machine, and the important property is that code, not a
model, owns every transition. Each state is one function that takes the filled form and
returns the next state and the facts for the reply, which maps directly onto a graph of
nodes and conditional edges if you wanted to move it into a framework later. Keeping it
as plain functions made the access rules easy to test.

## Other data

Point `data.folder` in `settings.toml` at another folder with the same six JSON files.
The loader checks every row, sets aside what it cannot store, and reports it in the
operator panel instead of failing.

## Tests

```bash
scripts/test.sh              # everything, inside Docker, against a real Postgres
scripts/test.sh tests/unit   # one folder or file
```

About 700 tests: unit tests for the form, identity matching, the reply check and the
loader, and whole-conversation tests that run each state against a real database with
scripted models. The security tests connect as the server's own account and try to read
another customer's claims.

## Known limits

- OpenAI only. The form filler needs a key with access to the Decisions API.
- The operator panel shows what matched during verification. It is a debugging view for
  this demo and would not be exposed to callers.
- A third party who knows three of a policyholder's details and does not say they are
  calling for someone else is verified as the policyholder. That is the three-of-five
  rule as specified.
- The reply check is pattern based (numbers, dates, amounts, claim numbers, status
  words). It catches invented figures and status claims, not every possible misstatement.
- The identity check has no per-caller rate limit; a real deployment needs one.
- Personal details are logged as typed because only sample data is used. Production
  would mask them where turn records are written.
- One message that asks several things is answered for its main question first.
