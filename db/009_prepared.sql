-- Replies to likely follow-up questions, worded ahead in the background after a reply is sent. Each one holds claim
-- wording, so it is claim data at rest: a row belongs to one conversation AND to the customer that conversation was
-- verified as when the row was written. The row rule shows and accepts a row only for the conversation set for the
-- transaction while that conversation is still verified as the same customer, so after a different-person reset, or
-- from another conversation, the rows are invisible even if a delete were missed, and a write made for a customer
-- who is no longer the verified one is refused.
-- `claim_updated_at` is the claim row's last-changed stamp and `facts` the facts the reply was worded from; a row
-- whose stamp, facts or `focus_key` (the claims in focus) no longer match is never used.
CREATE TABLE prepared_replies (
  conversation_id  uuid        NOT NULL,
  case_id          text        NOT NULL,
  answer           text        NOT NULL,
  text             text        NOT NULL,
  claim_updated_at timestamptz NOT NULL,
  focus_key        text        NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  party_id         text        NOT NULL,
  facts            jsonb       NOT NULL,
  PRIMARY KEY (conversation_id, case_id, answer)
);

ALTER TABLE prepared_replies ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON prepared_replies FOR ALL TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid
         AND party_id = (SELECT current_verified_party()))
  WITH CHECK (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid
              AND party_id = (SELECT current_verified_party()));
CREATE POLICY loader_all ON prepared_replies FOR ALL TO claims_loader USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, DELETE ON prepared_replies TO claims_app;
GRANT ALL ON prepared_replies TO claims_loader;
