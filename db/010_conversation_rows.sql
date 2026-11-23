-- Every per-conversation table follows the rule events and prepared replies follow: the server's account sees and
-- changes only the rows of the conversation its transaction is bound to (the transaction-local setting
-- app.conversation_id), and an unbound transaction sees none. Creating a conversation binds the transaction to the new
-- id first (conversations.create), so even a new row is written under the rule. The owner-rights functions in
-- 004_functions.sql run as the owner, whom these rules do not bind, and each of them acts only on the bound
-- conversation itself (bound_conversation). current_verified_party() reads the bound row, which the rule allows.
-- The loader keeps every power it had (setup and test fixtures only).

-- A rating belongs to the conversation of the turn it rates, so the rule can bind it like every other row.
ALTER TABLE feedback ADD COLUMN conversation_id uuid NOT NULL;

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON conversations FOR ALL TO claims_app
  USING (id = NULLIF(current_setting('app.conversation_id', true), '')::uuid)
  WITH CHECK (id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON conversations FOR ALL TO claims_loader USING (true) WITH CHECK (true);

ALTER TABLE turns ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON turns FOR ALL TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid)
  WITH CHECK (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON turns FOR ALL TO claims_loader USING (true) WITH CHECK (true);

ALTER TABLE escalations ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON escalations FOR ALL TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid)
  WITH CHECK (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON escalations FOR ALL TO claims_loader USING (true) WITH CHECK (true);

ALTER TABLE outbox ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON outbox FOR ALL TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid)
  WITH CHECK (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON outbox FOR ALL TO claims_loader USING (true) WITH CHECK (true);

ALTER TABLE consent_requests ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON consent_requests FOR ALL TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid)
  WITH CHECK (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON consent_requests FOR ALL TO claims_loader USING (true) WITH CHECK (true);

ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON feedback FOR ALL TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid)
  WITH CHECK (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON feedback FOR ALL TO claims_loader USING (true) WITH CHECK (true);
