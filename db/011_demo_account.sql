-- The demo account (claims_demo): what the claim-change control writes with, and nothing more. It may change a claim's
-- status and appeal deadline (and its last-changed stamp) and replace a claim's documents, and only the claims of the
-- customer the bound conversation is verified as: its row rules read that conversation the way the server's do. It
-- reads no claim detail, nothing about any customer, and no other table. The server's own account gains nothing.

GRANT USAGE ON SCHEMA public TO claims_demo;
GRANT EXECUTE ON FUNCTION current_verified_party() TO claims_demo;
GRANT SELECT (id, verified_party_id) ON conversations TO claims_demo;
CREATE POLICY demo_bound ON conversations FOR SELECT TO claims_demo
  USING (id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);

GRANT SELECT (case_id, party_id, updated_at), UPDATE (status, appeal_deadline, updated_at) ON claims TO claims_demo;
CREATE POLICY demo_change ON claims FOR ALL TO claims_demo
  USING (party_id = (SELECT current_verified_party())) WITH CHECK (party_id = (SELECT current_verified_party()));

GRANT SELECT (case_id, party_id), INSERT (case_id, party_id, position, name), DELETE ON claim_documents TO claims_demo;
CREATE POLICY demo_change ON claim_documents FOR ALL TO claims_demo
  USING (party_id = (SELECT current_verified_party())) WITH CHECK (party_id = (SELECT current_verified_party()));
