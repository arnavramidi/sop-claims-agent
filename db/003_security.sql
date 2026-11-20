-- Who may see and change what. Runs as claims_owner.
-- claims_owner owns everything and has no policy on claims, so with forced rules it reads zero claim rows.
-- claims_loader may write every row (setup and test fixtures only; never given to the server).
-- claims_app is the server's account.

GRANT USAGE ON SCHEMA public TO claims_app, claims_loader;   -- a recreated schema grants nothing to anyone else

-- The customer a conversation is verified as, read from the conversation's own row.
-- With no conversation set for this transaction it returns NULL, which matches no claim.
-- The search path is pinned with the temporary schema last and the table is named with its schema, so a
-- temporary table called `conversations` can never stand in for the real one.
-- Like every function the owner creates, it is closed to PUBLIC (migrate.apply sets that default);
-- the server's account is granted it because its policies call it.
CREATE FUNCTION current_verified_party() RETURNS text LANGUAGE sql STABLE
  SET search_path = pg_catalog, public, pg_temp AS $$
  SELECT verified_party_id FROM public.conversations
  WHERE id = NULLIF(current_setting('app.conversation_id', true), '')::uuid $$;
GRANT EXECUTE ON FUNCTION current_verified_party() TO claims_app;

ALTER TABLE claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE claims FORCE ROW LEVEL SECURITY;
CREATE POLICY app_read ON claims FOR SELECT TO claims_app USING (party_id = (SELECT current_verified_party()));
CREATE POLICY loader_all ON claims FOR ALL TO claims_loader USING (true) WITH CHECK (true);

ALTER TABLE claim_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE claim_documents FORCE ROW LEVEL SECURITY;
CREATE POLICY app_read ON claim_documents FOR SELECT TO claims_app USING (party_id = (SELECT current_verified_party()));
CREATE POLICY loader_all ON claim_documents FOR ALL TO claims_loader USING (true) WITH CHECK (true);

-- claims_app: nothing at all on customers, customer_aliases, representatives.
GRANT SELECT ON claims, claim_documents, guidance, consent_scripts, vocabulary, load_report, consent_requests, conversations
  TO claims_app;
GRANT INSERT (id, state, memory, consent_script, today) ON conversations TO claims_app;
GRANT UPDATE (state, memory, consent_script, today, updated_at) ON conversations TO claims_app;
GRANT SELECT, INSERT ON turns, escalations, feedback TO claims_app;
GRANT SELECT (id, conversation_id, party_id, subject, body, created_at) ON outbox TO claims_app;

-- claims_loader empties and refills every table at startup.
GRANT ALL ON ALL TABLES IN SCHEMA public TO claims_loader;
