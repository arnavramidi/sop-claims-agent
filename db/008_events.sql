-- What happened inside each turn: what each model was asked and answered, what each rule decided.
-- Written by the server in the turn's own transaction. The app account may add events and change an event's body
-- (a later turn marks an earlier extraction as corrected); it cannot change anything else about an event or delete
-- one. Each conversation sees only its own events, the way the claims rule works.
CREATE TABLE events (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  conversation_id uuid NOT NULL,
  turn_no         int  NOT NULL,
  kind            text NOT NULL,
  body            jsonb NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX events_conversation_turn ON events (conversation_id, turn_no, id);

ALTER TABLE events ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_own ON events FOR ALL TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid)
  WITH CHECK (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON events FOR ALL TO claims_loader USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE (body) ON events TO claims_app;
GRANT ALL ON events TO claims_loader;
