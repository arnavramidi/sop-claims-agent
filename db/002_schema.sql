-- Tables, with the starter's own field names. Runs as claims_owner, so the owner owns every object.
-- Every surrogate id is GENERATED ALWAYS AS IDENTITY, so an insert needs no privilege on a sequence.

CREATE TABLE customers (
  party_id      text PRIMARY KEY,
  name          text,
  policy_number text,
  dob           date,
  id_type       text,
  id_last4      text,
  phone         text,
  email         text
);

CREATE TABLE customer_aliases (
  party_id text NOT NULL,
  kind     text NOT NULL CHECK (kind IN ('name', 'phone', 'email')),
  value    text NOT NULL
);

CREATE TABLE representatives (
  party_id     text NOT NULL,
  rep_name     text NOT NULL,
  relationship text
);

CREATE TABLE claims (
  case_id                       text PRIMARY KEY,
  party_id                      text NOT NULL,
  case_type                     text,
  created_at                    date,
  status                        text NOT NULL,
  summary                       text,
  denial_reason                 text,
  appeal_deadline               date,
  expected_reimbursement_amount numeric(12,2),
  allowed_max_amount            numeric(12,2),
  net_pay                       numeric(12,2),
  net_fee                       numeric(12,2),
  updated_at                    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX claims_party_status_type_created ON claims (party_id, status, case_type, created_at);

-- The last-changed stamp a prepared reply is checked against: moves on every insert and update.
CREATE FUNCTION claims_touch_updated_at() RETURNS trigger LANGUAGE plpgsql
  SET search_path = pg_catalog, public, pg_temp AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END $$;

CREATE TRIGGER claims_updated_at BEFORE INSERT OR UPDATE ON claims
  FOR EACH ROW EXECUTE FUNCTION claims_touch_updated_at();

CREATE TABLE claim_documents (
  case_id  text NOT NULL,
  party_id text NOT NULL,
  position int  NOT NULL,
  name     text NOT NULL,
  PRIMARY KEY (case_id, position)
);

CREATE TABLE guidance (
  kind text  NOT NULL,
  key  text,
  body jsonb NOT NULL
);

CREATE TABLE consent_scripts (
  name            text PRIMARY KEY,
  status_sequence text[] NOT NULL
);

CREATE TABLE vocabulary (
  kind  text NOT NULL,
  value text NOT NULL
);

CREATE TABLE conversations (
  id                uuid PRIMARY KEY,
  state             text NOT NULL,
  verified_party_id text,
  pending_party_id  text,
  consent_script    text NOT NULL DEFAULT 'default',
  today             date,
  memory            jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE turns (
  id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  conversation_id    uuid NOT NULL,
  turn_no            int  NOT NULL,
  message            text,
  reply              text,
  state_before       text,
  state_after        text,
  record             jsonb,
  braintrust_span_id text,
  created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE escalations (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  conversation_id uuid NOT NULL,
  turn_no         int,
  kind            text NOT NULL,
  reason          text,
  note            jsonb,
  created_at      timestamptz NOT NULL DEFAULT now()
);

-- The recipient is filled in by a database function at the moment of the simulated send;
-- the app account cannot read it.
CREATE TABLE outbox (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  conversation_id uuid NOT NULL,
  party_id        text NOT NULL,
  recipient       text,
  subject         text,
  body            text,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE consent_requests (
  conversation_id uuid PRIMARY KEY,
  party_id        text NOT NULL,
  script          text NOT NULL,
  checks_used     int  NOT NULL DEFAULT 0,
  status          text NOT NULL
);

CREATE TABLE feedback (
  id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  turn_id    bigint NOT NULL,
  rating     text NOT NULL,
  comment    text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE load_report (
  id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  level  text NOT NULL,
  file   text,
  ref    text,
  field  text,
  reason text
);
