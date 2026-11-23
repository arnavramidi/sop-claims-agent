-- The wrong-value limit, kept by the database. Runs as claims_owner.
--
-- A caller is verified when three of five details match one customer (004_functions.sql). Every value that is
-- compared and judged wrong is counted here, per conversation, by the same function that compares it, and once a
-- conversation has counted as many wrong values as the limit, nothing more is compared for it. So the server cannot
-- skip the count, forget it or clear it, and calling the functions directly changes nothing. check_identity is the
-- one check: the server calls it to compare, and bind_policyholder and request_consent compare through it, so all
-- three refuse a conversation at its limit, with one error (SQLSTATE IL001) that cannot be read as "no match".
-- The count is written in the caller's own transaction: a transaction that is rolled back takes its count with it
-- (the engine hands such a turn over to a person; a server that rolls back on purpose is not stopped by this).
--
-- The call that reaches the limit still answers (its wrong values are counted and kept), and values that counted a
-- wrong value in this same transaction may be compared again in it, exactly as they were, so the turn that gives the
-- third wrong value can still act on what it found (a caller who also gave three right details is verified, as
-- before). They reveal nothing new: the answer is the one already given. Any other values, or a later transaction,
-- are refused.
--
-- What counts as wrong is the server's own rule (identity/verify.py, judge), made here on what the comparison found:
--   * a candidate is the one customer with the most matching kinds, or, among those tied, the only one whose policy
--     number matches; with no kind matching anyone, the only customer whose policy number matches;
--   * with a candidate: each held kind the candidate does not match is wrong, and a policy number that is not theirs;
--   * with no candidate: each held kind that matches nobody is wrong, and a policy number that matches nobody.
-- A held value is one the comparison reads: a whole name (or a one-word name), a date of birth, a phone of ten or more
-- digits, a well-formed email, four ID digits, a policy number. Only those are passed on to be compared, so no value
-- is compared without being judged, and counted when wrong.
--
-- A wrong value is kept as a fingerprint: its kind and a digest of the normalised value salted with the conversation
-- id, never the value. The same value said again is the same fingerprint and counts once; a corrected value is a
-- different value.

-- The limit: settings.toml [limits] wrong_values, fixed into the schema when it is built (migrate.apply passes it as
-- claims.wrong_values_limit, and the build stamp includes it). The server's account cannot change it, and a database
-- built with another limit is not kept by a start without a rebuild.
DO $$
DECLARE
  v_limit int := current_setting('claims.wrong_values_limit')::int;
BEGIN
  IF v_limit IS NULL OR v_limit < 1 THEN
    RAISE EXCEPTION 'limits.wrong_values must be a whole number above 0';
  END IF;
  EXECUTE format($f$
    CREATE FUNCTION wrong_values_limit() RETURNS int
      LANGUAGE sql IMMUTABLE PARALLEL SAFE
      SET search_path = pg_catalog, public, pg_temp
      AS 'SELECT %s'
  $f$, v_limit);
END $$;

-- The wrong values counted for each conversation. Only check_identity writes here. `counted_with` is a digest of every
-- value compared in the call that counted the row, and `counted_in` that call's transaction: together they let that
-- same call be repeated in the same transaction once the limit is reached, and nothing else.
CREATE TABLE identity_wrong_values (
  conversation_id uuid        NOT NULL,
  fingerprint     text        NOT NULL,
  counted_with    text        NOT NULL,
  counted_in      xid8        NOT NULL,
  counted_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (conversation_id, fingerprint)
);

-- The server's account may read its own conversation's fingerprints (the operator panel shows how many) and nothing
-- else: no insert, update or delete, and no other conversation's rows (the rule 010_conversation_rows.sql applies to
-- every per-conversation table). The owner-rights function below is not bound by these rules.
ALTER TABLE identity_wrong_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY app_read ON identity_wrong_values FOR SELECT TO claims_app
  USING (conversation_id = NULLIF(current_setting('app.conversation_id', true), '')::uuid);
CREATE POLICY loader_all ON identity_wrong_values FOR ALL TO claims_loader USING (true) WITH CHECK (true);

GRANT SELECT (conversation_id, fingerprint, counted_at) ON identity_wrong_values TO claims_app;
GRANT ALL ON identity_wrong_values TO claims_loader;

-- Compare the stated values for the bound conversation, count each wrong one once, and return what match_identity
-- returns. Refuses (IL001) a conversation already at its limit, unless these exact values counted a wrong value
-- earlier in this same transaction (the call that reached the limit, repeated).
CREATE FUNCTION check_identity(p_conversation uuid, p_name text, p_dobs date[], p_phone text, p_email text,
                               p_id_last4 text, p_policy text)
  RETURNS TABLE (party_id text, kinds text[], policy_ok boolean)
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
#variable_conflict use_column
DECLARE
  -- Each value as the comparison reads it (match_identity's own rules); NULL when it would not be compared.
  v_name   text;
  v_dob    text := (SELECT string_agg(DISTINCT to_char(d, 'YYYY-MM-DD'), ',' ORDER BY to_char(d, 'YYYY-MM-DD'))
                    FROM unnest(p_dobs) AS d);
  v_phone  text := CASE WHEN p_phone ~ '^[0-9]{10,}$' THEN p_phone END;
  v_email  text := CASE WHEN lower(btrim(p_email)) ~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$'
                        THEN lower(btrim(p_email)) END;
  v_id     text := CASE WHEN p_id_last4 ~ '^[0-9]{4}$' THEN p_id_last4 END;
  v_policy text := NULLIF(upper(regexp_replace(p_policy, '[^A-Za-z0-9]', '', 'g')), '');
  v_held   jsonb;
  v_call   text;
  v_count  int;
  v_again  int;
BEGIN
  PERFORM public.bound_conversation(p_conversation);
  SELECT CASE WHEN s.shape IN ('whole', 'single') THEN s.key END INTO v_name FROM public.name_shape(p_name) AS s;
  v_held := jsonb_strip_nulls(jsonb_build_object('name', v_name, 'dob', v_dob, 'phone', v_phone, 'email', v_email,
                                                 'id', v_id, 'policy', v_policy));
  v_call := encode(sha256(convert_to(p_conversation::text || '|' || v_held::text, 'UTF8')), 'hex');

  SELECT count(*), count(*) FILTER (WHERE w.counted_in = pg_current_xact_id() AND w.counted_with = v_call)
    INTO v_count, v_again
  FROM public.identity_wrong_values AS w
  WHERE w.conversation_id = p_conversation;
  IF v_count >= public.wrong_values_limit() AND v_again = 0 THEN
    RAISE EXCEPTION 'this conversation has reached its limit of wrong identity values; nothing more is compared'
      USING ERRCODE = 'IL001';
  END IF;

  RETURN QUERY
  WITH m AS MATERIALIZED (
         SELECT x.party_id, x.kinds, x.policy_ok
         FROM public.match_identity(CASE WHEN v_name IS NOT NULL THEN p_name END, p_dobs, v_phone, v_email, v_id,
                                    v_policy) AS x),
       held AS (SELECT h.key AS kind, h.value #>> '{}' AS value FROM jsonb_each(v_held) AS h),
       five AS (SELECT h.kind FROM held AS h WHERE h.kind <> 'policy'),
       scored AS (SELECT m.party_id, m.kinds, m.policy_ok IS TRUE AS policy_ok,
                         (SELECT count(*) FROM five AS r WHERE r.kind = ANY (m.kinds)) AS n
                  FROM m),
       top AS (SELECT coalesce(max(s.n), 0) AS n FROM scored AS s),
       tied AS (SELECT s.* FROM scored AS s, top AS t WHERE t.n > 0 AND s.n = t.n),
       candidate AS (
         SELECT s.* FROM scored AS s
         WHERE s.party_id = CASE
           WHEN (SELECT t.n FROM top AS t) > 0
             THEN coalesce((SELECT min(d.party_id) FROM tied AS d HAVING count(*) = 1),
                           (SELECT min(d.party_id) FROM tied AS d WHERE d.policy_ok HAVING count(*) = 1))
           ELSE (SELECT min(d.party_id) FROM scored AS d WHERE d.policy_ok HAVING count(*) = 1) END),
       wrong AS (
         SELECT r.kind FROM five AS r
         WHERE NOT EXISTS (SELECT 1 FROM candidate AS c WHERE r.kind = ANY (c.kinds))
           AND (EXISTS (SELECT 1 FROM candidate)
                OR NOT EXISTS (SELECT 1 FROM scored AS s WHERE r.kind = ANY (s.kinds)))
         UNION ALL
         SELECT h.kind FROM held AS h
         WHERE h.kind = 'policy'
           AND CASE WHEN EXISTS (SELECT 1 FROM candidate) THEN NOT (SELECT c.policy_ok FROM candidate AS c)
                    ELSE NOT EXISTS (SELECT 1 FROM scored AS s WHERE s.policy_ok) END),
       counted AS (
         INSERT INTO public.identity_wrong_values AS w (conversation_id, fingerprint, counted_with, counted_in)
         SELECT p_conversation,
                h.kind || ':' || left(encode(sha256(convert_to(
                  p_conversation::text || '|' || h.kind || '|' || h.value, 'UTF8')), 'hex'), 16),
                v_call, pg_current_xact_id()
         FROM wrong JOIN held AS h ON h.kind = wrong.kind
         ON CONFLICT (conversation_id, fingerprint) DO NOTHING)
  SELECT m.party_id, m.kinds, m.policy_ok FROM m ORDER BY m.party_id;
END $$;

REVOKE ALL ON FUNCTION wrong_values_limit() FROM PUBLIC;
REVOKE ALL ON FUNCTION check_identity(uuid, text, date[], text, text, text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION check_identity(uuid, text, date[], text, text, text, text) TO claims_app;
