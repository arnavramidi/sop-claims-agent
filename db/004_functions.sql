-- Identity matching inside the database. Runs as claims_owner, so the owner owns every function.
-- Stated values come in; only party ids, the kinds that matched, yes/no answers and a consent status go
-- out. Stored dates of birth, phones, emails, ID digits and aliases never leave the database.
--
-- Every argument is treated as hostile: the server's account may call these with anything. So each
-- function re-applies the stated-value rules itself (identity/normalize.py is only a convenience), uses
-- no dynamic SQL, pins its search path with the temporary schema last, names every table with its
-- schema, and raises only fixed messages. Owner-created functions are closed to PUBLIC by default
-- (migrate.apply); each one the server calls is granted to it by name at the end of this file.
-- Every function that takes a conversation id first passes it through bound_conversation, so it can act
-- only on the conversation this transaction is bound to.

-- The conversation a function may act on: the one this transaction is bound to (the transaction-local
-- setting app.conversation_id, which alone decides whose claims are visible). Acting on any other
-- conversation, or with none bound, would mix two customers, so it raises before anything is read or
-- written. The message names no id.
CREATE FUNCTION bound_conversation(p_conversation uuid) RETURNS uuid
  LANGUAGE plpgsql STABLE
  SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE
  v_bound uuid := NULLIF(current_setting('app.conversation_id', true), '')::uuid;
BEGIN
  IF p_conversation IS NULL OR v_bound IS NULL OR p_conversation <> v_bound THEN
    RAISE EXCEPTION 'the conversation id is not the one this transaction is bound to'
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  RETURN p_conversation;
END $$;

-- True when a stated name candidate equals the stored name, or (stored name of six letters or more) is
-- one inserted, deleted or substituted letter away (Ruling 14). Both sides are lower-cased and kept to
-- letters only, under a fixed Unicode collation so the result does not depend on the database's locale.
CREATE FUNCTION name_matches(candidates text[], stored text) RETURNS boolean
  LANGUAGE sql IMMUTABLE PARALLEL SAFE
  SET search_path = pg_catalog, public, pg_temp AS $$
  WITH s AS (SELECT regexp_replace(lower(stored COLLATE pg_c_utf8), '[^[:alpha:]]', '', 'g') AS key),
       c AS (SELECT regexp_replace(lower(x COLLATE pg_c_utf8), '[^[:alpha:]]', '', 'g') AS key
             FROM unnest(candidates) AS x)
  SELECT EXISTS (
    SELECT 1 FROM s, c
    WHERE s.key <> '' AND c.key <> ''
      AND (c.key = s.key
           OR CASE WHEN length(s.key) >= 6 AND length(s.key) <= 255 AND length(c.key) <= 255
                        AND abs(length(c.key) - length(s.key)) <= 1
                   THEN public.levenshtein_less_equal(c.key, s.key, 1) <= 1
                   ELSE false END))
$$;

-- One row per customer for whom at least one kind (or the policy number) matches.
-- `policy_ok` is NULL when no policy number was stated, so callers can tell "not given" from "not theirs".
-- An alias only ever makes its own kind match, so it never adds a second kind.
CREATE FUNCTION match_identity(p_names text[], p_dobs date[], p_phone text, p_email text, p_id_last4 text,
                               p_policy text)
  RETURNS TABLE (party_id text, kinds text[], policy_ok boolean)
  LANGUAGE plpgsql STABLE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
#variable_conflict use_column
DECLARE
  -- A phone of fewer than ten digits would let a suffix match read a stored number digit by digit.
  v_phone  text := CASE WHEN p_phone ~ '^[0-9]{10,}$' THEN p_phone END;
  v_email  text := CASE WHEN lower(btrim(p_email)) ~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$'
                        THEN lower(btrim(p_email)) END;
  v_id     text := CASE WHEN p_id_last4 ~ '^[0-9]{4}$' THEN p_id_last4 END;
  v_policy text := NULLIF(upper(regexp_replace(p_policy, '[^A-Za-z0-9]', '', 'g')), '');
BEGIN
  -- Every ordering of four words, and the two readings of one date: more would test many guesses at once.
  IF coalesce(cardinality(p_names), 0) > 24 OR coalesce(cardinality(p_dobs), 0) > 2 THEN
    RAISE EXCEPTION 'too many values for one identity check';
  END IF;
  RETURN QUERY
  SELECT m.party_id, m.kinds, m.policy_ok
  FROM (
    SELECT c.party_id,
           array_remove(ARRAY[
             CASE WHEN public.name_matches(p_names, c.name)
                    OR EXISTS (SELECT 1 FROM public.customer_aliases a
                               WHERE a.party_id = c.party_id AND a.kind = 'name'
                                 AND public.name_matches(p_names, a.value))
                  THEN 'name' END,
             CASE WHEN c.dob = ANY (p_dobs) THEN 'dob' END,
             CASE WHEN v_phone IS NOT NULL
                   AND (right(regexp_replace(c.phone, '[^0-9]', '', 'g'), length(v_phone)) = v_phone
                        OR EXISTS (SELECT 1 FROM public.customer_aliases a
                                   WHERE a.party_id = c.party_id AND a.kind = 'phone'
                                     AND right(regexp_replace(a.value, '[^0-9]', '', 'g'), length(v_phone)) = v_phone))
                  THEN 'phone' END,
             CASE WHEN v_email IS NOT NULL
                   AND (lower(btrim(c.email)) = v_email
                        OR EXISTS (SELECT 1 FROM public.customer_aliases a
                                   WHERE a.party_id = c.party_id AND a.kind = 'email'
                                     AND lower(btrim(a.value)) = v_email))
                  THEN 'email' END,
             CASE WHEN v_id IS NOT NULL AND c.id_last4 ~ '^[0-9]{4}$' AND c.id_last4 = v_id THEN 'id' END
           ], NULL) AS kinds,
           CASE WHEN v_policy IS NOT NULL
                THEN coalesce(upper(regexp_replace(c.policy_number, '[^A-Za-z0-9]', '', 'g')) = v_policy, false)
           END AS policy_ok
    FROM public.customers c
  ) m
  WHERE cardinality(m.kinds) > 0 OR m.policy_ok
  ORDER BY m.party_id;
END $$;

-- Marks the conversation verified as `p_party` only when at least three kinds match and any policy
-- number stated is theirs. A pending consent request is dropped, so a later approval cannot switch the
-- conversation to someone else.
CREATE FUNCTION bind_policyholder(p_conversation uuid, p_party text, p_names text[], p_dobs date[],
                                  p_phone text, p_email text, p_id_last4 text, p_policy text)
  RETURNS boolean
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE
  v_ok boolean;
BEGIN
  PERFORM public.bound_conversation(p_conversation);
  SELECT cardinality(m.kinds) >= 3 AND m.policy_ok IS NOT FALSE INTO v_ok
  FROM public.match_identity(p_names, p_dobs, p_phone, p_email, p_id_last4, p_policy) m
  WHERE m.party_id = p_party;
  IF v_ok IS NOT TRUE THEN
    RETURN false;
  END IF;
  UPDATE public.conversations
     SET verified_party_id = p_party, pending_party_id = NULL, updated_at = now()
   WHERE id = p_conversation;
  IF NOT FOUND THEN
    RETURN false;
  END IF;
  DELETE FROM public.consent_requests WHERE conversation_id = p_conversation;
  RETURN true;
END $$;

-- A representative asks for a policyholder's claims. The representative list is looked at only after
-- the same three-kind check as binding (Ruling 11). A stated relationship must equal the stored one
-- (case-insensitive) when both are known. Listed: the customer becomes pending and a consent request is
-- recorded with the conversation's script. Nothing is written otherwise.
CREATE FUNCTION request_consent(p_conversation uuid, p_party text, p_names text[], p_dobs date[],
                                p_phone text, p_email text, p_id_last4 text, p_policy text,
                                p_rep_names text[], p_relationship text)
  RETURNS text
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE
  v_ok           boolean;
  v_relationship text := lower(NULLIF(btrim(p_relationship), ''));
  v_script       text;
BEGIN
  PERFORM public.bound_conversation(p_conversation);
  IF coalesce(cardinality(p_rep_names), 0) > 24 THEN
    RAISE EXCEPTION 'too many values for one identity check';
  END IF;
  SELECT cardinality(m.kinds) >= 3 AND m.policy_ok IS NOT FALSE INTO v_ok
  FROM public.match_identity(p_names, p_dobs, p_phone, p_email, p_id_last4, p_policy) m
  WHERE m.party_id = p_party;
  IF v_ok IS NOT TRUE THEN
    RETURN 'refused';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM public.representatives r
                 WHERE r.party_id = p_party
                   AND public.name_matches(p_rep_names, r.rep_name)
                   AND (v_relationship IS NULL OR r.relationship IS NULL
                        OR lower(btrim(r.relationship)) = v_relationship)) THEN
    RETURN 'not_listed';
  END IF;
  UPDATE public.conversations SET pending_party_id = p_party, updated_at = now()
   WHERE id = p_conversation
  RETURNING consent_script INTO v_script;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'no such conversation';
  END IF;
  -- Asking again for the same customer keeps the request's progress; a different customer starts afresh.
  INSERT INTO public.consent_requests AS r (conversation_id, party_id, script, status)
  VALUES (p_conversation, p_party, v_script, 'requested')
  ON CONFLICT (conversation_id) DO UPDATE
     SET party_id = EXCLUDED.party_id, script = EXCLUDED.script, checks_used = 0, status = EXCLUDED.status
   WHERE r.party_id IS DISTINCT FROM EXCLUDED.party_id;
  RETURN 'requested';
END $$;

-- One more check of the consent request: the script's status at that position, repeating the last one
-- when the script runs out (Ruling 13). `approved` copies the pending customer (when one is set, and it is
-- the one asked about) to the verified customer. No request: 'no_request'. A script that is not loaded:
-- 'no_script'.
CREATE FUNCTION check_consent(p_conversation uuid) RETURNS text
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE
  v_used   int;
  v_script text;
  v_party  text;
  v_status text;
BEGIN
  PERFORM public.bound_conversation(p_conversation);
  UPDATE public.consent_requests r SET checks_used = r.checks_used + 1
   WHERE r.conversation_id = p_conversation
  RETURNING r.checks_used, r.script, r.party_id INTO v_used, v_script, v_party;
  IF NOT FOUND THEN
    RETURN 'no_request';
  END IF;
  SELECT s.status_sequence[LEAST(v_used, cardinality(s.status_sequence))] INTO v_status
  FROM public.consent_scripts s WHERE s.name = v_script;
  v_status := coalesce(v_status, 'no_script');
  UPDATE public.consent_requests SET status = v_status WHERE conversation_id = p_conversation;
  IF v_status = 'approved' THEN
    UPDATE public.conversations
       SET verified_party_id = pending_party_id, updated_at = now()
     WHERE id = p_conversation AND pending_party_id IS NOT NULL AND pending_party_id = v_party;
  END IF;
  RETURN v_status;
END $$;

-- Forget who the conversation is verified as, who is pending, and any consent request.
CREATE FUNCTION clear_verified(p_conversation uuid) RETURNS void
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
BEGIN
  PERFORM public.bound_conversation(p_conversation);
  UPDATE public.conversations SET verified_party_id = NULL, pending_party_id = NULL, updated_at = now()
   WHERE id = p_conversation;
  DELETE FROM public.consent_requests WHERE conversation_id = p_conversation;
END $$;

-- The verified customer's name and policy number; no row when the conversation is not verified.
CREATE FUNCTION account_info(p_conversation uuid) RETURNS TABLE (name text, policy_number text)
  LANGUAGE plpgsql STABLE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
#variable_conflict use_column
BEGIN
  PERFORM public.bound_conversation(p_conversation);
  RETURN QUERY
  SELECT c.name, c.policy_number
  FROM public.conversations v JOIN public.customers c ON c.party_id = v.verified_party_id
  WHERE v.id = p_conversation;
END $$;

-- The simulated send: an outbox row for the verified customer, addressed to their stored email, which the
-- server's account cannot read back. Raises when there is no verified customer or no email on file.
CREATE FUNCTION send_summary(p_conversation uuid, p_subject text, p_body text) RETURNS bigint
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER
  SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE
  v_party text;
  v_email text;
  v_id    bigint;
BEGIN
  PERFORM public.bound_conversation(p_conversation);
  SELECT v.verified_party_id, c.email INTO v_party, v_email
  FROM public.conversations v LEFT JOIN public.customers c ON c.party_id = v.verified_party_id
  WHERE v.id = p_conversation;
  IF v_party IS NULL THEN
    RAISE EXCEPTION 'the conversation has no verified customer';
  END IF;
  IF v_email IS NULL THEN
    RAISE EXCEPTION 'the verified customer has no email on file';
  END IF;
  INSERT INTO public.outbox (conversation_id, party_id, recipient, subject, body)
  VALUES (p_conversation, v_party, v_email, p_subject, p_body)
  RETURNING id INTO v_id;
  RETURN v_id;
END $$;

-- Closed by default already; stated again per function, then opened to the server's account only.
REVOKE ALL ON FUNCTION bound_conversation(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION name_matches(text[], text) FROM PUBLIC;
REVOKE ALL ON FUNCTION match_identity(text[], date[], text, text, text, text) FROM PUBLIC;
REVOKE ALL ON FUNCTION bind_policyholder(uuid, text, text[], date[], text, text, text, text) FROM PUBLIC;
REVOKE ALL ON FUNCTION request_consent(uuid, text, text[], date[], text, text, text, text, text[], text) FROM PUBLIC;
REVOKE ALL ON FUNCTION check_consent(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION clear_verified(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION account_info(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION send_summary(uuid, text, text) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION match_identity(text[], date[], text, text, text, text) TO claims_app;
GRANT EXECUTE ON FUNCTION bind_policyholder(uuid, text, text[], date[], text, text, text, text) TO claims_app;
GRANT EXECUTE ON FUNCTION request_consent(uuid, text, text[], date[], text, text, text, text, text[], text) TO claims_app;
GRANT EXECUTE ON FUNCTION check_consent(uuid) TO claims_app;
GRANT EXECUTE ON FUNCTION clear_verified(uuid) TO claims_app;
GRANT EXECUTE ON FUNCTION account_info(uuid) TO claims_app;
GRANT EXECUTE ON FUNCTION send_summary(uuid, text, text) TO claims_app;
