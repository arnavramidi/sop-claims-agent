import shutil
from collections import Counter
from decimal import Decimal

from claims_agent.loader.read import FILES, MAX_ID_CHARS, MAX_TEXT_CHARS, read_folder
from claims_agent.settings import ROOT

STARTER = ROOT / "data" / "starter"
BAD = ROOT / "tests" / "data" / "bad_data"
NEW_SHAPE = ROOT / "tests" / "data" / "new_shape"


def claim(r, case_id):
    return next(c for c in r.claims if c.case_id == case_id)


def customer(r, party_id):
    return next(c for c in r.customers if c.party_id == party_id)


def followup(r, topic):
    return next(f for f in r.followups if f.topic == topic)


def copy_starter_without(tmp_path, name):
    folder = tmp_path / "data"
    shutil.copytree(STARTER, folder)
    (folder / name).unlink()
    return folder


def test_starter_loads_clean():
    r = read_folder(STARTER)
    assert (len(r.customers), len(r.claims), len(r.representatives), len(r.report)) == (9, 13, 2, 0)
    topics = [f.topic for f in r.followups if f.topic != "fallback"]
    assert topics == ["missing_required_material_alternatives", "submission_timing", "processing_time_after_submission",
                      "submission_method", "file_format_requirements", "receipt_confirmation"]
    assert sum(len(f.needs) for f in r.followups) == 16 and all(f.requires_documents for f in r.followups)


def test_the_fallback_line_is_a_followup_with_no_needs():
    f = followup(read_folder(STARTER), "fallback")
    assert f.needs == () and f.requires_documents and f.enabled and f.position == 7


def test_money_becomes_exact_decimals():
    assert claim(read_folder(STARTER), "CL-2011").net_pay == Decimal("780.00")


def test_vocabulary_comes_from_the_data():
    v = read_folder(STARTER).vocabulary
    assert v["case_type"] == {"healthcare", "dental", "auto"} and v["status"] == {"denied", "closed", "open"}
    assert v["relationship"] == {"son", "spouse"}
    assert v["need"] == {"document_submission", "next_steps", "denial_question", "general_claim_question", "status_inquiry"}


def test_bad_rows_are_set_aside_and_bad_fields_emptied():
    r = read_folder(BAD)
    assert Counter(e.level for e in r.report) == {"row": 6, "field": 7, "followup": 1}
    b3 = customer(r, "B3")
    assert (b3.dob, b3.id_last4, b3.phone, b3.email) == (None, None, None, None)
    c = claim(r, "CL-9003")
    assert (c.net_pay, c.appeal_deadline, c.created_at) == (None, None, None)
    assert [c.case_id for c in r.claims].count("CL-2048") == 1


def test_unknown_blank_disables_only_that_followup():
    r = read_folder(BAD)
    by = {f.topic: f.enabled for f in r.followups}
    assert by["bad_blank"] is False and by["submission_timing"] is True


def test_repeated_example_phrases_are_stored_once():
    assert followup(read_folder(STARTER), "file_format_requirements").examples.count("pdf") == 1


def test_missing_file_is_reported_not_fatal(tmp_path):
    folder = copy_starter_without(tmp_path, "representatives.json")
    r = read_folder(folder)
    assert r.representatives == [] and [(e.level, e.file) for e in r.report] == [("file", "representatives.json")]


def test_new_shape_needs_no_code_change():
    r = read_folder(NEW_SHAPE)
    assert r.vocabulary["case_type"] == {"travel"} and r.vocabulary["status"] == {"under_review"}
    assert [c.case_id for c in r.claims] == ["TRV/77-A"]


HOSTILE = ROOT / "tests" / "data" / "hostile"
BLANK_TOPICS = [f"hostile_{i}" for i in range(9)]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)


def _personal_values():
    """Every stored personal value in the hostile files (good and bad), as typed."""
    import json
    out = set()
    for raw in json.loads((HOSTILE / "policyholders.json").read_text()):
        for key, v in raw.items():
            if key not in ("party_id", "policy_number", "id_type"):
                out.update(s for s in _strings(v) if len(s) > 3)
    for raw in json.loads((HOSTILE / "representatives.json").read_text()):
        out.update(_strings(raw.get("rep_name")))
        out.update(_strings(raw.get("buyer_name")))
    return out


def test_no_report_entry_carries_a_personal_value():
    r = read_folder(HOSTILE)
    assert r.report
    text = "\n".join(f"{e.file}|{e.ref}|{e.field}|{e.reason}" for e in r.report)
    leaked = [v for v in _personal_values() if v in text]
    assert leaked == []
    assert "Secretname" not in text and "SECRET" not in text.upper().replace("SECRETNAME", "")


def test_only_plain_allowed_blanks_enable_a_followup():
    r = read_folder(HOSTILE)
    by = {f.topic: f.enabled for f in r.followups}
    assert [by[t] for t in BLANK_TOPICS] == [False] * 9 and by["submission_timing"] is True
    assert Counter(e.level for e in r.report if e.reason.endswith("follow-up disabled")) == {"followup": 9}


def test_hostile_json_is_an_unreadable_file_not_a_crash(tmp_path):
    for name, content in [("consent_scenarios.json", "[" * 200_000), ("representatives.json", "x" * 6_000_000)]:
        folder = tmp_path / name
        shutil.copytree(STARTER, folder)
        (folder / name).write_text(content)
        r = read_folder(folder)
        assert [(e.level, e.file) for e in r.report] == [("file", name)]


def test_a_thousand_claims_pass_the_size_cap(tmp_path):
    import json
    shutil.copytree(STARTER, tmp_path / "d")
    claims = [{"case_id": f"X-{i}", "party_id": "P9", "case_type": "auto", "status": "open",
               "summary": "s" * 200} for i in range(1000)]
    (tmp_path / "d" / "claims.json").write_text(json.dumps(claims))
    r = read_folder(tmp_path / "d")
    assert len(r.claims) == 1000 and r.report == []


def test_text_the_database_cannot_store_is_set_aside_or_emptied():
    r = read_folder(HOSTILE)
    ids = {c.party_id for c in r.customers}
    assert {"H1", "H4"} <= ids and not ({"H2", "H3"} & ids) and not any("\x00" in p for p in ids)
    assert {c.case_id for c in r.claims} >= {"HC-1"} and "HC-3" not in {c.case_id for c in r.claims}
    assert claim(r, "HC-1").summary is None
    assert [d.name for d in r.documents if d.case_id == "HC-1"] == ["good doc"]
    for text in _all_text(r):
        assert "\x00" not in text
        text.encode("utf-8")


def _all_text(r):
    from dataclasses import asdict
    for rows in (r.customers, r.aliases, r.representatives, r.claims, r.documents, r.guidance,
                 r.followups, r.consent_scripts, r.report):
        for row in rows:
            yield from _strings(asdict(row))
    for k, vs in r.vocabulary.items():
        yield k
        yield from vs


def test_phone_separators_are_stripped_to_one_form():
    r = read_folder(HOSTILE)
    assert customer(r, "H1").phone == "+16505212836"
    assert [a.value for a in r.aliases if a.party_id == "H1" and a.kind == "phone"] == ["6505212830"]
    assert customer(read_folder(STARTER), "P9").phone == "+16505212836"


def test_a_settings_key_is_a_blank_only_if_it_is_a_plain_identifier():
    r = read_folder(HOSTILE)
    stored = {g.key for g in r.guidance if g.kind == "claim_followup_settings"}
    assert stored == {"average_processing_time_after_submission", "human_review_after_document_alternatives_exhausted"}


def test_non_ascii_digits_are_emptied_and_reported():
    r = read_folder(HOSTILE)
    h6 = customer(r, "H6")
    assert (h6.phone, h6.id_last4) == (None, None)
    assert {(e.ref, e.field) for e in r.report} >= {("H6", "phone"), ("H6", "id_last4")}


def _positions(value):
    """Every (path, kind) of a string or number leaf, and of a dict key, in a parsed JSON value. Iterative."""
    stack = [((), value)]
    while stack:
        path, v = stack.pop()
        if isinstance(v, dict):
            for k, x in v.items():
                yield path + (k,), "key"
                stack.append((path + (k,), x))
        elif isinstance(v, list):
            stack.extend((path + (i,), x) for i, x in enumerate(v))
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
            yield path, "leaf"


def _plant(root, path, kind, bad):
    """A copy of `root` with the leaf at `path` (or the key ending `path`) replaced by `bad`."""
    import copy
    root = copy.deepcopy(root)
    if not path:
        return bad
    parent = root
    for step in path[:-1]:
        parent = parent[step]
    last = path[-1]
    if kind == "key":
        parent[bad] = parent.pop(last)
    else:
        parent[last] = bad
    return root


PLANTS = {"nul": "bad\u0000text", "surrogate": "bad\ud800text", "nan": float("nan"), "inf": float("inf")}


def _bad_anywhere(r):
    import math
    for text in _all_text(r):
        if "\x00" in text:
            return True
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return True
    stack = [g.body for g in r.guidance] + [dict(f.__dict__) for f in r.followups]
    while stack:
        v = stack.pop()
        if isinstance(v, float) and not math.isfinite(v):
            return True
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
    amounts = [getattr(c, m) for c in r.claims for m in ("net_pay", "net_fee", "allowed_max_amount", "expected_reimbursement_amount")]
    return any(a is not None and not a.is_finite() for a in amounts)


def test_one_gate_no_unstorable_value_reaches_a_row_wherever_it_is_planted(tmp_path):
    import json
    folder = tmp_path / "d"
    shutil.copytree(STARTER, folder)
    planted = 0
    for file in sorted(p.name for p in STARTER.glob("*.json") if p.name in FILES):  # the loader reads only its own files
        original = json.loads((STARTER / file).read_text())
        for path, kind in list(_positions(original)):
            for label, bad in PLANTS.items():
                if kind == "key" and not isinstance(bad, str):
                    continue
                text = json.dumps(_plant(original, path, kind, bad))
                if label == "inf":
                    text = text.replace("Infinity", "1e400")
                (folder / file).write_text(text)
                r = read_folder(folder)
                assert not _bad_anywhere(r), (file, path, label)
                assert r.report, (file, path, label)
                planted += 1
        (folder / file).write_text((STARTER / file).read_text())
    assert planted > 500


def test_a_gated_value_is_reported_in_its_own_words():
    r = read_folder(HOSTILE)
    reasons = {e.reason for e in r.report if (e.ref, e.field) == ("H1", "id_type")}
    assert reasons == {"a value in this field cannot be stored (a character, number or length the database refuses)"}


def test_overlong_fields_follow_the_two_level_rule(tmp_path):
    import json
    shutil.copytree(STARTER, tmp_path / "d")
    folder = tmp_path / "d"
    claims = json.loads((folder / "claims.json").read_text())
    claims.append({"case_id": "L-1", "party_id": "P9", "status": "s" * (MAX_ID_CHARS + 1)})
    claims.append({"case_id": "L-2", "party_id": "P9", "status": "open", "case_type": "t" * (MAX_ID_CHARS + 1),
                   "summary": "x" * (MAX_TEXT_CHARS + 1)})
    claims.append({"case_id": "L-3", "party_id": "P9", "status": "open", "summary": "x" * MAX_TEXT_CHARS})
    (folder / "claims.json").write_text(json.dumps(claims))
    r = read_folder(folder)
    assert {c.case_id for c in r.claims} == {c["case_id"] for c in json.loads((STARTER / "claims.json").read_text())} | {"L-2", "L-3"}
    assert (claim(r, "L-2").case_type, claim(r, "L-2").summary) == (None, None)
    assert len(claim(r, "L-3").summary) == MAX_TEXT_CHARS
    assert Counter(e.level for e in r.report) == {"row": 1, "field": 2, "file": 1}  # the file entry is the gate removing the over-long summary


def test_a_rejected_settings_key_is_not_an_allowed_blank(tmp_path):
    import json
    shutil.copytree(STARTER, tmp_path / "d")
    path = tmp_path / "d" / "required_document_guideline.json"
    g = json.loads(path.read_text())
    g["claim_followup_settings"]["rejected_setting"] = {"en": float("nan")}
    g["claim_followup_guidance"].append({"topic": "uses_it", "intent_hints": ["next_steps"], "requires_documents": True,
                                         "en": "For {case_id}: {rejected_setting}"})
    path.write_text(json.dumps(g))
    r = read_folder(tmp_path / "d")
    assert followup(r, "uses_it").enabled is False and followup(r, "submission_timing").enabled is True


def _with_a_bad_first_row(folder):
    """Each list file (and the follow-up list) starts with a row the reader sets aside, so a count of
    accepted rows and a position in the file differ by one."""
    import json
    for name, bad in [("policyholders.json", {"name": "No Id"}), ("claims.json", {"party_id": "P9", "status": "open"}),
                      ("representatives.json", {"buyer_party_id": "P9"})]:
        data = json.loads((folder / name).read_text())
        (folder / name).write_text(json.dumps([bad, *data]))
    g = json.loads((folder / "required_document_guideline.json").read_text())
    g["claim_followup_guidance"].insert(0, {"en": "no topic"})
    (folder / "required_document_guideline.json").write_text(json.dumps(g))


def test_every_row_carries_its_position_in_its_own_file(tmp_path):
    import json
    from claims_agent.loader import read
    from claims_agent.loader.read import Origin
    folder = tmp_path / "d"
    shutil.copytree(STARTER, folder)
    _with_a_bad_first_row(folder)
    raw = {name: json.loads((folder / name).read_text()) for name in (
        read.CUSTOMERS_FILE, read.CLAIMS_FILE, read.REPRESENTATIVES_FILE, read.CONSENT_FILE, read.GUIDANCE_FILE, read.GLOSSARY_FILE)}
    r = read_folder(folder)

    def at(name, key, value):
        return f"#{[x.get(key) for x in raw[name]].index(value) + 1}"

    assert r.customers and r.aliases and r.claims and r.documents and r.representatives
    for c in r.customers:
        assert c.origin == Origin(read.CUSTOMERS_FILE, None, at(read.CUSTOMERS_FILE, "party_id", c.party_id))
    for a in r.aliases:
        assert a.origin == Origin(read.CUSTOMERS_FILE, f"{a.kind}_aliases", at(read.CUSTOMERS_FILE, "party_id", a.party_id))
    for c in r.claims:
        assert c.origin == Origin(read.CLAIMS_FILE, None, at(read.CLAIMS_FILE, "case_id", c.case_id))
    for d in r.documents:
        assert d.origin == Origin(read.CLAIMS_FILE, "documents_needed", at(read.CLAIMS_FILE, "case_id", d.case_id))
    for p in r.representatives:
        assert p.origin == Origin(read.REPRESENTATIVES_FILE, None, at(read.REPRESENTATIVES_FILE, "rep_name", p.rep_name))
    consent = list(raw[read.CONSENT_FILE])
    for s in r.consent_scripts:
        assert s.origin == Origin(read.CONSENT_FILE, None, f"#{consent.index(s.name) + 1}")
    guidance, glossary = raw[read.GUIDANCE_FILE], raw[read.GLOSSARY_FILE]["field_descriptions"]
    for g in r.guidance:
        if g.kind == "field_description":
            assert g.origin == Origin(read.GLOSSARY_FILE, "field_descriptions", f"#{list(glossary).index(g.key) + 1}")
        elif g.kind == "default_guidance":
            assert g.origin == Origin(read.GUIDANCE_FILE, "default_guidance", None)
        else:
            assert g.origin == Origin(read.GUIDANCE_FILE, g.kind, f"#{list(guidance[g.kind]).index(g.key) + 1}")
    topics = [x.get("topic") for x in guidance["claim_followup_guidance"]]
    for f in r.followups:
        if f.topic == "fallback":
            assert f.origin == Origin(read.GUIDANCE_FILE, "claim_followup_fallback", None)
        else:
            assert f.origin == Origin(read.GUIDANCE_FILE, "claim_followup_guidance", f"#{topics.index(f.topic) + 1}")


def test_a_reader_ref_is_the_position_in_the_file_not_a_count_of_accepted_rows(tmp_path):
    import json
    folder = tmp_path / "d"
    shutil.copytree(STARTER, folder)
    consent = json.loads((folder / "consent_scenarios.json").read_text())
    consent["x" * (MAX_ID_CHARS + 1)] = {"status_sequence": ["pending"]}  # third in the file, second accepted
    (folder / "consent_scenarios.json").write_text(json.dumps({"bad": {"status_sequence": []}, **consent}))
    g = json.loads((folder / "required_document_guideline.json").read_text())
    g["claim_followup_guidance"] = [{"en": "no topic"}, *g["claim_followup_guidance"], {"topic": "no_text"}]
    (folder / "required_document_guideline.json").write_text(json.dumps(g))
    r = read_folder(folder)
    refs = {(e.file, e.ref) for e in r.report if e.level == "row"}
    assert ("consent_scenarios.json", "#4") in refs   # bad, default, timeout, then the long name
    assert {("required_document_guideline.json", "#1"), ("required_document_guideline.json", "#8")} <= refs


DEEP = 9_000


def _deep_folder(tmp_path, file, put):
    """A copy of the starter data where `put(data)` places the string "__DEEP__" somewhere in `file`; that
    string becomes a list nested DEEP levels (written as text: nothing here recurses)."""
    import json
    folder = tmp_path / "deep"
    shutil.copytree(STARTER, folder)
    data = json.loads((folder / file).read_text())
    put(data)
    (folder / file).write_text(json.dumps(data).replace('"__DEEP__"', "[" * DEEP + "1" + "]" * DEEP))
    return folder


def _depth(value):
    """How many containers deep a value goes (a scalar is 0). Iterative."""
    best, stack = 0, [(value, 1)]
    while stack:
        v, d = stack.pop()
        if isinstance(v, dict):
            best = max(best, d)
            stack.extend((x, d + 1) for x in [*v.keys(), *v.values()])
        elif isinstance(v, (list, tuple, set, frozenset)):
            best = max(best, d)
            stack.extend((x, d + 1) for x in v)
    return best


def _deepest(r):
    """The deepest value anywhere in a LoadResult: every field of every row, the vocabulary, the report."""
    from dataclasses import fields
    out = _depth(r.vocabulary)
    for rows in (r.customers, r.aliases, r.representatives, r.claims, r.documents, r.guidance, r.followups,
                 r.consent_scripts, r.report):
        for row in rows:
            out = max(out, *(_depth(getattr(row, f.name)) for f in fields(row)))
    return out


def test_a_deep_guidance_body_is_set_aside_and_reported(tmp_path):
    from claims_agent.loader.read import DEEP_REASON, MAX_DEPTH
    folder = _deep_folder(tmp_path, "required_document_guideline.json",
                          lambda g: g["case_type_guidance"].__setitem__("deep_entry", {"en": "__DEEP__"}))
    r = read_folder(folder)
    assert "deep_entry" not in {g.key for g in r.guidance}
    assert {g.key for g in r.guidance if g.kind == "case_type_guidance"} == \
        {g.key for g in read_folder(STARTER).guidance if g.kind == "case_type_guidance"}
    assert ("row", "case_type_guidance.deep_entry", DEEP_REASON) in {(e.level, e.ref, e.reason) for e in r.report}
    assert _deepest(r) <= MAX_DEPTH


def test_a_deep_claim_field_is_emptied_and_reported(tmp_path):
    from claims_agent.loader.read import DEEP_REASON, MAX_DEPTH
    folder = _deep_folder(tmp_path, "claims.json", lambda c: c[0].__setitem__("summary", "__DEEP__"))
    r = read_folder(folder)
    first = r.claims[0]
    assert first.case_id == "CL-2048" and first.summary is None and len(r.claims) == 13
    assert ("field", "CL-2048", "summary", DEEP_REASON) in {(e.level, e.ref, e.field, e.reason) for e in r.report}
    assert _deepest(r) <= MAX_DEPTH


def _data_sets():
    return [STARTER, *sorted(p for p in (ROOT / "tests" / "data").iterdir() if p.is_dir())]


def test_nothing_leaves_the_reader_nested_deeper_than_the_cap():
    from claims_agent.loader.read import MAX_DEPTH
    sets = _data_sets()
    assert len(sets) >= 6
    for folder in sets:
        assert _deepest(read_folder(folder)) <= MAX_DEPTH, folder.name


def test_a_position_is_counted_in_the_file_before_anything_is_dropped(tmp_path):
    import json
    folder = tmp_path / "d"
    shutil.copytree(STARTER, folder)
    seq = {"status_sequence": ["pending"]}
    # consent: the second key cannot be stored (the gate drops it); the third entry is too long, so reported
    (folder / "consent_scenarios.json").write_text(
        '{"default": %s, "bad\\u0000key": %s, "%s": %s}' % (json.dumps(seq), json.dumps(seq), "x" * (MAX_ID_CHARS + 1), json.dumps(seq)))
    g = json.loads((folder / "required_document_guideline.json").read_text())
    body = {"en": "text"}
    g["case_type_guidance"] = {"first": body, "bad\x00key": body, "third": body}
    (folder / "required_document_guideline.json").write_text(json.dumps(g))
    # glossary: a duplicate key (the second occurrence is the one used) and then a third entry
    (folder / "claim_schema.json").write_text(
        '{"field_descriptions": {"net_pay": {"en": "a"}, "net_pay": {"en": "b"}, "net_fee": {"en": "c"}}}')
    r = read_folder(folder)
    assert ("consent_scenarios.json", "#3") in {(e.file, e.ref) for e in r.report if e.level == "row"}
    assert {s.name: s.origin.ref for s in r.consent_scripts} == {"default": "#1"}
    assert {g.key: g.origin.ref for g in r.guidance if g.kind == "case_type_guidance"} == {"first": "#1", "third": "#3"}
    glossary = {g.key: (g.origin.ref, g.body["en"]) for g in r.guidance if g.kind == "field_description"}
    assert glossary == {"net_pay": ("#2", "b"), "net_fee": ("#3", "c")}


def test_a_glossary_key_is_capped_like_every_other_key(tmp_path):
    import json
    folder = tmp_path / "d"
    shutil.copytree(STARTER, folder)
    long_key = "k" * (MAX_ID_CHARS + 1)
    schema = json.loads((folder / "claim_schema.json").read_text())
    schema["field_descriptions"][long_key] = {"en": "too long a key"}
    (folder / "claim_schema.json").write_text(json.dumps(schema))
    r = read_folder(folder)
    assert long_key not in {g.key for g in r.guidance}
    assert [(e.level, e.file) for e in r.report] == [("row", "claim_schema.json")]
    assert all(len(e.ref or "") <= MAX_ID_CHARS and len(e.field or "") <= MAX_ID_CHARS for e in r.report)
