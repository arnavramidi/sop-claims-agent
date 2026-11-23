from collections import Counter
from dataclasses import replace
from string import Formatter

import psycopg
import pytest

from claims_agent.loader import read
from claims_agent.loader.read import (AliasRow, CustomerRow, DocumentRow, FollowupRow, GuidanceRow,
                                      ReportEntry, RepresentativeRow, read_folder)
from claims_agent.loader.write import write
from claims_agent.settings import ROOT

STARTER = ROOT / "data" / "starter"
REFUSED = "the database could not store this row"


async def test_written_guidance_and_vocabulary_are_readable_by_the_app(load_dataset):
    db = await load_dataset("starter")
    async with db.app.conversation(None) as c:
        assert (await (await c.execute("SELECT count(*) FROM vocabulary WHERE kind = 'case_type'")).fetchone())[0] == 3
        assert (await (await c.execute("SELECT count(*) FROM load_report")).fetchone())[0] == 0


async def test_bad_data_loads_and_reports(load_dataset):
    db = await load_dataset("bad_data")
    async with db.app.conversation(None) as c:
        assert (await (await c.execute("SELECT count(*) FROM load_report")).fetchone())[0] == 14


async def test_loading_twice_leaves_the_same_rows(load_dataset):
    db = await load_dataset("starter")
    await write(read_folder(ROOT / "data" / "starter"), db.loader_url)
    assert await db.loader_count("claims") == 13


async def test_the_app_reads_no_personal_value_in_the_load_report(load_dataset):
    import json
    db = await load_dataset("hostile")
    async with db.app.conversation(None) as c:
        rows = await (await c.execute("SELECT level, file, ref, field, reason FROM load_report")).fetchall()
    assert rows
    text = "\n".join("|".join(map(str, r)) for r in rows)
    folder = ROOT / "tests" / "data" / "hostile"
    for raw in json.loads((folder / "policyholders.json").read_text()):
        for key in ("name", "dob", "phone", "email", "id_last4", "name_aliases", "phone_aliases", "email_aliases"):
            for v in raw.get(key) if isinstance(raw.get(key), list) else [raw.get(key)]:
                if isinstance(v, str) and len(v) > 3:
                    assert v not in text
    assert "Secretname" not in text


async def test_unstorable_text_does_not_stop_the_load(load_dataset):
    db = await load_dataset("hostile")
    assert await db.loader_count("claims") == 6
    assert await db.loader_count("customers") == 7


async def _report(db):
    async with db.app.conversation(None) as c:
        return await (await c.execute("SELECT level, file, ref, field, reason FROM load_report")).fetchall()


async def test_an_overlong_status_is_reported_and_the_rest_loads(load_dataset, tmp_path):
    import hashlib
    import json
    import shutil
    db = await load_dataset("starter")
    shutil.copytree(ROOT / "data" / "starter", tmp_path / "d")
    claims = json.loads((tmp_path / "d" / "claims.json").read_text())
    noise = "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(47))[:3000]
    claims.append({"case_id": "LONG-1", "party_id": "P9", "status": noise})
    (tmp_path / "d" / "claims.json").write_text(json.dumps(claims))
    result = read_folder(tmp_path / "d")
    await write(result, db.loader_url)
    assert await db.loader_count("claims") == 13 and await db.loader_count("customers") == 9
    assert [r[0] for r in await _report(db)] == ["row"]


async def test_the_writer_sets_aside_a_row_the_database_refuses(load_dataset):
    import base64
    import random
    db = await load_dataset("starter")
    r = read_folder(ROOT / "data" / "starter")
    base = r.claims[0]
    # incompressible, so the database cannot shrink it under its index limit
    noise = base64.b85encode(random.Random(7).randbytes(6000)).decode()
    r.customers.append(CustomerRow(noise, "Too Long", None, None, None, None, None, None))
    r.customers.append(CustomerRow("Z2", "Nul\x00Name", None, None, None, None, None, None))
    r.customers.append(CustomerRow("Z3", "Fine", None, None, None, None, None, None))
    r.aliases.append(AliasRow(noise, "name", "alias of the refused one"))
    r.aliases.append(AliasRow("Z3", "name", "Fine Alias"))
    r.claims.append(replace(base, case_id="ZC-1", party_id=noise))
    r.claims.append(replace(base, case_id="ZC-2", party_id="Z3", status=noise))
    r.claims.append(replace(base, case_id="ZC-3", party_id="Z3"))
    r.documents.append(DocumentRow("ZC-2", "Z3", 1, "doc of a refused claim"))
    r.documents.append(DocumentRow("ZC-3", "Z3", 1, "doc of a good claim"))
    await write(r, db.loader_url)
    assert await db.loader_count("customers") == 10 and await db.loader_count("customer_aliases") == 4
    assert await db.loader_count("claims") == 14
    assert await db.loader_count("claim_documents") == 10
    rows = await _report(db)
    assert len(rows) == 6   # two customers, one alias, two claims, one document
    assert {r[4] for r in rows} == {"the database could not store this row"}
    text = "\n".join("|".join(map(str, x)) for x in rows)
    assert "Nul" not in text and noise[:50] not in text and "alias of" not in text


async def test_a_thousand_rows_through_the_savepoint_path_are_fast(load_dataset):
    import time
    db = await load_dataset("starter")
    r = read_folder(ROOT / "data" / "starter")
    r.claims += [replace(r.claims[0], case_id=f"G-{i}") for i in range(1000)]
    started = time.monotonic()
    await write(r, db.loader_url)
    assert time.monotonic() - started < 10
    assert await db.loader_count("claims") == 1013


async def _select(db, query):
    async with await psycopg.AsyncConnection.connect(db.loader_url) as conn:
        return await (await conn.execute(query)).fetchall()


async def _assert_derived_from_stored(db):
    """The general rule: everything derived is derived from the rows that were actually stored. The
    vocabulary is exactly the values of stored rows, and an enabled follow-up uses only blanks that have a
    stored value (case_id, documents, or the key of a stored settings row)."""
    vocabulary = await _select(db, "SELECT kind, value FROM vocabulary")
    expected = {("case_type", v) for (v,) in await _select(db, "SELECT case_type FROM claims WHERE case_type IS NOT NULL")}
    expected |= {("status", v) for (v,) in await _select(db, "SELECT status FROM claims")}
    expected |= {("relationship", v) for (v,) in await _select(
        db, "SELECT relationship FROM representatives WHERE relationship IS NOT NULL")}
    expected |= {("need", v) for (v,) in await _select(
        db, "SELECT jsonb_array_elements_text(body->'needs') FROM guidance WHERE kind = 'followup'")}
    assert len(vocabulary) == len(set(vocabulary)) and set(vocabulary) == expected
    settings = {k for (k,) in await _select(db, "SELECT key FROM guidance WHERE kind = 'claim_followup_settings'")}
    enabled = await _select(db, "SELECT key, body->>'template' FROM guidance "
                                "WHERE kind = 'followup' AND (body->>'enabled')::boolean")
    assert enabled
    for topic, template in enabled:
        blanks = {name for _, name, _, _ in Formatter().parse(template) if name is not None}
        assert blanks <= {"case_id", "documents"} | settings, topic


@pytest.mark.parametrize("name", ["starter", "bad_data", "hostile", "new_shape"])
async def test_everything_derived_comes_from_the_rows_stored(load_dataset, name):
    await _assert_derived_from_stored(await load_dataset(name))


async def test_a_followup_whose_setting_row_was_refused_is_disabled(load_dataset):
    """Built by hand: the reader would never pass on a settings row the database refuses."""
    db = await load_dataset("starter")
    r = read_folder(STARTER)
    r.guidance.append(GuidanceRow("claim_followup_settings", "refused_setting", {"en": "nul\x00"}))
    r.followups.append(FollowupRow("uses_refused_setting", ("next_steps",), True, (), "For {case_id}: {refused_setting}",
                                   8, True))
    await write(r, db.loader_url)
    enabled = dict(await _select(db, "SELECT key, (body->>'enabled')::boolean FROM guidance WHERE kind = 'followup'"))
    assert enabled["uses_refused_setting"] is False
    assert enabled["processing_time_after_submission"] is True   # its setting was stored
    rows = await _report(db)
    assert [(lv, ref, fld) for lv, _, ref, fld, _ in rows if lv == "followup"] == [("followup", "uses_refused_setting", "en")]
    assert [r[4] for r in rows if r[0] == "row"] == [REFUSED]
    await _assert_derived_from_stored(db)


async def test_vocabulary_holds_no_value_that_came_only_from_a_refused_row(load_dataset):
    """Built by hand: a claim, a representative and a follow-up that the database refuses each carry a
    vocabulary value no stored row has, and the reader's vocabulary also holds a value from no row at all."""
    db = await load_dataset("starter")
    r = read_folder(STARTER)
    base = r.claims[0]
    r.claims.append(replace(base, status="only_refused_status", case_type="only_refused_type"))  # duplicate case_id
    r.representatives.append(RepresentativeRow(r.customers[0].party_id, "Nul\x00Rep", "only_refused_relationship"))
    r.followups.append(FollowupRow("refused_followup", ("only_refused_need",), True, (), "nul\x00", 8, True))
    for kind, value in [("status", "only_refused_status"), ("case_type", "only_refused_type"),
                        ("relationship", "only_refused_relationship"), ("need", "only_refused_need"),
                        ("status", "in_no_row_at_all")]:
        r.vocabulary[kind].add(value)
    await write(r, db.loader_url)
    values = {v for (v,) in await _select(db, "SELECT value FROM vocabulary")}
    assert not {v for v in values if v.startswith("only_refused") or v == "in_no_row_at_all"}
    assert {"denied", "closed", "open", "son", "status_inquiry"} <= values
    assert Counter(r[4] for r in await _report(db)) == {REFUSED: 3}
    await _assert_derived_from_stored(db)


@pytest.mark.parametrize("failure", ["connection_terminated", "operational_error", "statement_timeout"])
async def test_a_failure_that_is_not_about_the_row_aborts_the_whole_load(load_dataset, monkeypatch, failure):
    """A lost connection, a timeout or any other operational failure during one insert says nothing about
    the row: the load raises and the data loaded before stays in place (the load is one transaction)."""
    db = await load_dataset("starter")
    original = psycopg.AsyncConnection.execute
    fired = []

    async def execute(self, query, params=None, **kwargs):
        if not fired and isinstance(params, (tuple, list)) and params[:1] in (("CL-2011",), ["CL-2011"]):
            fired.append(True)
            if failure == "connection_terminated":
                async with await psycopg.AsyncConnection.connect(db.superuser_url, autocommit=True) as su:
                    killed = await original(su, "SELECT pg_terminate_backend(%s, 5000)", [self.info.backend_pid])
                    assert (await killed.fetchone())[0] is True
            elif failure == "operational_error":
                raise psycopg.OperationalError("simulated network failure")
            else:
                raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")
        return await original(self, query, params, **kwargs)

    monkeypatch.setattr(psycopg.AsyncConnection, "execute", execute)
    with pytest.raises(psycopg.OperationalError):
        await write(read_folder(ROOT / "tests" / "data" / "bad_data"), db.loader_url)
    monkeypatch.undo()
    assert fired
    assert (await db.loader_count("claims"), await db.loader_count("customers"), await db.loader_count("load_report")) \
        == (13, 9, 0)


async def test_a_refused_row_is_reported_by_its_position_in_its_own_file(load_dataset, tmp_path):
    import json
    import shutil
    db = await load_dataset("starter")
    folder = tmp_path / "d"
    shutil.copytree(STARTER, folder)
    for name, bad in [("policyholders.json", {"name": "No Id"}), ("claims.json", {"party_id": "P9", "status": "open"}),
                      ("representatives.json", {"buyer_party_id": "P9"})]:
        data = json.loads((folder / name).read_text())
        (folder / name).write_text(json.dumps([bad, *data]))
    g = json.loads((folder / "required_document_guideline.json").read_text())
    g["claim_followup_guidance"].insert(0, {"en": "no topic"})
    (folder / "required_document_guideline.json").write_text(json.dumps(g))
    raw = {name: json.loads((folder / name).read_text()) for name in (
        read.CUSTOMERS_FILE, read.CLAIMS_FILE, read.REPRESENTATIVES_FILE, read.CONSENT_FILE, read.GUIDANCE_FILE,
        read.GLOSSARY_FILE)}
    r = read_folder(folder)

    # The database refuses customers P9 and P13, claim CL-3001, the second case-type guidance entry, the
    # follow-up processing_time_after_submission, the third glossary entry and the second consent script.
    nul = "nul\x00"
    r.customers = [replace(c, name=nul) if c.party_id in ("P9", "P13") else c for c in r.customers]
    r.claims = [replace(c, summary=nul) if c.case_id == "CL-3001" else c for c in r.claims]
    second_type = list(raw[read.GUIDANCE_FILE]["case_type_guidance"])[1]
    third_field = list(raw[read.GLOSSARY_FILE]["field_descriptions"])[2]
    r.guidance = [replace(x, body={"en": nul}) if (x.kind, x.key) in (("case_type_guidance", second_type),
                                                                      ("field_description", third_field)) else x
                  for x in r.guidance]
    r.followups = [replace(f, template=nul) if f.topic == "processing_time_after_submission" else f for f in r.followups]
    second_script = list(raw[read.CONSENT_FILE])[1]
    r.consent_scripts = [replace(s, status_sequence=(nul,)) if s.name == second_script else s for s in r.consent_scripts]
    await write(r, db.loader_url)

    def at(name, key, value):
        return f"#{[x.get(key) for x in raw[name]].index(value) + 1}"

    expected = Counter()
    for c in raw[read.CUSTOMERS_FILE]:
        if c.get("party_id") in ("P9", "P13"):
            expected[(read.CUSTOMERS_FILE, at(read.CUSTOMERS_FILE, "party_id", c["party_id"]), None)] += 1
            for key in ("name_aliases", "phone_aliases", "email_aliases"):
                expected[(read.CUSTOMERS_FILE, at(read.CUSTOMERS_FILE, "party_id", c["party_id"]), key)] += len(c.get(key, []))
    for c in raw[read.CLAIMS_FILE]:
        if c.get("party_id") == "P9" and "case_id" in c or c.get("case_id") == "CL-3001":
            expected[(read.CLAIMS_FILE, at(read.CLAIMS_FILE, "case_id", c["case_id"]), None)] += 1
            expected[(read.CLAIMS_FILE, at(read.CLAIMS_FILE, "case_id", c["case_id"]), "documents_needed")] += \
                len(c.get("documents_needed", []))
    for p in raw[read.REPRESENTATIVES_FILE]:
        if p.get("buyer_party_id") == "P9" and p.get("rep_name"):
            expected[(read.REPRESENTATIVES_FILE, at(read.REPRESENTATIVES_FILE, "rep_name", p["rep_name"]), None)] += 1
    expected[(read.GUIDANCE_FILE, "#2", "case_type_guidance")] += 1
    topics = [x.get("topic") for x in raw[read.GUIDANCE_FILE]["claim_followup_guidance"]]
    expected[(read.GUIDANCE_FILE, f"#{topics.index('processing_time_after_submission') + 1}", "claim_followup_guidance")] += 1
    expected[(read.GLOSSARY_FILE, "#3", "field_descriptions")] += 1
    expected[(read.CONSENT_FILE, "#2", None)] += 1
    expected = +expected   # drop the zero counts (customers with no aliases of a kind, claims with no documents)

    rows = [x for x in await _report(db) if x[4] == REFUSED]
    assert {x[0] for x in rows} == {"row"}
    assert Counter((f, ref, fld) for _, f, ref, fld, _ in rows) == expected


async def test_a_report_entry_the_database_refuses_leaves_a_fixed_entry(load_dataset):
    db = await load_dataset("starter")
    r = read_folder(STARTER)
    r.report += [ReportEntry("field", read.CLAIMS_FILE, "CL-2048", "summary", "a reason with a nul\x00"),
                 ReportEntry("row", read.CUSTOMERS_FILE, "#1\ud800", None, "a lone surrogate in the ref"),
                 ReportEntry("file", "a file\x00name", None, None, "a nul in the file name"),
                 ReportEntry("field", read.CLAIMS_FILE, "CL-2011", "net_pay", "an entry that is kept")]
    r.claims.append(r.claims[0])   # refused by the database: its entry is written with the rest
    await write(r, db.loader_url)
    rows = await _report(db)
    fixed = "a load-report entry could not be stored"
    assert Counter(rows) == Counter([("field", read.CLAIMS_FILE, None, None, fixed),
                                     ("row", read.CUSTOMERS_FILE, None, None, fixed),
                                     ("file", None, None, None, fixed),
                                     ("field", read.CLAIMS_FILE, "CL-2011", "net_pay", "an entry that is kept"),
                                     ("row", read.CLAIMS_FILE, "#1", None, REFUSED)])
