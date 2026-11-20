"""One customer with a thousand claims: the caller is never shown or sent a thousand of anything, totals come from
the database, only the caller's own claims count, and the database part of a turn stays quick."""

import json
import re
import statistics
import time
from dataclasses import asdict
from datetime import date

import psycopg
import pytest
from helpers import GAVE, facts, instructions, make_deps, make_form, new_conversation, say, stated
from make_thousand_claims import expected as expected_numbers
from make_thousand_claims import make

from claims_agent.db import claims as claims_module
from claims_agent.db.claims import ClaimFilter, aggregate, count, fetch, overview
from claims_agent.loader.read import read_folder
from claims_agent.loader.write import write
from claims_agent.rules import post_process, resolve_intent
from claims_agent.settings import load_settings
from claims_agent.states import State

TODAY = date(2026, 3, 5)

# Measured on the Compose Postgres: loading the data set 4.7 s, a filtered count 16 ms (median), the first overview on a
# fresh connection 60 ms, the first total 95 ms. The limits are several times that so a busy shared machine passes.
LOAD_SECONDS_LIMIT = 10          # the stated bound: a thousand claims load in under ten seconds
COUNT_SECONDS_LIMIT = 0.2        # the stated bound: a filtered count returns in under 200 milliseconds
OVERVIEW_SECONDS_LIMIT = 0.5
TOTAL_SECONDS_LIMIT = 0.5


@pytest.fixture
def expected():
    return expected_numbers()


class Statements:
    """Every statement the server's connections run while it is installed."""

    def __init__(self):
        self.sql: list[str] = []

    def count_matching(self, text: str) -> int:
        return sum(text in s for s in self.sql)


@pytest.fixture
def statements(monkeypatch):
    seen = Statements()
    original = psycopg.AsyncConnection.execute

    def record(self, query, *args, **kwargs):
        seen.sql.append(query if isinstance(query, str) else str(query))
        return original(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.AsyncConnection, "execute", record)
    return seen


class FetchCalls:
    def __init__(self):
        self.calls: list[object] = []


@pytest.fixture
def spy_fetch(monkeypatch):
    """Records the `limit` of every list of claims a rule asks the database for."""
    spy = FetchCalls()

    def wrap(real):
        async def spied(conn, f, today, limit):
            spy.calls.append(type("Call", (), {"limit": limit}))
            return await real(conn, f, today, limit)
        return spied

    for module in (resolve_intent, post_process):
        monkeypatch.setattr(module, "fetch", wrap(fetch))
    return spy


@pytest.fixture
async def thousand_deps(thousand_db):
    return make_deps(thousand_db)


async def _verify(deps, name, dob, last4):
    cid = await new_conversation(deps)
    await say(deps, cid, f"{name}, born {dob}, ID ends {last4}", make_form(**GAVE),
              stated(policyholder_name=name, date_of_birth=[dob], id_digits=last4))
    return cid


@pytest.fixture
async def iris(thousand_deps):
    return await _verify(thousand_deps, "Iris Novak", "1975-05-05", "1000")


@pytest.fixture
async def iris_conn(thousand_deps, iris):
    async with thousand_deps.db.conversation(iris) as conn:
        yield conn


def _text(r) -> str:
    return " ".join(f.text for f in facts(r))


# ---- what the caller is shown ------------------------------------------------------------------------------------

async def test_all_denied_gets_a_fixed_size_overview_and_no_claim_list_reaches_a_model(thousand_deps, iris, statements):
    r = await say(thousand_deps, iris, "show me all my denied claims",
                  make_form(how_many="all", claim_statuses={"denied": 0.97}))
    assert "overview" in instructions(r) and len([f for f in facts(r) if f.kind != "destination"]) <= 9
    brief = json.dumps(asdict(thousand_deps.wording.briefs[-1]), default=str)
    assert len(set(re.findall(r"CL-\d{5}", brief))) <= 10
    assert all(len(q.listed_claims) <= 5 for q in thousand_deps.filler.requests)
    # the customer's vocabulary, the count of every claim, the filtered count, the overview, and the count of visible
    # claims the turn's record keeps
    assert statements.count_matching("FROM claims") <= 5


async def test_the_overview_is_one_statement(thousand_db, iris_conn, statements, expected):
    o = await overview(iris_conn, ClaimFilter(statuses=frozenset({"denied"})), TODAY)
    assert o.count == expected["denied"] == 120 and len(o.most_recent) == 5
    assert statements.count_matching("FROM claims") == 1
    assert o.by_type == expected["by_type"] and o.by_year == expected["by_year"]


async def test_totals_are_computed_in_the_database(thousand_deps, iris, expected, spy_fetch, statements):
    r = await say(thousand_deps, iris, "how much have you paid me in total?",
                  make_form(calculation="total", figures={"net_pay": 0.95}, how_many="all"))
    assert str(expected["total_net_pay"]) in _text(r) and all(call.limit <= 5 for call in spy_fetch.calls)
    assert len(facts(r)) <= 9
    assert statements.count_matching("sum(") == 1


async def test_a_description_that_matches_hundreds_asks_a_narrowing_question_not_a_list(thousand_deps, iris):
    r = await say(thousand_deps, iris, "my claim", make_form(how_many="one", mentions_claim=True))
    assert "narrow_down" in instructions(r) and "which_claim" not in instructions(r) and len(facts(r)) <= 9


async def test_a_single_claim_asked_for_among_a_thousand_never_lists_more_than_the_limit(thousand_deps, iris):
    limit = load_settings().limits.claims_in_focus
    r = await say(thousand_deps, iris, "my denied dental claims",
                  make_form(how_many="several", claim_types={"dental": 0.95}, claim_statuses={"denied": 0.95}))
    assert len(set(re.findall(r"CL-\d{5}", _text(r)))) <= limit + 5
    assert all(len(q.listed_claims) <= limit for q in thousand_deps.filler.requests)


# ---- only the caller's own claims --------------------------------------------------------------------------------

async def test_another_customers_claims_are_never_counted_listed_or_summed(thousand_deps, thousand_db, iris, expected):
    async with await psycopg.AsyncConnection.connect(thousand_db.loader_url) as c:
        await c.execute("INSERT INTO customers (party_id, name, policy_number, dob, id_type, id_last4, phone, email) "
                        "VALUES ('P2000', 'Tobias Ward', 'POL-2000', '1969-02-02', 'ssn_last4', '2000', "
                        "'+15035552000', 'tobias.ward@example.com')")
        await c.execute("INSERT INTO claims (case_id, party_id, case_type, created_at, status, summary, denial_reason, "
                        "net_pay) SELECT 'CL-2' || lpad(g::text, 4, '0'), 'P2000', 'healthcare', '2026-01-01', "
                        "'denied', 'Healthcare claim denied', 'a reason', 1000 FROM generate_series(1, 200) g")
        await c.commit()
    async with thousand_deps.db.conversation(iris) as conn:
        denied = ClaimFilter(statuses=frozenset({"denied"}))
        assert await count(conn, ClaimFilter(), TODAY) == 1000
        assert await count(conn, denied, TODAY) == expected["denied"]
        assert (await aggregate(conn, ClaimFilter(), TODAY, "net_pay")).total == expected["total_net_pay"]
        assert not [c for c in await fetch(conn, ClaimFilter(), TODAY, 50) if not c.case_id.startswith("CL-1")]
    r = await say(thousand_deps, iris, "how much have you paid me in total?",
                  make_form(calculation="total", figures={"net_pay": 0.95}, how_many="all"))
    assert str(expected["total_net_pay"]) in _text(r) and "CL-2" not in _text(r).replace("CL-1", "")


# ---- time ----------------------------------------------------------------------------------------------------------

async def timed_counts(db, party_id: str, f: ClaimFilter, runs: int) -> list[float]:
    """Seconds one count takes, `runs` times, as the verified customer's conversation."""
    deps = make_deps(db)
    cid = await new_conversation(deps)
    async with await psycopg.AsyncConnection.connect(db.owner_url) as owner:
        await owner.execute("UPDATE conversations SET verified_party_id = %s WHERE id = %s", [party_id, cid])
        await owner.commit()
    took = []
    for _ in range(runs):
        async with deps.db.conversation(cid) as conn:
            started = time.perf_counter()
            await count(conn, f, TODAY)
            took.append(time.perf_counter() - started)
    return took


async def test_load_and_query_stay_within_the_stated_bounds(tmp_path, test_db):
    make(tmp_path)
    started = time.perf_counter()
    await write(read_folder(tmp_path), test_db.loader_url)
    assert time.perf_counter() - started < LOAD_SECONDS_LIMIT
    f = ClaimFilter(types=frozenset({"healthcare"}), statuses=frozenset({"denied"}), years=frozenset({2025}))
    assert statistics.median(await timed_counts(test_db, "P1000", f, runs=5)) < COUNT_SECONDS_LIMIT


async def test_the_overview_and_a_total_are_quick(thousand_db, iris_conn):
    f = ClaimFilter(statuses=frozenset({"denied"}))
    started = time.perf_counter()
    await overview(iris_conn, f, TODAY)
    overview_took = time.perf_counter() - started
    started = time.perf_counter()
    await aggregate(iris_conn, ClaimFilter(), TODAY, "net_pay")
    assert overview_took < OVERVIEW_SECONDS_LIMIT and time.perf_counter() - started < TOTAL_SECONDS_LIMIT
