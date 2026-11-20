"""Hosted mode's pure parts: the key holder, the header check, the rate limiter, the start-up guards and the
environments the hosted start-up builds."""

import copy
import json
import logging
import pickle

import httpx
import pytest
from psycopg.conninfo import conninfo_to_dict

from claims_agent import api, hosted, hosted_start
from claims_agent.engine import Deps
from claims_agent.models import client as client_module
from claims_agent.settings import load_settings, load_thresholds

SENTINEL = "sk-test-SENTINEL-93f1"


def test_every_text_form_of_a_key_is_a_placeholder(caplog):
    key = hosted.ModelKey(SENTINEL)
    caplog.set_level(logging.DEBUG)
    logging.getLogger("claims_agent.test").info("key %s %r", key, key)
    shown = [str(key), repr(key), f"{key}", f"{key!r}", f"{key:>40}", "%s" % key, str([key]), str({"k": key}),
             str(ValueError(key)), caplog.text]
    assert all(SENTINEL not in t for t in shown) and key.reveal() == SENTINEL


def test_a_key_cannot_be_dumped_pickled_or_copied():
    key = hosted.ModelKey(SENTINEL)
    with pytest.raises(TypeError):
        json.dumps({"k": key})
    for copier in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            copier(key)
    with pytest.raises(AttributeError):
        key.extra = "x"  # no attribute can be hung on it


@pytest.mark.parametrize("values,code", [
    ([], "model_key_needed"), ([""], "model_key_needed"), (["   "], "model_key_needed"),
    (["sk-a b"], "model_key_invalid"), (["sk-a\nb"], "model_key_invalid"), (["sk-a\r\nX-Other: 1"], "model_key_invalid"),
    (["sk-a\tb"], "model_key_invalid"), (["sk-\x00"], "model_key_invalid"), (["sk-caf\xe9"], "model_key_invalid"),
    (["sk-" + "a" * 510], "model_key_invalid"), (["sk-a", "sk-b"], "model_key_invalid"),
])
def test_a_header_that_holds_no_usable_key_is_refused_with_a_fixed_code(values, code):
    with pytest.raises(hosted.KeyRefused) as e:
        hosted.read_key(values, 512)
    assert e.value.code == code and str(e.value) == code


def test_a_usable_key_is_read_without_its_surrounding_spaces():
    assert hosted.read_key([f"  {SENTINEL} "], 512).reveal() == SENTINEL
    assert hosted.read_key(["sk-" + "a" * 509], 512).reveal() == "sk-" + "a" * 509


def test_hosted_limits_are_settings():
    h = load_settings().hosted
    assert (h.max_key_chars, h.conversations_per_minute, h.messages_per_minute, h.tracked_addresses) == (512, 10, 30, 10000)


async def test_the_request_clients_send_the_key_only_as_the_bearer_header(monkeypatch):
    seen = []

    def make_client(*, api_key=None, http_client=None):
        transport = httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(401, json={"error": {"message": "no"}}))
        return client_module.make_client(api_key=api_key, http_client=httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(hosted, "make_client", make_client)
    client, filler, extractor, wording, settler = hosted.request_clients(load_settings(), hosted.ModelKey(SENTINEL))
    try:
        assert {filler.primary._client, filler.fallback._client, extractor._client, wording._client, settler._client} == {client}
        await client.responses.create(model="m", input="hi")
    except Exception as e:
        assert type(e).__name__ == "AuthenticationError"
    finally:
        await client.close()
    assert len(seen) == 1 and seen[0].headers["authorization"] == f"Bearer {SENTINEL}"
    assert SENTINEL not in str(seen[0].url) and SENTINEL.encode() not in seen[0].content


def test_hosted_mode_switches_prepared_replies_off():
    s = load_settings()
    on = s.model_copy(update={"prepared_replies": s.prepared_replies.model_copy(update={"enabled": True})})
    deps = Deps(settings=on, thresholds=load_thresholds(), db=None, guidance=None, filler=None, extractor=None, wording=None)
    assert hosted.lock_down(deps).settings.prepared_replies.enabled is False
    assert on.prepared_replies.enabled is True  # the original is left as it was


@pytest.mark.parametrize("environ,expected", [
    ({}, None), ({"BRAINTRUST_API_KEY": "bt"}, None), ({"BRAINTRUST_API_KEY": "bt", "HOSTED_BRAINTRUST": "0"}, None),
    ({"BRAINTRUST_API_KEY": "bt", "HOSTED_BRAINTRUST": "1"}, "bt"), ({"HOSTED_BRAINTRUST": "1"}, None),
])
def test_braintrust_is_used_in_hosted_mode_only_when_the_owner_opted_in(environ, expected):
    assert hosted.braintrust_key(environ) == expected


def _deps(folder: str) -> Deps:
    s = load_settings()
    return Deps(settings=s.model_copy(update={"data": s.data.model_copy(update={"folder": folder})}),
                thresholds=load_thresholds(), db=None, guidance=None, filler=None, extractor=None, wording=None)


def test_hosted_mode_refuses_to_start_on_any_data_but_the_sample(monkeypatch):
    monkeypatch.delenv(hosted.ALLOW_OTHER_DATA, raising=False)
    with pytest.raises(hosted.HostedRefused, match="sample data only"):
        api.create_app(_deps("tests/data/other_customers"), hosted_mode=True)
    with pytest.raises(hosted.HostedRefused):
        api.create_app(_deps("data/starter/../real"), hosted_mode=True)
    api.create_app(_deps("data/starter"), hosted_mode=True)
    api.create_app(_deps("data/./starter/"), hosted_mode=True)
    api.create_app(_deps("tests/data/other_customers"), hosted_mode=False)  # only hosted mode is guarded
    monkeypatch.setenv(hosted.ALLOW_OTHER_DATA, "1")
    api.create_app(_deps("tests/data/other_customers"), hosted_mode=True)


@pytest.mark.parametrize("value,on", [("1", True), ("true", True), ("YES", True), ("0", False), ("", False), ("no", False)])
def test_hosted_switch_values(value, on):
    assert hosted.is_hosted({"HOSTED": value}) is on


def test_the_visitor_address_comes_from_the_trusted_header_only_when_named_and_well_formed():
    h = {"cf-connecting-ip": "203.0.113.7"}
    assert hosted.visitor_address(h, "10.0.0.1", None) == "10.0.0.1"
    assert hosted.visitor_address(h, "10.0.0.1", "CF-Connecting-IP".lower()) == "203.0.113.7"
    assert hosted.visitor_address({"cf-connecting-ip": "2001:db8::1"}, "10.0.0.1", "cf-connecting-ip") == "2001:db8::1"
    assert hosted.visitor_address({"cf-connecting-ip": "a" * 65}, "10.0.0.1", "cf-connecting-ip") == "10.0.0.1"
    assert hosted.visitor_address({"cf-connecting-ip": "1.2.3.4, 5.6.7.8"}, "10.0.0.1", "cf-connecting-ip") == "10.0.0.1"
    assert hosted.visitor_address({}, None, "cf-connecting-ip") == "unknown"


def test_the_rate_limiter_counts_a_sliding_minute_per_address():
    now = [0.0]
    limiter = hosted.RateLimiter(2, 100, clock=lambda: now[0])
    assert [limiter.allow("a"), limiter.allow("a"), limiter.allow("a"), limiter.allow("b")] == [True, True, False, True]
    now[0] = 59.0
    assert limiter.allow("a") is False
    now[0] = 60.5
    assert limiter.allow("a") is True and limiter.allow("a") is True and limiter.allow("a") is False


def test_the_rate_limiter_remembers_a_bounded_number_of_addresses():
    limiter = hosted.RateLimiter(1, 3, clock=lambda: 0.0)
    for address in ("a", "b", "c", "d"):
        assert limiter.allow(address)
    assert len(limiter._seen) == 3 and "a" not in limiter._seen
    assert limiter.allow("a") is True  # forgotten, so counted afresh (the bound is on memory, not on fairness)


# ---- the hosted start-up's environments ---------------------------------------------------------------------------

ODD = "a+b/c=d e'f\\g@h:i?j#k"  # every character a URL or a connection string treats specially
PARTS = {"APP_PASSWORD": ODD, "LOADER_PASSWORD": ODD + "L", "OWNER_PASSWORD": "owner",
         "SUPERUSER_DATABASE_URL": "postgresql://admin_x1:p%2Fw@dpg-x:5432/claims_ab12",
         "ADMIN_DATABASE_URL": "postgresql://claims_loader:pw@dpg-x/claims", "HOSTED": "1"}


def test_setup_gets_the_loaders_address_built_from_parts():
    env = hosted_start.setup_environment(PARTS)
    assert conninfo_to_dict(env["LOADER_DATABASE_URL"]) == {
        "host": "dpg-x", "port": "5432", "dbname": "claims_ab12", "user": "claims_loader", "password": ODD + "L"}
    assert env["SUPERUSER_DATABASE_URL"] == PARTS["SUPERUSER_DATABASE_URL"]


def test_the_server_gets_its_own_address_and_none_of_the_setup_only_values():
    env = hosted_start.serve_environment(PARTS)
    assert conninfo_to_dict(env["DATABASE_URL"]) == {
        "host": "dpg-x", "port": "5432", "dbname": "claims_ab12", "user": "claims_app", "password": ODD}
    assert not set(hosted_start.SETUP_ONLY) & set(env)
    assert env["HOSTED"] == "1"


def test_given_addresses_are_kept():
    env = dict(PARTS, DATABASE_URL="postgresql://claims_app:x@h/d", LOADER_DATABASE_URL="postgresql://claims_loader:y@h/d")
    assert hosted_start.serve_environment(env)["DATABASE_URL"] == env["DATABASE_URL"]
    assert hosted_start.setup_environment(env)["LOADER_DATABASE_URL"] == env["LOADER_DATABASE_URL"]


def test_the_start_up_refuses_an_unknown_command():
    with pytest.raises(SystemExit):
        hosted_start.main(["hosted_start", "deploy"])
