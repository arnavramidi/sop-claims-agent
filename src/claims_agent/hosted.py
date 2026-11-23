"""Hosted mode (HOSTED=1): the server holds no model key of its own, and each visitor brings theirs.

The visitor's key arrives in the X-Model-Key header of a message request and lives only as long as that request. It
is carried in a `ModelKey`, whose text form is a fixed placeholder, so formatting it, logging it, putting it in an
exception message or dumping it as JSON can never show it. Its value is read in one place only: where the model
client for that request is built. The clients are closed and dropped when the turn ends; nothing that outlives the
request (the database, an event record, a trace, a log line, a background job) ever receives them or the key.

Hosted means strangers, so a hosted server also: serves the bundled sample data only, never prepares replies in the
background (there is no key once the request ends), has no claim-change control, sends nothing to Braintrust unless
its owner opted in, and limits how many conversations and messages one visitor address may start or send a minute.
"""

import logging
import re
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from claims_agent.engine import Deps
from claims_agent.form.extractor import TextExtractor
from claims_agent.form.settle import ModelSettler
from claims_agent.form.switching import build_switching_filler
from claims_agent.models.client import make_client
from claims_agent.reply.wording import ModelWording
from claims_agent.settings import ROOT, Settings

logger = logging.getLogger("claims_agent.hosted")

HEADER = "X-Model-Key"
SAMPLE_DATA = ROOT / "data" / "starter"
ALLOW_OTHER_DATA = "HOSTED_ALLOW_OTHER_DATA"  # "1": serve another data folder in hosted mode (its owner's decision)
BRAINTRUST_OPT_IN = "HOSTED_BRAINTRUST"  # "1": hosted mode may use BRAINTRUST_API_KEY (tracing, reviewed examples)
CLIENT_ADDRESS_HEADER = "CLIENT_ADDRESS_HEADER"  # the header a trusted proxy writes the visitor's address into
_KEY_CHARACTERS = re.compile(r"[!-~]+")  # visible ASCII only: no spaces, control characters or line breaks
_ADDRESS = re.compile(r"[0-9A-Fa-f:.]{1,64}")  # an IPv4 or IPv6 address, as a proxy writes it
_ON = ("1", "true", "yes", "on")


class ModelKey:
    """A visitor's model key for one request. Every text form of it is a fixed placeholder; `reveal()` gives the
    value and is called only where the model client is built. It cannot be pickled or copied."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "ModelKey(<hidden>)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __reduce__(self):
        raise TypeError("a model key cannot be pickled or copied")


class KeyRefused(Exception):
    """The header holds no usable key. `code` is the API's error code; the message is fixed and never the value."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class HostedRefused(RuntimeError):
    """Hosted mode cannot start as configured (the message says why and what to change)."""


def is_hosted(environ: Mapping[str, str]) -> bool:
    return (environ.get("HOSTED") or "").strip().lower() in _ON


def read_key(values: Sequence[str], max_chars: int) -> ModelKey:
    """The key from the header's values. None given, or an empty one: `model_key_needed`. More than one, longer
    than `max_chars`, or anything but visible ASCII (it goes into an outgoing HTTP header): `model_key_invalid`."""
    if not values or not values[0].strip():
        raise KeyRefused("model_key_needed")
    value = values[0].strip()
    if len(values) > 1 or len(value) > max_chars or not _KEY_CHARACTERS.fullmatch(value):
        raise KeyRefused("model_key_invalid")
    return ModelKey(value)


def request_clients(settings: Settings, key: ModelKey) -> tuple[Any, Any, Any, Any, Any]:
    """One model client built from the visitor's key, and the filler, extractor, wording and settler over it."""
    client = make_client(api_key=key.reveal())
    kw = {"retries": settings.limits.model_retries, "timeout": settings.limits.model_timeout_seconds}
    return (client, build_switching_filler(settings, client), TextExtractor(client, settings.models.text, **kw),
            ModelWording(client, settings.models.text, settings.models.larger, **kw),
            ModelSettler(client, settings.models.larger, **kw))


def _over(base: Deps, clients: Sequence[Any]) -> Deps:
    filler, extractor, wording, settler = clients
    return replace(base, filler=filler, extractor=extractor, wording=wording, settler=settler)


def make_request_deps(base: Deps, model_key: ModelKey) -> Deps:
    """The server's dependencies with the model clients built from this request's key (everything else shared).
    The caller owns the new client; `request_deps` is the form that also closes it."""
    return _over(base, request_clients(base.settings, model_key)[1:])


@asynccontextmanager
async def request_deps(base: Deps, model_key: ModelKey) -> AsyncIterator[Deps]:
    """`make_request_deps` for one turn: the client is closed when the turn ends, whatever happens."""
    client, *clients = request_clients(base.settings, model_key)
    try:
        yield _over(base, clients)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()


def lock_down(deps: Deps) -> Deps:
    """The dependencies with background preparation switched off (no key exists once a request has ended)."""
    s = deps.settings
    return replace(deps, settings=s.model_copy(update={
        "prepared_replies": s.prepared_replies.model_copy(update={"enabled": False})}))


def braintrust_key(environ: Mapping[str, str]) -> str | None:
    """BRAINTRUST_API_KEY in hosted mode: used only when the owner also set HOSTED_BRAINTRUST=1."""
    key = environ.get("BRAINTRUST_API_KEY") or None
    if key and (environ.get(BRAINTRUST_OPT_IN) or "").strip().lower() not in _ON:
        logger.info("hosted mode: BRAINTRUST_API_KEY is ignored (set %s=1 to use it)", BRAINTRUST_OPT_IN)
        return None
    return key


def require_sample_data(settings: Settings, environ: Mapping[str, str]) -> None:
    """Refuse to run hosted on anything but the bundled sample data, unless HOSTED_ALLOW_OTHER_DATA=1."""
    folder = (ROOT / settings.data.folder).resolve()
    if folder == SAMPLE_DATA.resolve() or (environ.get(ALLOW_OTHER_DATA) or "").strip() == "1":
        return
    raise HostedRefused(
        f"Hosted mode serves the bundled sample data only ({SAMPLE_DATA.relative_to(ROOT)}), and settings [data] "
        f"folder names another folder. Strangers can talk to a hosted server, so real customer data must never be "
        f"behind it. Point [data] folder back at the sample data, or set {ALLOW_OTHER_DATA}=1 if every person in "
        f"that data agreed to be shown to anyone who visits.")


def visitor_address(headers: Mapping[str, str], peer: str | None, header_name: str | None) -> str:
    """The address a visitor is counted under: the one a trusted proxy wrote into `header_name` (when that is set
    and holds an address), else the connection's own peer address."""
    if header_name:
        written = (headers.get(header_name) or "").strip()
        if _ADDRESS.fullmatch(written):
            return written
    return peer or "unknown"


class RateLimiter:
    """At most `per_minute` events per address in any 60 seconds, in memory. It remembers at most `max_addresses`
    addresses (the least recently seen is forgotten first), so a flood of addresses cannot grow it without bound."""

    def __init__(self, per_minute: int, max_addresses: int, *, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute, self.max_addresses, self.window, self._clock = per_minute, max_addresses, window, clock
        self._seen: OrderedDict[str, deque[float]] = OrderedDict()

    def allow(self, address: str) -> bool:
        now = self._clock()
        times = self._seen.pop(address, None)
        if times is None:
            times = deque()
        while times and times[0] <= now - self.window:
            times.popleft()
        self._seen[address] = times
        while len(self._seen) > self.max_addresses:
            self._seen.popitem(last=False)
        if len(times) >= self.per_minute:
            return False
        times.append(now)
        return True


def lists_sample_callers(settings: Settings, environ: Mapping[str, str]) -> bool:
    """Whether a hosted server may list the sample callers: only while it serves the bundled sample data, and never
    when the override that lets it serve other data is on (then the data is not known to be the sample)."""
    if (environ.get(ALLOW_OTHER_DATA) or "").strip() == "1":
        return False
    return (ROOT / settings.data.folder).resolve() == SAMPLE_DATA.resolve()
