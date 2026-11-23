"""Token counts of the model calls a turn makes.

Every model client is built by `make_client`, which hands back the SDK client inside `Metered`. `Metered` passes every
call through unchanged and, when the response carries a `usage` block, reports it to the turn that is running (if
there is one). A model call added later is counted without its author doing anything, as long as the client came from
`make_client`. Each call's short job label ("form", "extract", ...) is set by `with_retry(..., job=...)`; a call made
without one is still counted, labelled by the endpoint it used.

A turn collects its calls with `collecting()`. A call made outside a turn (a reply worded ahead in the background after
the turn has ended) finds no collector and is not recorded. Nothing in an entry names a key or any text: only the job,
the model and the counts.

Bookkeeping never changes a call: any error in reading or recording usage is swallowed.
"""

import contextvars
import functools
import inspect
import time
from contextlib import contextmanager
from typing import Any

_COUNTS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")

_turn: contextvars.ContextVar["TurnUsage | None"] = contextvars.ContextVar("turn_usage", default=None)
_job: contextvars.ContextVar[str] = contextvars.ContextVar("model_job", default="")


class TurnUsage:
    """The calls one turn has made, in order."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.actions: list[dict] = []  # the database actions the turn took (db.actions), in order
        self.actions_dropped = 0  # actions past db.actions.MAX_ACTIONS, counted but not kept

    def add(self, job: str, model: str, counts: dict, started: float, ended: float) -> None:
        self.calls.append({"job": job, "model": model, **counts, "started": started, "ended": ended})

    def summary(self) -> dict:
        """`{"calls": [...], "totals": {...}}`, plain JSON."""
        totals = {k: sum(c.get(k, 0) for c in self.calls) for k in _COUNTS}
        return {"calls": [dict(c) for c in self.calls], "totals": totals}


@contextmanager
def collecting():
    """Everything reported inside this block, in this task and the tasks it starts, belongs to one turn."""
    usage = TurnUsage()
    token = _turn.set(usage)
    try:
        yield usage
    finally:
        _turn.reset(token)


def turn_usage() -> dict | None:
    """The running turn's usage so far, or None outside a turn."""
    try:
        usage = _turn.get()
        return None if usage is None else usage.summary()
    except Exception:
        return None


@contextmanager
def job(name: str):
    """Label the model calls made inside this block."""
    token = _job.set(name)
    try:
        yield
    finally:
        _job.reset(token)


def _number(source: Any, name: str) -> int:
    value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _detail(usage: Any, name: str, field: str) -> int:
    inner = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return 0 if inner is None else _number(inner, field)


def read_usage(response: Any) -> dict | None:
    """The counts of a response, or None when it carries no usage. Both the Responses API and the Decisions API are
    read as one shape: `usage.input_tokens`, `usage.output_tokens`, `usage.total_tokens`,
    `usage.input_tokens_details.cached_tokens` and `usage.output_tokens_details.reasoning_tokens` (the shape the
    Decisions test fixture gives; not confirmed against a real Decisions response). A field that is missing counts 0."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    counts = {"input_tokens": _number(usage, "input_tokens"), "output_tokens": _number(usage, "output_tokens"),
              "total_tokens": _number(usage, "total_tokens"),
              "cached_tokens": _detail(usage, "input_tokens_details", "cached_tokens"),
              "reasoning_tokens": _detail(usage, "output_tokens_details", "reasoning_tokens")}
    if not counts["total_tokens"]:
        counts["total_tokens"] = counts["input_tokens"] + counts["output_tokens"]
    return counts


def report(response: Any, *, model: str, endpoint: str, started: float = 0.0, ended: float = 0.0) -> None:
    """Add this response's usage to the running turn. Never raises."""
    try:
        turn = _turn.get()
        if turn is None:
            return
        counts = read_usage(response)
        if counts is None:
            return
        named = getattr(response, "model", None)
        turn.add(_job.get() or endpoint, named if isinstance(named, str) and named else model, counts,
                 started, ended)
    except Exception:
        pass


class Metered:
    """A stand-in for the SDK client (or any part of it) that reports the usage of every call made through it."""

    def __init__(self, target: Any, path: str = "") -> None:
        self._target, self._path = target, path

    @property
    def __class__(self):  # reads as the SDK client it stands in for (isinstance checks keep working)
        return type(self._target)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._target, name)
        path = f"{self._path}.{name}" if self._path else name
        if name.startswith("_"):
            return value
        if callable(value):
            return self._wrap(value, path)
        if isinstance(value, (str, bytes, int, float, bool, type(None), dict, list, tuple)):
            return value
        return Metered(value, path)

    def _wrap(self, fn, path: str):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            started = time.time()
            result = fn(*args, **kwargs)
            if not inspect.isawaitable(result):
                return result
            return self._reported(result, str(kwargs.get("model", "")), path, started)
        return call

    @staticmethod
    async def _reported(pending, model: str, path: str, started: float):
        response = await pending
        report(response, model=model, endpoint=path, started=started, ended=time.time())
        return response


def running() -> TurnUsage | None:
    """The running turn's bookkeeping, or None outside a turn (other per-turn lists, such as the database actions,
    are kept on the same object). Never raises."""
    try:
        return _turn.get()
    except Exception:
        return None
