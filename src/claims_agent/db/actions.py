"""The database actions a turn takes, for the inspector: which action, a coarse outcome, and when it ran.

A function of the database layer is marked with `@action(name, outcome)`. While a turn is running (see
`models.usage.collecting`), each call of a marked function adds one entry to that turn's list; outside a turn
nothing is kept. An entry never holds a value the caller typed, a party id or a claim detail: only the action's
name, the outcome that `outcome` reads from the result (a count, a yes/no, a status word), and its start and end
times. An action that runs inside another marked action is part of it and is not listed again.

Bookkeeping never changes a call: the result (or the error) is passed through unchanged, and any error in the
bookkeeping itself is swallowed.
"""

import contextvars
import functools
import time
from collections.abc import Callable
from typing import Any

from claims_agent.models.usage import running

MAX_ACTIONS = 60  # the most actions one turn's record lists; the rest are only counted

_inside: contextvars.ContextVar[bool] = contextvars.ContextVar("db_action_inside", default=False)


def rows(result: Any) -> dict:
    """The outcome of a lookup that returns rows: how many."""
    return {"rows": len(result)}


def found(result: Any) -> dict:
    return {"found": result is not None}


def answer(result: Any) -> dict:
    """The outcome of a call that answers with a status word or a yes/no."""
    return {"result": result}


def nothing(result: Any) -> dict:
    return {}


def _add(name: str, outcome: dict, started: float, ended: float) -> None:
    try:
        turn = running()
        if turn is None:
            return
        if len(turn.actions) >= MAX_ACTIONS:
            turn.actions_dropped += 1
            return
        turn.actions.append({"action": name, "outcome": outcome, "started": started, "ended": ended})
    except Exception:
        pass


def _read(outcome: Callable[[Any], dict], result: Any) -> dict:
    try:
        out = outcome(result)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def action(name: str, outcome: Callable[[Any], dict] = nothing):
    """Mark an async database function as an action the inspector lists under `name`."""
    def mark(fn):
        @functools.wraps(fn)
        async def run(*args, **kwargs):
            if _inside.get():
                return await fn(*args, **kwargs)
            token = _inside.set(True)
            started = time.time()
            try:
                result = await fn(*args, **kwargs)
            except BaseException as e:
                _add(name, {"error": type(e).__name__}, started, time.time())
                raise
            finally:
                _inside.reset(token)
            _add(name, _read(outcome, result), started, time.time())
            return result
        return run
    return mark


def turn_db_actions() -> dict | None:
    """`{"actions": [...], "dropped": n}` for the running turn, plain JSON; None outside a turn or when the
    bookkeeping failed."""
    try:
        turn = running()
        if turn is None:
            return None
        return {"actions": [dict(a, outcome=dict(a["outcome"])) for a in turn.actions],
                "dropped": turn.actions_dropped}
    except Exception:
        return None
