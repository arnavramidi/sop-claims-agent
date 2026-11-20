"""Turn logging to Braintrust, switched on only by a key in the environment.

With no key nothing is imported, nothing leaves the machine and every call here does nothing. With a key, a failure
of any kind in Braintrust is logged once per process and never reaches the caller: the turn's record is already in
the database. A turn's span is created only when the turn has committed, so no database transaction waits on the
network, and its upload happens on a background thread.

What is sent is masked first: a full ID number (the way the inspector masks it) and every email address, in every
string of the message, the reply, the record and a rating's comment.
"""

import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID, uuid4

from claims_agent.inspect import mask_all, mask_ids
from claims_agent.settings import Settings, Thresholds

logger = logging.getLogger("claims_agent.observability")

NEEDS_REVIEW = "needs_review"
_EMAIL = re.compile(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+")
_MAX_OPEN = 1000  # started turns that never finished (rolled back) are forgotten beyond this many
_reported = False


class TurnLog(Protocol):
    def start(self, conversation_id: UUID, turn_no: int) -> str | None:
        """A span id for the turn about to run, or None when logging is off."""

    def finish(self, span_id: str | None, *, message: str, reply: str, record: dict, tags: Sequence[str] = ()) -> None:
        """Send the committed turn. Call it after the turn's transaction has committed."""

    def feedback(self, span_id: str | None, *, rating: int, comment: str | None) -> None:
        """Attach a rating (the score: 1 for thumbs up, 0 for thumbs down) to a turn's span."""


class NoLog:
    def start(self, conversation_id: UUID, turn_no: int) -> str | None:
        return None

    def finish(self, span_id: str | None, *, message: str, reply: str, record: dict, tags: Sequence[str] = ()) -> None:
        return None

    def feedback(self, span_id: str | None, *, rating: int, comment: str | None) -> None:
        return None


def mask_for_trace(value):
    """Every string inside a JSON-like value with ID numbers and email addresses hidden."""
    return mask_all(_hide_emails(value))


def _hide_emails(value):
    if isinstance(value, str):
        return _EMAIL.sub("[email]", value)
    if isinstance(value, dict):
        return {_hide_emails(k) if isinstance(k, str) else k: _hide_emails(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hide_emails(v) for v in value]
    return value


def _mask_text(text: str) -> str:
    return mask_ids(_EMAIL.sub("[email]", text))


def needs_review(record: dict, t: Thresholds) -> bool:
    """True when a form field fell below its threshold (a yes/no inside the unsure band, a pick under the confident
    line) or a reply check failed: such a turn is worth a person's look."""
    for check in record.get("checks") or []:
        if isinstance(check, dict) and check.get("passed") is False:
            return True
    return _unsure_field(record.get("form"), t)


def _unsure_field(node, t: Thresholds) -> bool:
    if isinstance(node, dict):
        if "options" in node and "p" in node:  # a pick
            return node.get("value") is not None and (node["p"] or 0.0) < t.pick_confident
        if "p" in node:  # a yes/no
            p = node["p"]
            return p is None or t.unsure_low < p < t.unsure_high
        return any(_unsure_field(v, t) for v in node.values())
    if isinstance(node, list):
        return any(_unsure_field(v, t) for v in node)
    return False


def trace_event(*, message: str, reply: str, record: dict, tags: Sequence[str] = (), conversation_id: str | None = None,
                turn_no: int | None = None) -> dict:
    """What is logged for a turn, already masked: the message as input, the reply as output, the record as metadata."""
    metadata = {"record": mask_for_trace(record)}
    if conversation_id is not None:
        metadata["conversation_id"] = conversation_id
    if turn_no is not None:
        metadata["turn_no"] = turn_no
    event: dict[str, Any] = {"input": _mask_text(message), "output": _mask_text(reply), "metadata": metadata}
    if tags:
        event["tags"] = list(tags)
    return event


def _report(error: BaseException) -> None:
    """Log a Braintrust failure once per process; later ones stay quiet."""
    global _reported
    if _reported:
        return
    _reported = True
    logger.warning("Braintrust logging failed (%s: %.200s); turns carry on and later failures are not logged",
                   type(error).__name__, str(error))


class BraintrustLog:
    def __init__(self, project: str, api_key: str) -> None:
        import braintrust  # only when a key is set
        self._braintrust = braintrust
        braintrust.set_masking_function(mask_for_trace)
        self._logger = braintrust.init_logger(project=project, api_key=api_key)
        self._open: OrderedDict[str, tuple[float, str, int]] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, conversation_id: UUID, turn_no: int) -> str | None:
        span_id = str(uuid4())
        with self._lock:
            self._open[span_id] = (time.time(), str(conversation_id), turn_no)
            while len(self._open) > _MAX_OPEN:
                self._open.popitem(last=False)
        return span_id

    def finish(self, span_id: str | None, *, message: str, reply: str, record: dict, tags: Sequence[str] = ()) -> None:
        if span_id is None:
            return
        with self._lock:
            began, conversation_id, turn_no = self._open.pop(span_id, (None, None, None))
        span = self._logger.start_span(name="turn", type="task", id=span_id, start_time=began)
        span.log(**trace_event(message=message, reply=reply, record=record, tags=tags,
                               conversation_id=conversation_id, turn_no=turn_no))
        span.end()
        threading.Thread(target=self._flush, daemon=True, name="braintrust-flush").start()

    def feedback(self, span_id: str | None, *, rating: int, comment: str | None) -> None:
        if span_id is None:
            return
        self._logger.log_feedback(id=span_id, scores={"rating": rating},
                                  comment=None if comment is None else _mask_text(comment))  # comments are not masked by the SDK
        threading.Thread(target=self._flush, daemon=True, name="braintrust-flush").start()

    def _flush(self) -> None:
        try:
            self._logger.flush()
        except Exception as e:
            _report(e)


def make_turn_log(settings: Settings, api_key: str | None) -> TurnLog:
    """The Braintrust log when a key is given, otherwise one that does nothing (and says so once)."""
    if not api_key or not api_key.strip():
        logger.info("Braintrust tracing is off: BRAINTRUST_API_KEY is not set")
        return NoLog()
    try:
        return BraintrustLog(settings.braintrust.project, api_key.strip())
    except Exception as e:
        _report(e)
        logger.info("Braintrust tracing is off: it could not be started")
        return NoLog()


# ---- the calls the engine and the API make: they never raise ------------------------------------------------------

def safe_start(log: TurnLog, conversation_id: UUID, turn_no: int) -> str | None:
    try:
        return log.start(conversation_id, turn_no)
    except Exception as e:
        _report(e)
        return None


def safe_finish(log: TurnLog, span_id: str | None, *, message: str, reply: str, record: dict,
                tags: Sequence[str] = ()) -> None:
    try:
        log.finish(span_id, message=message, reply=reply, record=record, tags=tags)
    except Exception as e:
        _report(e)


def safe_feedback(log: TurnLog, span_id: str | None, *, rating: int, comment: str | None) -> None:
    try:
        log.feedback(span_id, rating=rating, comment=comment)
    except Exception as e:
        _report(e)
