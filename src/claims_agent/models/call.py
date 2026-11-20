"""The one place a model call is retried and its failures are sorted into our own errors."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import openai
from pydantic import ValidationError

from claims_agent.models.errors import (ModelAccountProblem, ModelKeyRejected, ModelNotPermitted, ModelRequestRejected,
                                        ModelUnavailable)

T = TypeVar("T")
logger = logging.getLogger("claims_agent.models")


# What the API puts in an error's `type` or `code` when the account is out of credit or quota (read from the
# exception's own fields, never from its message text).
_NO_CREDIT = ("insufficient_quota", "credit_balance_exhausted")


class BadResponse(Exception):
    """A response that arrived but cannot be used (cut off, refused, empty, malformed). Retried like a timeout."""


async def with_retry(call: Callable[[], Awaitable[T]], *, retries: int, timeout: float, model: str = "") -> T:
    """Run `call()` with a time limit. Only what can fail at run time is caught: the SDK's own errors, timeouts,
    a response that fails validation, and BadResponse. A bug in our own code propagates as itself.
    401 -> ModelKeyRejected, 402 or a 429 whose error type/code says credit or quota is exhausted ->
    ModelAccountProblem, 403/404 -> ModelNotPermitted, other 4xx except 408/409/429 -> ModelRequestRejected
    (our request was malformed), all at once. Everything else is retried `retries` times, then ModelUnavailable.
    The error names the exception class, the HTTP status if any, and the model; one warning is logged with the
    same facts. The project error is raised outside any handler, so it carries no context (the original
    error's request and headers could hold the key)."""
    failure: type[Exception] = ModelUnavailable
    detail = ""
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(call(), timeout)
        except openai.APIStatusError as e:
            code = e.status_code
            detail = f"{type(e).__name__} HTTP {code}"
            err_code, err_type = getattr(e, "code", None), getattr(e, "type", None)
            if code == 402 or (code == 429 and (err_code in _NO_CREDIT or err_type in _NO_CREDIT)):
                failure = ModelAccountProblem
                detail += f" code={err_code if err_code in _NO_CREDIT else err_type if err_type in _NO_CREDIT else 'none'}"
            elif code == 401:
                failure = ModelKeyRejected
            elif code in (403, 404):
                failure = ModelNotPermitted
            elif code in (408, 409, 429) or code >= 500:
                continue
            else:
                failure = ModelRequestRejected
            break
        except (asyncio.TimeoutError, openai.OpenAIError, ValidationError, BadResponse) as e:
            detail = type(e).__name__
    level = logging.ERROR if failure in (ModelRequestRejected, ModelAccountProblem) else logging.WARNING
    logger.log(level, "model call failed: model=%s kind=%s outcome=%s", model, detail, failure.__name__)
    raise failure(model, detail)
