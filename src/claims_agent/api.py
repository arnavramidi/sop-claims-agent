"""The HTTP API that runs the turn engine, and the chat page.

Every response that is not a success is `{"error": <code>, "message": <fixed text>}` and nothing else: the
framework's own validation errors, a missing route or method, the model and database problems an operator must fix,
and any unexpected error. No message ever carries caller text, an SDK's or the database's own words, or a trace.
No route lists conversations, and nothing returns an email address.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from psycopg import AsyncConnection
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.exceptions import HTTPException as StarletteHTTPException

from claims_agent.db import conversations as store
from claims_agent.db.guidance import load_guidance
from claims_agent.db.pool import AppDb
from claims_agent.engine import Deps, TurnFailed, run_turn
from claims_agent.form.decisions_filler import DecisionsFiller
from claims_agent.form.extractor import TextExtractor
from claims_agent.inspect import mask_all
from claims_agent.models.client import make_client
from claims_agent.models.errors import ModelAccountProblem, ModelKeyRejected, ModelNotPermitted
from claims_agent.reply.wording import ModelWording
from claims_agent.settings import Settings, load_settings, load_thresholds

logger = logging.getLogger("claims_agent.api")

WEB = Path(__file__).parent / "web"
MAX_BODY_BYTES = 64 * 1024  # a whole message at the cap, every character escaped, fits with room to spare
RECIPIENT = "the policyholder's email on file"
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                               "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; "
                               "require-trusted-types-for 'script'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# Every error this API can send: code -> (HTTP status, fixed message).
ERRORS: dict[str, tuple[int, str]] = {
    "invalid_request": (422, "The request was not in the expected form."),
    "empty_message": (422, "The message is empty."),
    "unknown_consent_script": (422, "That consent script is not one of the loaded scripts."),
    "message_too_long": (413, "The message is too long."),
    "body_too_large": (413, "The request is too large."),
    "not_found": (404, "There is nothing here."),
    "conversation_not_found": (404, "There is no such conversation."),
    "method_not_allowed": (405, "That method is not allowed here."),
    "model_key_rejected": (401, "The model key was rejected. Check OPENAI_API_KEY."),
    "model_account_problem": (402, "The model account has no credit or quota. Add credit to the account, then try again."),
    "model_not_permitted": (403, "The model key may not use the models this agent needs. Check the key's model access."),
    "turn_failed": (500, "The message could not be handled, and nothing was saved. Try again."),
    "database_busy": (503, "The database is busy. Try again in a moment."),
    "database_unavailable": (503, "The database could not be reached. Check that it is running."),
    "http_error": (400, "The request could not be handled."),
    "internal_error": (500, "Something went wrong on the server."),
}


class ApiError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def error_response(code: str, status: int | None = None, headers: dict | None = None) -> JSONResponse:
    fixed_status, message = ERRORS[code]
    return JSONResponse({"error": code, "message": message}, status_code=status or fixed_status, headers=headers)


# ---- request bodies ---------------------------------------------------------------------------------------------

class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageBody(_Body):
    text: StrictStr


class ControlsBody(_Body):
    consent_script: StrictStr | None = None
    today: StrictStr | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


def _today(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ApiError("invalid_request") from None


# ---- model clients ----------------------------------------------------------------------------------------------

class _NoKey:
    """Stands in for every model client when OPENAI_API_KEY is not set: each call fails as a rejected key, so the
    messages endpoint answers exactly as it does for a key the API refuses, and nothing moves."""
    name = "decisions"

    async def _fail(self, *args, **kwargs):
        raise ModelKeyRejected("none", "OPENAI_API_KEY is not set")

    fill = extract = word = rewrite = _fail


def model_clients(s: Settings) -> tuple[Any, Any, Any]:
    """The filler, extractor and wording clients, all over one client built by `make_client`."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set: every message will be answered with model_key_rejected")
        stand_in = _NoKey()
        return stand_in, stand_in, stand_in
    client = make_client()
    kw = {"retries": s.limits.model_retries, "timeout": s.limits.model_timeout_seconds}
    return (DecisionsFiller(client, s.models.fast, **kw), TextExtractor(client, s.models.text, **kw),
            ModelWording(client, s.models.text, s.models.larger, **kw))


async def _real_deps() -> Deps:
    s = load_settings()
    db = AppDb(os.environ["DATABASE_URL"])
    async with db.conversation(None) as conn:
        guidance = await load_guidance(conn)
    filler, extractor, wording = model_clients(s)
    return Deps(settings=s, thresholds=load_thresholds(), db=db, guidance=guidance, filler=filler,
                extractor=extractor, wording=wording)


# ---- the request body cap ---------------------------------------------------------------------------------------

class BodyLimit:
    """Reads the whole request body before anything else sees it and refuses one over `max_bytes` (by its declared
    length, or by counting as it arrives), then hands the body on unchanged."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            return await error_response("body_too_large")(scope, receive, send)
        chunks, size, more = [], 0, True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                return await error_response("body_too_large")(scope, receive, send)
            chunks.append(body)
            more = message.get("more_body", False)
        whole = b"".join(chunks)
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": whole, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


# ---- error handlers ---------------------------------------------------------------------------------------------

async def _api_error(request: Request, exc: ApiError):
    return error_response(exc.code)


async def _validation(request: Request, exc: RequestValidationError):
    return error_response("invalid_request")


async def _http(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return error_response("not_found")
    if exc.status_code == 405:
        return error_response("method_not_allowed", headers=exc.headers)
    return error_response("http_error", status=exc.status_code if 400 <= exc.status_code < 600 else None)


def _fixed(code: str):
    async def handler(request: Request, exc: Exception):
        return error_response(code)
    return handler


async def _pool_timeout(request: Request, exc: Exception):
    logger.warning("no database connection became free in time")
    return error_response("database_busy")


async def _db_down(request: Request, exc: Exception):
    logger.error("the database could not be reached: %s", type(exc).__name__)
    return error_response("database_unavailable")


async def _unexpected(request: Request, exc: Exception):
    return error_response("internal_error")  # the server logs the traceback; the browser gets none of it


# ---- the app ----------------------------------------------------------------------------------------------------

def _controls(s: Settings, consent_script: str, today: date | None) -> dict:
    return {"consent_script": consent_script, "today": (today or s.today.date).isoformat()}


async def _scripts(conn: AsyncConnection) -> list[str]:
    return [n for (n,) in await (await conn.execute("SELECT name FROM consent_scripts ORDER BY name")).fetchall()]


async def _check_script(conn: AsyncConnection, name: str | None) -> None:
    if name is not None and name not in await _scripts(conn):
        raise ApiError("unknown_consent_script")


async def _row(conn: AsyncConnection, cid: UUID) -> tuple[str, str, date | None]:
    row = await (await conn.execute("SELECT state, consent_script, today FROM conversations WHERE id = %s",
                                    [cid])).fetchone()
    if row is None:
        raise ApiError("conversation_not_found")
    return row


def _page(name: str, media_type: str):
    async def serve():
        return FileResponse(WEB / name, media_type=media_type, headers=SECURITY_HEADERS)
    return serve


def create_app(deps: Deps | None = None) -> FastAPI:
    """The app. With no `deps`, real ones are built at startup from the environment (guidance loaded once)."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = app.state.deps is None
        if owned:
            app.state.deps = await _real_deps()
        try:
            yield
        finally:
            if owned:
                await app.state.deps.db.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None, exception_handlers={
        ApiError: _api_error, RequestValidationError: _validation, StarletteHTTPException: _http,
        ModelKeyRejected: _fixed("model_key_rejected"), ModelAccountProblem: _fixed("model_account_problem"),
        ModelNotPermitted: _fixed("model_not_permitted"), TurnFailed: _fixed("turn_failed"),
        PoolTimeout: _pool_timeout, psycopg.OperationalError: _db_down, Exception: _unexpected,
    })
    app.state.deps = deps
    app.add_middleware(BodyLimit, max_bytes=MAX_BODY_BYTES)

    def d(request: Request) -> Deps:
        return request.app.state.deps

    app.get("/")(_page("index.html", "text/html; charset=utf-8"))
    app.get("/app.js")(_page("app.js", "text/javascript; charset=utf-8"))
    app.get("/styles.css")(_page("styles.css", "text/css; charset=utf-8"))

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.get("/api/meta")
    async def meta(request: Request) -> dict:
        deps = d(request)
        async with deps.db.conversation(None) as conn:
            scripts = await _scripts(conn)
            report = await (await conn.execute(
                "SELECT level, file, ref, field, reason FROM load_report ORDER BY id")).fetchall()
        return {"consent_scripts": scripts,
                "load_report": [dict(zip(("level", "file", "ref", "field", "reason"), r)) for r in report],
                "filler": deps.filler.name}

    @app.post("/api/conversations")
    async def create(request: Request, body: ControlsBody) -> dict:
        deps = d(request)
        today = _today(body.today)
        script = body.consent_script if body.consent_script is not None else "default"
        async with deps.db.conversation(None) as conn:
            await _check_script(conn, body.consent_script)
            cid = await store.create(conn, consent_script=script, today=today)
        return {"conversation_id": str(cid), "state": "VERIFY_ID", "controls": _controls(deps.settings, script, today)}

    @app.get("/api/conversations/{conversation_id}")
    async def get(request: Request, conversation_id: UUID) -> dict:
        deps = d(request)
        async with deps.db.conversation(conversation_id) as conn:
            state, script, today = await _row(conn, conversation_id)
            turns = await store.transcript(conn, conversation_id)
        return {"state": state,
                "turns": [{"turn_no": t["turn_no"], "message": t["message"], "reply": t["reply"],
                           "state_after": t["state_after"], "inspector": t["record"]} for t in turns],
                "controls": _controls(deps.settings, script, today)}

    @app.post("/api/conversations/{conversation_id}/messages")
    async def message(request: Request, conversation_id: UUID, body: MessageBody) -> dict:
        deps = d(request)
        if len(body.text) > deps.settings.limits.message_chars:
            raise ApiError("message_too_long")
        if not body.text.strip():
            raise ApiError("empty_message")
        async with deps.db.conversation(conversation_id) as conn:
            await _row(conn, conversation_id)
        result = await run_turn(deps, conversation_id, body.text)
        return {"turn_id": result.turn_id, "reply": result.reply, "state": result.state.value,
                "inspector": result.inspector}

    @app.patch("/api/conversations/{conversation_id}/controls")
    async def controls(request: Request, conversation_id: UUID, body: ControlsBody) -> dict:
        deps = d(request)
        today = _today(body.today)
        async with deps.db.conversation(conversation_id) as conn:
            await _row(conn, conversation_id)
            await _check_script(conn, body.consent_script)
            await store.set_controls(conn, conversation_id, consent_script=body.consent_script, today=today)
            _, script, today = await _row(conn, conversation_id)
        return {"controls": _controls(deps.settings, script, today)}

    @app.get("/api/conversations/{conversation_id}/outbox")
    async def outbox(request: Request, conversation_id: UUID) -> list[dict]:
        deps = d(request)
        async with deps.db.conversation(conversation_id) as conn:
            await _row(conn, conversation_id)
            rows = await (await conn.execute(
                "SELECT subject, body FROM outbox WHERE conversation_id = %s ORDER BY id", [conversation_id])).fetchall()
        return [mask_all({"subject": sub, "body": text, "recipient": RECIPIENT}) for sub, text in rows]

    return app
