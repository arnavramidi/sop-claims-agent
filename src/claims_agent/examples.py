"""Reviewed example phrasings for the form's options.

A person reviews turns that read badly and approves the caller phrasings worth keeping, as rows of `field`, `option`
and `message`: in evals/approved_examples.json, and (when a Braintrust key is set) in the Braintrust dataset
`approved-examples`. They are read once when the app starts and shown to the form-filling model after the data's own
example phrases of that option, as quoted example phrasings. Nothing a caller types becomes an example on its own.

Every example is untrusted text, since it ends up inside what the form-filling model reads: only a known field and
one of that field's options is accepted, the message must be short, ordinary message text (letters, digits, spaces
and everyday punctuation; no line break, control character, quote, colon, semicolon, web or email address, or long
number), and there are caps per option and overall. A remote read that fails, hangs or sends nonsense is logged and
the file's examples are used: it never stops the app starting, and turns only ever see the examples read at start.
"""

import json
import logging
import re
import threading
import unicodedata
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claims_agent.settings import ROOT, Settings

logger = logging.getLogger("claims_agent.examples")

REMOTE_DATASET = "approved-examples"
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PUNCTUATION = set(" .,?!'’-()&%$£€/#+¿¡，。？！、")  # with ¿ ¡ ， 。 ？ ！ 、
_WEB = re.compile(r"(?i)https?|://|\bwww\b|[a-z0-9-]\.[a-z]{2,}\b")
_LONG_NUMBER = re.compile(r"\d{5,}")


@dataclass(frozen=True)
class Example:
    field: str
    option: str
    message: str


# ---- reading -------------------------------------------------------------------------------------------------------

def _example(raw: Any) -> Example | None:
    if isinstance(raw, Example):
        raw = {"field": raw.field, "option": raw.option, "message": raw.message}
    if not isinstance(raw, dict):
        return None
    field, option, message = raw.get("field"), raw.get("option"), raw.get("message")
    if not all(isinstance(v, str) for v in (field, option, message)):
        return None
    return Example(field, option, message)


def load_local(path: Path) -> list[Example]:
    """The examples in the file (a JSON list of {field, option, message}); a missing file is none, an unreadable one
    is none and logged, and a row of the wrong shape is skipped and counted in the log."""
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("approved examples file could not be read (%s); none are used from it", type(e).__name__)
        return []
    if not isinstance(rows, list):
        logger.warning("approved examples file is not a list; none are used from it")
        return []
    out = [e for e in (_example(r) for r in rows) if e is not None]
    if len(out) < len(rows):
        logger.warning("approved examples file: %d row(s) skipped (not a {field, option, message} record)",
                       len(rows) - len(out))
    return out


def rows_to_examples(rows: Iterable[Any], *, limit: int = 500) -> list[Example]:
    """Dataset rows (input {field, message}, expected option) as examples; any other shape is skipped. At most
    `limit` rows are read."""
    out: list[Example] = []
    skipped = 0
    for i, row in enumerate(rows):
        if i >= limit:
            logger.warning("approved examples dataset: only the first %d rows are read", limit)
            break
        inp = row.get("input") if isinstance(row, dict) else None
        expected = row.get("expected") if isinstance(row, dict) else None
        example = _example({**inp, "option": expected}) if isinstance(inp, dict) else None
        if example is None:
            skipped += 1
        else:
            out.append(example)
    if skipped:
        logger.warning("approved examples dataset: %d row(s) skipped (not input {field, message} with an expected option)",
                       skipped)
    return out


def load_remote(project: str, dataset: str = REMOTE_DATASET) -> list[Example]:
    """The examples in the Braintrust dataset (the key comes from BRAINTRUST_API_KEY). May raise or take long: the
    caller bounds both."""
    import braintrust  # only when a key is set

    return rows_to_examples(braintrust.init_dataset(project=project, name=dataset))


def _remote_within(settings: Settings) -> list[Example]:
    """The remote examples, or none when the read fails or takes longer than the limit (the read is left to finish
    on its own in the background; nothing waits for it)."""
    limit = settings.examples.remote_timeout_seconds
    box: dict[str, Any] = {}

    def read() -> None:
        try:
            box["examples"] = load_remote(settings.braintrust.project, settings.examples.dataset)
        except BaseException as e:  # noqa: BLE001  reported below, never raised into the app's start
            box["error"] = e

    worker = threading.Thread(target=read, daemon=True, name="approved-examples")
    worker.start()
    worker.join(limit)
    if worker.is_alive():
        logger.warning("approved examples dataset took too long (over %s s); only the file's examples are used", limit)
        return []
    if "error" in box:
        logger.warning("approved examples dataset could not be read (%s); only the file's examples are used",
                       type(box["error"]).__name__)
        return []
    got = box.get("examples")
    return list(got) if isinstance(got, list) else []


# ---- checking ------------------------------------------------------------------------------------------------------

def ordinary_text(message: str, max_chars: int) -> bool:
    """Short, ordinary message text: letters (any language), digits, spaces and everyday punctuation, with at least
    one letter; no web address and no long number."""
    if not message or len(message) > max_chars or not message.strip():
        return False
    for ch in message:
        if not (ch.isalnum() or ch in _PUNCTUATION) or unicodedata.category(ch).startswith("C"):
            return False
    if not any(ch.isalpha() for ch in message):
        return False
    return not _WEB.search(message) and not _LONG_NUMBER.search(message)


def _key(message: str) -> str:
    return " ".join(message.split()).casefold()


def checked(candidates: Iterable[Any], settings: Settings, known_fields: Collection[str],
            known_options: Mapping[str, Collection[str]] | None = None) -> dict[str, tuple[str, ...]]:
    """The candidates that pass every check, keyed "<field>:<option>", first seen first, duplicates dropped, capped."""
    cfg = settings.examples
    out: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    total = 0
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for raw in candidates:
        e = _example(raw)
        if e is None:
            reject("not an example")
            continue
        if not _NAME.fullmatch(e.field) or not _NAME.fullmatch(e.option):
            reject("not a plain field or option name")
            continue
        if e.field not in known_fields or (known_options is not None and e.option not in known_options.get(e.field, ())):
            reject("unknown field or option")
            continue
        if not ordinary_text(e.message, cfg.message_chars):
            reject("not short ordinary message text")
            continue
        message = " ".join(e.message.split())
        mark = (e.field, e.option, _key(message))
        if mark in seen:
            continue
        seen.add(mark)
        key = f"{e.field}:{e.option}"
        if len(out.get(key, ())) >= cfg.per_option or total >= cfg.total:
            reject("over the cap")
            continue
        out.setdefault(key, []).append(message)
        total += 1
    for reason, count in rejected.items():
        logger.warning("approved examples: %d left out (%s)", count, reason)
    return {k: tuple(v) for k, v in out.items()}


def load_examples(settings: Settings, api_key: str | None, known_fields: Collection[str],
                  known_options: Mapping[str, Collection[str]] | None = None) -> dict[str, tuple[str, ...]]:
    """The approved examples, keyed "<field>:<option>" (the shape FormRequest.examples takes): the file's, then the
    Braintrust dataset's when a key is set. Never raises for anything the file or the network does."""
    path = Path(settings.examples.path)
    local = load_local(path if path.is_absolute() else ROOT / path)
    remote = _remote_within(settings) if api_key and api_key.strip() else []
    return checked([*local, *remote], settings, known_fields, known_options)


def with_approved(own: Mapping[str, tuple[str, ...]],
                  approved: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """The data's own example phrases of each option, then the approved ones that repeat none of them."""
    out = {k: tuple(v) for k, v in own.items()}
    for key, messages in approved.items():
        have = out.get(key, ())
        known = {_key(m) for m in have}
        extra = tuple(m for m in messages if _key(m) not in known)
        if extra:
            out[key] = have + extra
    return out


# ---- what the app reads at start -----------------------------------------------------------------------------------

def known_choices(needs: Iterable[str], relationships: Iterable[str], answers: Iterable[str]) -> dict[str, set[str]]:
    """Every choice question the form asks, with every option it can offer for this data: asked before
    verification (who the caller is) and after it, in the state that asks the most. A listed claim's options are claim
    numbers and a document's are document names, not phrasings, so neither takes examples."""
    from claims_agent.form.questions import FormRequest, build_questions
    from claims_agent.states import State

    vocabulary = {"needs": tuple(needs), "relationships": tuple(relationships),
                  "answers": tuple((a, a) for a in answers)}
    requests = (FormRequest(message="-", **vocabulary),
                FormRequest(message="-", state=State.POST_PROCESS, verified=True, **vocabulary))
    known: dict[str, set[str]] = {}
    for q in (q for r in requests for q in build_questions(r)):
        if q.type == "choice" and q.name not in ("listed_claim", "which_document"):
            known.setdefault(q.name, set()).update(v for v, _ in q.choices)
    return known


async def startup_examples(settings: Settings, api_key: str | None, db) -> dict[str, tuple[str, ...]]:
    """The approved examples for the app, checked against the choices the form asks for the loaded data. Any failure
    here leaves the app with no approved examples."""
    import asyncio

    from claims_agent.engine import ANSWER_KINDS, _followups, _vocabulary

    try:
        async with db.conversation(None) as conn:
            topics = [topic for topic, _ in await _followups(conn)]
            known = known_choices(await _vocabulary(conn, "need"), await _vocabulary(conn, "relationship"),
                                  (*ANSWER_KINDS, *topics))
        return await asyncio.to_thread(load_examples, settings, api_key, set(known), known)
    except Exception as e:  # noqa: BLE001  the app starts without approved examples
        logger.warning("approved examples were not loaded (%s); the app starts without them", type(e).__name__)
        return {}


# ---- what may be uploaded ------------------------------------------------------------------------------------------

def for_upload(row: dict) -> dict | None:
    """A row to send to Braintrust, with ID numbers and email addresses masked the way traces are; None (refused)
    when something address-like is still in it after masking."""
    from claims_agent.observability.turn_log import mask_for_trace

    masked = mask_for_trace(row)
    text = json.dumps(masked, ensure_ascii=False)
    if "@" in text or re.search(r"\d{9,}", text):
        return None
    return masked
