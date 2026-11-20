"""The guidance data (document advice, follow-ups, settings, glossary) as one value the answer code reads.

`load_guidance` reads the `guidance` and `vocabulary` tables; `guidance_from` builds the same value from a
loader result without a database (for unit tests). Both go through `_assemble`, so they cannot drift apart."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection

from claims_agent.loader import read
from claims_agent.loader.read import FollowupRow, LoadResult


@dataclass(frozen=True)
class Guidance:
    default_text: str
    case_type_text: Mapping[str, str]
    document_text: Mapping[str, str]
    alternative_text: Mapping[str, str]
    alternative_default: str
    followups: tuple[FollowupRow, ...]
    followup_settings: Mapping[str, str]
    glossary: Mapping[str, str]
    vocabulary: Mapping[str, tuple[str, ...]]


def _en(body: Any, field: str = "en") -> str | None:
    value = body.get(field) if isinstance(body, dict) else None
    return value if isinstance(value, str) else None


def _assemble(rows: Iterable[tuple[str, str | None, Any]], followups: Iterable[FollowupRow],
              vocab: Mapping[str, Iterable[str]]) -> Guidance:
    by_kind: dict[str, dict[str | None, str]] = {}
    for kind, key, body in rows:
        # A glossary entry is the schema file's own object: its wording is the "description" field.
        text = _en(body, "description" if kind == "field_description" else "en")
        if text is not None:
            by_kind.setdefault(kind, {})[key] = text
    alternatives = {k: v for k, v in by_kind.get("document_alternative_guidance", {}).items() if k is not None}
    default_alt = alternatives.pop("default", "")
    only = lambda kind: {k: v for k, v in by_kind.get(kind, {}).items() if k is not None}  # noqa: E731
    return Guidance(
        default_text=by_kind.get("default_guidance", {}).get(None, ""),
        case_type_text=only("case_type_guidance"),
        document_text=only("document_guidance"),
        alternative_text=alternatives,
        alternative_default=default_alt,
        followups=tuple(sorted(followups, key=lambda f: f.position)),
        followup_settings=only(read.SETTINGS_KIND),
        glossary=only("field_description"),
        vocabulary={k: tuple(sorted(v)) for k, v in vocab.items()})


def guidance_from(result: LoadResult) -> Guidance:
    return _assemble(((g.kind, g.key, g.body) for g in result.guidance), result.followups, result.vocabulary)


async def load_guidance(conn: AsyncConnection) -> Guidance:
    rows = await (await conn.execute("SELECT kind, key, body FROM guidance")).fetchall()
    words = await (await conn.execute("SELECT kind, value FROM vocabulary")).fetchall()
    followups = []
    plain = []
    for kind, key, body in rows:
        if kind == "followup":
            followups.append(FollowupRow(
                key, tuple(body.get("needs", ())), body.get("requires_documents") is True,
                tuple(body.get("examples", ())), body.get("template", ""), int(body.get("position", 0)),
                body.get("enabled") is True))
        else:
            plain.append((kind, key, body))
    vocab: dict[str, set[str]] = {}
    for kind, value in words:
        vocab.setdefault(kind, set()).add(value)
    return _assemble(plain, followups, vocab)
