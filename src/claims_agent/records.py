"""Event records: what each model was asked and answered and what each rule decided, one row per event, written in
the turn's own transaction and shown in the inspector.

An event holds nothing the inspector does not already show: every string in it passes through the same masking a
trace goes through before it leaves the machine (full ID numbers and email addresses are hidden), and no step puts a
claim detail into an event. Each event's body and the number of events per turn are capped by the settings, so the
table cannot grow without bound.
"""

import json
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from claims_agent.observability.turn_log import mask_for_trace

EVENT_KINDS = ("escalation", "below_threshold", "extraction", "mood_change", "consent_check", "resolved_need",
               "followup_choice", "wrap_up")
CORRECTED = "corrected_next_turn"
TRANSCRIPT_TURNS = 6  # how many earlier turns a transcript carries, besides the turn being recorded
_MIN_TEXT = 24


class Recorder:
    """The events one turn has gathered. They are written by `flush`, inside the turn's transaction."""

    def __init__(self, conversation_id: UUID, turn_no: int, *, body_chars: int = 6000, max_events: int = 40,
                 floor: int = 0) -> None:
        self.conversation_id, self.turn_no = conversation_id, turn_no
        self.body_chars, self.max_events, self.floor = body_chars, max_events, floor
        self.events: list[tuple[str, dict]] = []
        self.current: tuple[str, str] | None = None  # this turn's message and reply, for a transcript
        self.dropped = 0
        self.recent_turns: list[int] = []  # the turn numbers of the earlier messages the form was asked about

    def add(self, kind: str, /, **body: Any) -> None:
        """Keep one event. A `transcript` of None is filled in when the events are written."""
        if kind not in EVENT_KINDS:
            raise ValueError(f"not an event kind: {kind}")
        if len(self.events) >= self.max_events:
            self.dropped += 1
            return
        self.events.append((kind, dict(body)))

    def amend(self, kind: str, match: dict, **changes: Any) -> None:
        """Change the latest event of this kind whose body holds every `match` value (a step records the event when
        it first sees the field and says where the field went when it knows)."""
        for k, body in reversed(self.events):
            if k == kind and all(body.get(a) == b for a, b in match.items()):
                body.update(changes)
                return

    async def _transcript(self, conn: AsyncConnection) -> list[dict]:
        cur = await conn.execute(
            "SELECT message, reply FROM turns WHERE conversation_id = %s AND turn_no >= %s AND turn_no < %s "
            "ORDER BY turn_no DESC, id DESC LIMIT %s",
            [self.conversation_id, self.floor, self.turn_no, TRANSCRIPT_TURNS])
        rows = [{"message": m, "reply": r} for m, r in reversed(await cur.fetchall())]
        if self.current is not None:
            rows.append({"message": self.current[0], "reply": self.current[1]})
        return rows

    async def flush(self, conn: AsyncConnection) -> None:
        """Mask, cap and write the events (and finish their transcripts). `events` then holds what was written."""
        transcript = None
        written: list[tuple[str, dict]] = []
        for kind, body in self.events:
            if "transcript" in body and body["transcript"] is None:
                if transcript is None:
                    transcript = await self._transcript(conn)
                body = {**body, "transcript": transcript}
            written.append((kind, fit(mask_for_trace(body), self.body_chars)))
        self.events = written
        for kind, body in written:
            await conn.execute("INSERT INTO events (conversation_id, turn_no, kind, body) VALUES (%s, %s, %s, %s)",
                               [self.conversation_id, self.turn_no, kind, Jsonb(body)])

    def shown(self) -> list[dict]:
        """The events as the inspector lists them."""
        out = [{"kind": k, "body": body} for k, body in self.events]
        if self.dropped:
            out.append({"kind": "dropped", "body": {"count": self.dropped}})
        return out


def _size(value: Any) -> int:
    return len(json.dumps(value, default=str))


def _shorten(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _shorten(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten(v, limit) for v in value]
    return value


def fit(body: dict, limit: int) -> dict:
    """The body within `limit` characters of JSON: every text cut shorter, then the oldest items of the longest
    list dropped, and failing that nothing but a marker. A body that was cut says `truncated`."""
    if _size(body) <= limit:
        return body
    room = limit - _size({"truncated": True})
    n = limit
    while n > _MIN_TEXT and _size(body) > room:
        n //= 2
        body = _shorten(body, n)
    while _size(body) > room:
        lists = [k for k, v in body.items() if isinstance(v, list) and v]
        if not lists:
            return {"truncated": True}
        longest = max(lists, key=lambda k: _size(body[k]))
        body = {**body, longest: body[longest][1:]}
    return {**body, "truncated": True}


async def flag_corrected(conn: AsyncConnection, conversation_id: UUID, turn_no: int, kind: str) -> None:
    """The caller corrected a detail on a later turn: that turn's extraction of this kind says so."""
    await conn.execute(
        "UPDATE events SET body = jsonb_set(body, '{flags}', COALESCE(body->'flags', '[]'::jsonb) || to_jsonb(%s::text)) "
        "WHERE conversation_id = %s AND turn_no = %s AND kind = 'extraction' AND body->>'kind' = %s "
        "AND NOT COALESCE(body->'flags', '[]'::jsonb) ? %s",
        [CORRECTED, conversation_id, turn_no, kind, CORRECTED])


async def events_of(conn: AsyncConnection, conversation_id: UUID) -> dict[int, list[dict]]:
    """Every event of a conversation as the inspector lists them, by turn number."""
    cur = await conn.execute("SELECT turn_no, kind, body FROM events WHERE conversation_id = %s ORDER BY turn_no, id",
                             [conversation_id])
    out: dict[int, list[dict]] = {}
    for turn_no, kind, body in await cur.fetchall():
        out.setdefault(turn_no, []).append({"kind": kind, "body": body})
    return out


def with_events(record: dict | None, stored: list[dict] | None) -> dict | None:
    """A turn's record as the inspector shows it later: its events as the table holds them now (a later turn may have
    marked an extraction as corrected), keeping the note of events that were dropped for the cap."""
    if not isinstance(record, dict) or "events" not in record:
        return record
    dropped = [e for e in record["events"] if isinstance(e, dict) and e.get("kind") == "dropped"]
    return {**record, "events": [*(stored or []), *dropped]}


def shaped_by(recorder: Recorder, form, request, t) -> list[dict]:
    """The earlier caller messages the fast model said helped make up the need: the turn each came from, its words and
    the probability the model gave. The form asked one question per message, in the order the request listed them."""
    if request is None:
        return []
    messages = request.recent_caller_messages
    out = []
    for i, answer in enumerate(form.shaped_need[: len(messages)]):
        if answer.p is not None and answer.p >= t.yes:
            out.append({"turn": recorder.recent_turns[i] if i < len(recorder.recent_turns) else None,
                        "message": messages[i], "p": answer.p})
    return out
