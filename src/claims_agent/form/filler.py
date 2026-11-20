from dataclasses import dataclass
from typing import Protocol

from claims_agent.form.fields import Form, Pick, StatedValues
from claims_agent.form.questions import FormRequest, Question


class FormFiller(Protocol):
    name: str

    async def fill(self, req: FormRequest) -> Form: ...

    async def fill_one(self, req: FormRequest, question: Question) -> Pick: ...  # one extra question, same context


class Extractor(Protocol):
    async def extract(self, message: str, *, want_identity: bool, want_claim_number: bool,
                      want_parts: bool = False) -> StatedValues: ...


@dataclass(frozen=True)
class ExtractRequest:
    message: str
    want_identity: bool
    want_claim_number: bool
    want_parts: bool = False


class _Scripted:
    def __init__(self) -> None:
        self._items: list = []

    def queue(self, item) -> None:
        self._items.append(item)

    def fail_with(self, error: BaseException) -> None:
        self._items.append(error)

    def _next(self):
        if not self._items:
            raise AssertionError(f"{type(self).__name__} has nothing queued")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class ScriptedFiller(_Scripted):
    name = "scripted"

    def __init__(self, forms=(), one=(), *, name: str | None = None) -> None:
        super().__init__()
        if name is not None:
            self.name = name
        for f in forms:
            self.queue(f)
        self._one: list = list(one)
        self.requests: list[FormRequest] = []
        self.one_questions: list[Question] = []

    def queue_one(self, pick: Pick) -> None:
        self._one.append(pick)

    async def fill(self, req: FormRequest) -> Form:
        self.requests.append(req)
        return self._next()

    async def fill_one(self, req: FormRequest, question: Question) -> Pick:
        self.one_questions.append(question)
        if not self._one:
            return Pick(value=None)
        item = self._one.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class ScriptedExtractor(_Scripted):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ExtractRequest] = []

    async def extract(self, message: str, *, want_identity: bool, want_claim_number: bool,
                      want_parts: bool = False) -> StatedValues:
        self.requests.append(ExtractRequest(message, want_identity, want_claim_number, want_parts))
        return self._next()
