from dataclasses import dataclass
from typing import Protocol

from claims_agent.form.fields import Form, StatedValues
from claims_agent.form.questions import FormRequest


class FormFiller(Protocol):
    name: str

    async def fill(self, req: FormRequest) -> Form: ...


class Extractor(Protocol):
    async def extract(self, message: str, *, want_identity: bool, want_claim_number: bool) -> StatedValues: ...


@dataclass(frozen=True)
class ExtractRequest:
    message: str
    want_identity: bool
    want_claim_number: bool


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

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[FormRequest] = []

    async def fill(self, req: FormRequest) -> Form:
        self.requests.append(req)
        return self._next()


class ScriptedExtractor(_Scripted):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ExtractRequest] = []

    async def extract(self, message: str, *, want_identity: bool, want_claim_number: bool) -> StatedValues:
        self.requests.append(ExtractRequest(message, want_identity, want_claim_number))
        return self._next()
