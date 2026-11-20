from pydantic import BaseModel

from claims_agent.form.fields import Pick, YesNo
from claims_agent.settings import Thresholds


class YesNoAnswer(BaseModel):
    """A model's answer to a predicate question."""
    probability: float


class PickAnswer(BaseModel):
    """A model's answer to a choice question."""
    choice: str
    probabilities: dict[str, float] = {}
    confidence: float


class Refused(BaseModel):
    """The model declined to answer this question."""


def yes(f: YesNo, t: Thresholds) -> bool:
    return f.p is not None and f.p >= t.yes


def confident_yes(f: YesNo, t: Thresholds) -> bool:
    """Yes at or above the top of the unsure band; None and refused are never confident."""
    return f.p is not None and f.p >= t.unsure_high


def unsure(f: YesNo, t: Thresholds) -> bool:
    return f.p is None or t.unsure_low < f.p < t.unsure_high


def picked(f: Pick, t: Thresholds) -> str | None:
    return f.value if f.value is not None and f.p >= t.pick_confident else None


def top_two(f: Pick) -> list[str]:
    if f.options:
        ranked = sorted(f.options, key=lambda k: f.options[k], reverse=True)
        return ranked[:2]
    return [f.value] if f.value is not None else []
