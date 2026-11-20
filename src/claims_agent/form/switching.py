"""The filler the app uses: Decisions first, and for the rest of the process the Structured Outputs filler once a key
is refused the Decisions API. FORM_FILLER = auto (default) | decisions | structured."""

import logging
import os

from claims_agent.form.decisions_filler import DecisionsFiller
from claims_agent.form.fields import Form, Pick
from claims_agent.form.filler import FormFiller
from claims_agent.form.questions import FormRequest, Question
from claims_agent.form.structured_filler import StructuredFiller
from claims_agent.models.errors import ModelNotPermitted

logger = logging.getLogger("claims_agent.models")


class SwitchingFiller:
    def __init__(self, primary: FormFiller, fallback: FormFiller):
        self.primary, self.fallback = primary, fallback
        self._switched = False

    @property
    def name(self) -> str:
        return (self.fallback if self._switched else self.primary).name

    async def fill(self, req: FormRequest) -> Form:
        if not self._switched:
            try:
                return await self.primary.fill(req)
            except ModelNotPermitted:
                self._switch()
        return await self.fallback.fill(req)

    async def fill_one(self, req: FormRequest, question: Question) -> Pick:
        if not self._switched:
            try:
                return await self.primary.fill_one(req, question)
            except ModelNotPermitted:
                self._switch()
        return await self.fallback.fill_one(req, question)

    def _switch(self) -> None:
        self._switched = True
        logger.warning("form filler switched from %s to %s: the key is not permitted the first", self.primary.name, self.fallback.name)


def build_switching_filler(settings, client) -> FormFiller:
    """The filler for the app, from the settings and a client built by make_client. FORM_FILLER picks: auto
    (Decisions, switching to Structured Outputs when refused), decisions, or structured."""
    kw = dict(retries=settings.limits.model_retries, timeout=settings.limits.model_timeout_seconds)
    decisions = DecisionsFiller(client, settings.models.fast, **kw)
    structured = StructuredFiller(client, settings.models.text, **kw)
    mode = os.environ.get("FORM_FILLER", "auto").strip().lower() or "auto"
    if mode == "decisions":
        return decisions
    if mode == "structured":
        return structured
    if mode != "auto":
        raise ValueError(f"FORM_FILLER must be auto, decisions or structured (got {mode!r})")
    return SwitchingFiller(decisions, structured)


def active_filler_name(filler: FormFiller) -> str:
    """Which filler is active now (the switching filler's current one, or the plain filler's own name)."""
    return filler.name
