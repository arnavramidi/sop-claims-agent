"""The one way to build the OpenAI client. The SDK's own automatic retries are switched off, so the project's
single retry (models.call.with_retry) is the only one; the HTTP timeout comes from settings."""

from openai import AsyncOpenAI

from claims_agent.settings import load_settings


def make_client(*, api_key: str | None = None, http_client=None) -> AsyncOpenAI:
    """`api_key` defaults to the OPENAI_API_KEY environment variable. `http_client` is for tests only."""
    return AsyncOpenAI(api_key=api_key, max_retries=0, timeout=load_settings().limits.model_timeout_seconds,
                       http_client=http_client)
