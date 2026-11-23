"""What a model call can fail with. A message names the kind of failure (exception class, HTTP status) and the
model id, nothing more: never the key, the headers, the request body or the SDK's own text."""


class ModelError(Exception):
    pass


def _msg(what: str, model: str, detail: str) -> str:
    return f"{what}: {model}" + (f" ({detail})" if detail else "")


class ModelUnavailable(ModelError):
    """Timeouts, connection trouble, rate limits, server errors or a bad response, still failing after the retries."""

    def __init__(self, model: str, detail: str = ""):
        super().__init__(_msg("model unavailable", model, detail))


class ModelRequestRejected(ModelUnavailable):
    """The API refused our own request as malformed (400, 422, ...). Not retried; a bug on our side."""

    def __init__(self, model: str, detail: str = ""):
        ModelError.__init__(self, _msg("model request rejected", model, detail))


class ModelKeyRejected(ModelError):
    def __init__(self, model: str, detail: str = ""):
        super().__init__(_msg("model key rejected", model, detail))


class ModelKeyNotSet(ModelKeyRejected):
    """No model key is set at all (nothing was sent to the provider). A kind of rejected key for the code that
    re-raises setup problems, but the API words it on its own."""

    def __init__(self, model: str = "none", detail: str = "OPENAI_API_KEY is not set"):
        super().__init__(model, detail)


class ModelNotPermitted(ModelError):
    def __init__(self, model: str, detail: str = ""):
        super().__init__(_msg("model not permitted for this key", model, detail))


class ModelAccountProblem(ModelError):
    """The model account has no credit or quota. Like a rejected key, it is for the operator to fix: the engine
    must re-raise it, never hand callers over."""

    def __init__(self, model: str, detail: str = ""):
        super().__init__(_msg("model account has no credit or quota", model, detail))
