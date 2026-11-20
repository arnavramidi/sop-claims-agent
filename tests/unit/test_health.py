from fastapi.testclient import TestClient

from claims_agent.api import create_app


def test_health():
    assert TestClient(create_app()).get("/health").json() == {"ok": True}
