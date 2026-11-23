import psycopg
from fastapi.testclient import TestClient

from claims_agent.api import create_app


class _Unreachable:
    async def ping(self):
        raise psycopg.OperationalError("connection refused")


class _Deps:
    db = _Unreachable()


def test_health_says_when_the_database_cannot_be_reached():
    app = create_app()
    app.state.deps = _Deps()
    r = TestClient(app).get("/health")
    assert r.status_code == 503 and r.json() == {"ok": False, "database": "unavailable"}
