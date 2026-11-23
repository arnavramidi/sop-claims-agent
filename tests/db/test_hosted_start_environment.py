"""The hosted start-up's serving process holds only the server's own secrets: the setup-only values are not in its
environment, including the original one the operating system keeps for the process's whole life."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from claims_agent import hosted_start

pytestmark = pytest.mark.db

SECRETS = {
    "SUPERUSER_DATABASE_URL": "postgresql://admin:superuser-SENTINEL@nowhere/claims",
    "LOADER_DATABASE_URL": "postgresql://claims_loader:loader-url-SENTINEL@nowhere/claims",
    "OWNER_PASSWORD": "owner-SENTINEL",
    "LOADER_PASSWORD": "loader-SENTINEL",
    "DEMO_PASSWORD": "demo-SENTINEL",
    "ADMIN_DATABASE_URL": "postgresql://claims_demo:admin-url-SENTINEL@nowhere/claims",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def test_the_serving_process_environment_holds_no_setup_only_value(test_db):
    # No model or tracing key goes in: a failure message must never be able to show one.
    env = {k: v for k, v in os.environ.items() if k not in hosted_start.SETUP_ONLY and not k.endswith("API_KEY")}
    env.update(SECRETS, DATABASE_URL=test_db.app_url, HOSTED="1", PORT=str(_free_port()))
    proc = subprocess.Popen([sys.executable, "-m", "claims_agent.hosted_start", "serve"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc_dir = Path("/proc") / str(proc.pid)
    try:
        deadline = time.monotonic() + 20
        while b"uvicorn" not in _read(proc_dir / "cmdline"):  # the start-up has replaced itself with the server
            assert proc.poll() is None and time.monotonic() < deadline, "the server did not start"
            time.sleep(0.05)
        served = dict(line.split("=", 1) for line in _read(proc_dir / "environ").decode().split("\0") if "=" in line)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert served["DATABASE_URL"] == test_db.app_url and served["HOSTED"] == "1"
    assert not set(hosted_start.SETUP_ONLY) & set(served)
    assert [k for k, v in served.items() if "SENTINEL" in v] == []
