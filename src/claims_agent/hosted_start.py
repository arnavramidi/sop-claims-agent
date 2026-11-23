"""Start the app on a hosting service that hands over one database address: its administrative account's.

    python -m claims_agent.hosted_start setup   rebuild the database and load the sample data (before each deploy)
    python -m claims_agent.hosted_start serve   run the server

The service gives the address of the database's administrative account (SUPERUSER_DATABASE_URL), and generates the
three accounts' passwords. From those, `setup` builds the loader's address and `serve` builds the server's own
(claims_app): the same host, port and database, another account. Each address is built with psycopg's own quoting,
so a generated password with any character in it is passed exactly.

On such a service the setup step and the server share one environment. So `serve` starts the server as a fresh
program (the same process, replaced) given only the environment it may hold: the administrative address, the loader's,
owner's and demo account's passwords and a claim-change address are not in it, nor anywhere the server process can
read back (a process's original environment stays readable for its whole life, so clearing it in place is not
enough). The server runs with its own account only, as it does under Docker Compose.
"""

import os
import sys
from collections.abc import Mapping

from psycopg.conninfo import conninfo_to_dict, make_conninfo

from claims_agent import hosted

# What only the setup step may hold; the server never sees these.
SETUP_ONLY = ("SUPERUSER_DATABASE_URL", "LOADER_DATABASE_URL", "OWNER_PASSWORD", "LOADER_PASSWORD",
              "DEMO_PASSWORD", "ADMIN_DATABASE_URL")


def account_address(environ: Mapping[str, str], user: str, password_variable: str) -> str:
    """A connection string for one account: the administrative address with that account's name and password."""
    parts = conninfo_to_dict(environ["SUPERUSER_DATABASE_URL"])
    parts.update(user=user, password=environ[password_variable])
    return make_conninfo(**parts)


def setup_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment for the setup step: the loader's address added when it is not given."""
    out = dict(environ)
    if not out.get("LOADER_DATABASE_URL"):
        out["LOADER_DATABASE_URL"] = account_address(environ, "claims_loader", "LOADER_PASSWORD")
    return out


def serve_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment for the server: its own address added when it is not given, and every setup-only value
    removed."""
    out = {k: v for k, v in environ.items() if k not in SETUP_ONLY}
    if not out.get("DATABASE_URL"):
        out["DATABASE_URL"] = account_address(environ, "claims_app", "APP_PASSWORD")
    return out


def serve_command(environ: Mapping[str, str]) -> list[str]:
    """The server's command line (the program and its arguments), run with exactly `environ`."""
    return [sys.executable, "-m", "uvicorn", "claims_agent.api:create_app", "--factory", "--host", "0.0.0.0",
            "--port", str(int(environ.get("PORT") or 8000))]


def _replace_environment(new: Mapping[str, str]) -> None:
    os.environ.clear()
    os.environ.update(new)


def main(argv: list[str]) -> None:
    command = argv[1] if len(argv) == 2 else ""
    if command == "setup":
        from claims_agent import setup
        from claims_agent.settings import load_settings
        if hosted.is_hosted(os.environ):
            hosted.require_sample_data(load_settings(), os.environ)
        _replace_environment(setup_environment(os.environ))
        setup.main()
    elif command == "serve":
        environment = serve_environment(os.environ)
        os.execve(sys.executable, serve_command(environment), environment)
    else:
        raise SystemExit("usage: python -m claims_agent.hosted_start setup|serve")


if __name__ == "__main__":
    main(sys.argv)
