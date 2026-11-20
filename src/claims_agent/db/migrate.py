"""Rebuild the database from db/*.sql.

Order: create the three accounts if missing (as the superuser); close the database to temporary tables
and make every function the owner creates closed unless granted; drop schema `public`; create it again
owned by claims_owner; create extensions (as the superuser, after the schema exists, or the drop would
take them with it); then run the remaining files in name order as claims_owner, so the owner owns every
object. All of it runs in one transaction: a failure leaves the database as it was.

Why the two closures: a temporary table is looked up before the schema's own tables, so an account that
may create one could stand it in for a table a policy reads; and Postgres lets PUBLIC execute every new
function unless the default is changed, so a function that forgot its own REVOKE would be open to all.
"""

import os

import psycopg
from psycopg import sql

from claims_agent.settings import ROOT

SQL_DIR = ROOT / "db"
ROLES_FILE = "001_roles.sql"
ROLE_PASSWORDS = {
    "claims_owner": "OWNER_PASSWORD",
    "claims_loader": "LOADER_PASSWORD",
    "claims_app": "APP_PASSWORD",
}
EXTENSIONS: tuple[str, ...] = ("fuzzystrmatch",)  # levenshtein, for the one-wrong-letter name rule
# No account may create temporary tables in the database being migrated (granted to PUBLIC by default).
CLOSE_TEMPORARY = """
DO $$
BEGIN
  EXECUTE format('REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC, claims_owner, claims_loader, claims_app',
                 current_database());
END $$
"""


def _password(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is not set; copy .env.example to .env")
    return value


def apply(superuser_url: str) -> None:
    files = sorted(p for p in SQL_DIR.glob("*.sql") if p.name != ROLES_FILE)
    with psycopg.connect(superuser_url) as conn, conn.transaction():
        for role, env_name in ROLE_PASSWORDS.items():
            conn.execute("SELECT set_config(%s, %s, true)", [f"claims.{role}_password", _password(env_name)])
        conn.execute((SQL_DIR / ROLES_FILE).read_text())
        conn.execute(CLOSE_TEMPORARY)
        # Owner-wide on purpose: a per-schema REVOKE cannot remove a default PUBLIC grant.
        conn.execute("ALTER DEFAULT PRIVILEGES FOR ROLE claims_owner REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC")
        conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        conn.execute("CREATE SCHEMA public AUTHORIZATION claims_owner")
        for name in EXTENSIONS:
            conn.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS {} SCHEMA public").format(sql.Identifier(name)))
        conn.execute("SET LOCAL ROLE claims_owner")
        for path in files:
            conn.execute(path.read_text())
        conn.execute("RESET ROLE")
