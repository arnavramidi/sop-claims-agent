"""Rebuild the database from db/*.sql.

Order: create the three accounts if missing (as the superuser); close the database to temporary tables
and make every function the owner creates closed unless granted; drop schema `public`; create it again
owned by claims_owner; create extensions (as the superuser, after the schema exists, or the drop would
take them with it); then run the remaining files in name order as claims_owner, so the owner owns every
object. All of it runs in one transaction: a failure leaves the database as it was.

Why the two closures: a temporary table is looked up before the schema's own tables, so an account that
may create one could stand it in for a table a policy reads; and Postgres lets PUBLIC execute every new
function unless the default is changed, so a function that forgot its own REVOKE would be open to all.

On a managed database the administrative account is not a superuser and takes membership of claims_owner to build
the schema (db/001_roles.sql); that membership is revoked again before the transaction commits, so it never outlasts
the migration.

The wrong-value limit (settings: [limits] wrong_values) is fixed into the schema when it is built: it reaches
db/012_identity_lockout.sql as the session setting claims.wrong_values_limit, so the server's account can never change
it. The schema is stamped with a digest of the migration files it was built from and of that limit.
`apply(..., rebuild=False)` keeps a database built from the same files as it is (only the accounts are re-stated),
builds one that has no schema yet, and refuses one built from other files (or with another limit): the files describe a
whole schema, not steps from an older one.
"""

import hashlib
import os

import psycopg
from psycopg import sql

from claims_agent.settings import ROOT, load_settings

SQL_DIR = ROOT / "db"
ROLES_FILE = "001_roles.sql"
ROLE_PASSWORDS = {
    "claims_owner": "OWNER_PASSWORD",
    "claims_loader": "LOADER_PASSWORD",
    "claims_app": "APP_PASSWORD",
    "claims_demo": "DEMO_PASSWORD",
}
OPTIONAL_PASSWORDS = ("DEMO_PASSWORD",)  # unset: the account has no password and cannot log in
STAMP = "claims_agent build "
EXTENSIONS: tuple[str, ...] = ("fuzzystrmatch",)  # levenshtein, for the one-wrong-letter name rule
# No account may create temporary tables in the database being migrated (granted to PUBLIC by default).
CLOSE_TEMPORARY = """
DO $$
BEGIN
  EXECUTE format('REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC, claims_owner, claims_loader, claims_app',
                 current_database());
END $$
"""


class BuiltFromOtherFiles(RuntimeError):
    """The database was built from other migration files than these, and a rebuild was not asked for."""


def _password(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value and env_name not in OPTIONAL_PASSWORDS:
        raise RuntimeError(f"{env_name} is not set; copy .env.example to .env")
    return value or ""


def schema_files() -> list:
    return sorted(p for p in SQL_DIR.glob("*.sql") if p.name != ROLES_FILE)


def wrong_values_limit() -> int:
    """The wrong-value limit the schema is built with: settings.toml [limits] wrong_values."""
    return load_settings().limits.wrong_values


def build_stamp() -> str:
    """A digest of every migration file, roles included, in name order, and of the wrong-value limit."""
    digest = hashlib.sha256()
    for path in sorted(SQL_DIR.glob("*.sql")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    digest.update(f"wrong_values_limit={wrong_values_limit()}".encode())
    return STAMP + digest.hexdigest()


def _stamp_of(conn) -> str | None:
    """The stamp of the schema in this database: None when it holds no table of ours yet."""
    row = conn.execute("SELECT obj_description('public'::regnamespace, 'pg_namespace'), "
                       "to_regclass('public.conversations') IS NOT NULL").fetchone()
    comment, built = row
    if comment and comment.startswith(STAMP):
        return comment
    return "" if built else None


def apply(superuser_url: str, *, rebuild: bool = True) -> bool:
    """Bring the accounts in line and (re)build the schema; True when the schema was built. With `rebuild` False a
    schema built from these same files is kept with its data (False is returned), a missing one is built, and one
    built from other files raises BuiltFromOtherFiles before anything is changed."""
    files = schema_files()
    stamp = build_stamp()
    with psycopg.connect(superuser_url) as conn, conn.transaction():
        if not rebuild:
            found = _stamp_of(conn)
            if found is not None and found != stamp:
                raise BuiltFromOtherFiles(
                    "the database was built from other migration files (or another wrong-value limit) than these; set "
                    "[data] reset_on_start = true for one start to rebuild it (this deletes its conversations)")
        for role, env_name in ROLE_PASSWORDS.items():
            conn.execute("SELECT set_config(%s, %s, true)", [f"claims.{role}_password", _password(env_name)])
        conn.execute((SQL_DIR / ROLES_FILE).read_text())
        if not rebuild and _stamp_of(conn) == stamp:
            _release_owner(conn)
            return False
        conn.execute(CLOSE_TEMPORARY)
        # Owner-wide on purpose: a per-schema REVOKE cannot remove a default PUBLIC grant.
        conn.execute("ALTER DEFAULT PRIVILEGES FOR ROLE claims_owner REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC")
        conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        conn.execute("CREATE SCHEMA public AUTHORIZATION claims_owner")
        for name in EXTENSIONS:
            conn.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS {} SCHEMA public").format(sql.Identifier(name)))
        conn.execute("SELECT set_config('claims.wrong_values_limit', %s, true)", [str(wrong_values_limit())])
        conn.execute("SET LOCAL ROLE claims_owner")
        for path in files:
            conn.execute(path.read_text())
        conn.execute("RESET ROLE")
        conn.execute(sql.SQL("COMMENT ON SCHEMA public IS {}").format(sql.Literal(stamp)))
        _release_owner(conn)
    return True


def _release_owner(conn) -> None:
    """Without superuser rights the roles file made this account a member of claims_owner to build the schema as it;
    that membership ends here, inside the same transaction, so it never outlasts the migration."""
    if not conn.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()[0]:
        conn.execute(sql.SQL("REVOKE claims_owner FROM {}").format(sql.Identifier(
            conn.execute("SELECT current_user").fetchone()[0])))
