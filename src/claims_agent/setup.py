import asyncio
import os

from claims_agent.db import migrate
from claims_agent.loader.read import read_folder
from claims_agent.loader.write import write
from claims_agent.settings import ROOT, Settings, load_settings


def prepare(superuser_url: str, loader_url: str, settings: Settings) -> str:
    """With [data] reset_on_start (the default): rebuild the database and load the data folder, which deletes every
    conversation. Without it: a database built from these migration files is kept as it is, conversations included,
    and only a database with no schema yet is built and loaded. Returns a line saying which happened."""
    if not migrate.apply(superuser_url, rebuild=settings.data.reset_on_start):
        return "kept the database and its conversations as they are (reset_on_start = false)"
    result = read_folder(ROOT / settings.data.folder)
    asyncio.run(write(result, loader_url))
    return f"loaded {len(result.customers)} customers, {len(result.claims)} claims; {len(result.report)} load report entries"


def main() -> None:
    """Prepare the database (see `prepare`)."""
    print(prepare(os.environ["SUPERUSER_DATABASE_URL"], os.environ["LOADER_DATABASE_URL"], load_settings()))


if __name__ == "__main__":
    main()
