import asyncio
import os

from claims_agent.db import migrate
from claims_agent.loader.read import read_folder
from claims_agent.loader.write import write
from claims_agent.settings import ROOT, load_settings


def main() -> None:
    """Rebuild the database, then load the data folder into it."""
    migrate.apply(os.environ["SUPERUSER_DATABASE_URL"])
    result = read_folder(ROOT / load_settings().data.folder)
    asyncio.run(write(result, os.environ["LOADER_DATABASE_URL"]))
    print(f"loaded {len(result.customers)} customers, {len(result.claims)} claims; {len(result.report)} load report entries")


if __name__ == "__main__":
    main()
