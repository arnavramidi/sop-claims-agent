"""The server's database access: a pool of claims_app connections, one transaction per use.

The conversation id is set for the transaction only (set_config(..., true)), so it is gone when the
transaction ends and a reused connection carries nothing into the next use. The claims rules read the
verified customer from that conversation's own row.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool


class AppDb:
    def __init__(self, url: str, *, min_size: int = 1, max_size: int = 5) -> None:
        self._pool = AsyncConnectionPool(
            url, min_size=min_size, max_size=max_size, kwargs={"autocommit": True}, open=False
        )
        self._opening = asyncio.Lock()
        self._opened = False

    async def open(self) -> None:
        async with self._opening:
            if not self._opened:
                await self._pool.open(wait=True)
                self._opened = True

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    @asynccontextmanager
    async def conversation(self, conversation_id: UUID | None) -> AsyncIterator[AsyncConnection]:
        await self.open()
        async with self._pool.connection() as conn, conn.transaction():
            if conversation_id is not None:
                await conn.execute("SELECT set_config('app.conversation_id', %s, true)", [str(conversation_id)])
            yield conn
