"""The server's database access: a pool of claims_app connections, one transaction per use.

The conversation id is set for the transaction only (set_config(..., true)), so it is gone when the
transaction ends and a reused connection carries nothing into the next use. The claims rules read the
verified customer from that conversation's own row.

Every connection gets a lock wait limit and an idle-in-transaction limit (settings [database]) when it is opened, so
a turn waiting on another turn's lock is answered "busy" instead of hanging, and a transaction left paused is ended
by the database (which frees its locks). A request waits a limited time for a free connection.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from claims_agent.settings import Database


class AppDb:
    def __init__(self, url: str, *, min_size: int = 1, max_size: int | None = None,
                 limits: Database | None = None) -> None:
        self.limits = limits or Database()
        self._pool = AsyncConnectionPool(
            url, min_size=min_size, max_size=max_size or self.limits.pool_size, kwargs={"autocommit": True},
            timeout=self.limits.pool_wait_seconds, configure=self._configure, open=False
        )
        self._opening = asyncio.Lock()
        self._opened = False

    async def _configure(self, conn: AsyncConnection) -> None:
        """Set once per new connection, for the connection's whole life (no account or database setting changes)."""
        await conn.execute("SELECT set_config('lock_timeout', %s, false), "
                           "set_config('idle_in_transaction_session_timeout', %s, false)",
                           [f"{round(self.limits.lock_wait_seconds * 1000)}ms",
                            f"{round(self.limits.idle_in_transaction_seconds * 1000)}ms"])

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

    async def ping(self) -> None:
        """One trivial query through the pool; raises what the pool or the database raises (a busy pool, a database
        that cannot be reached)."""
        await self.open()
        async with self._pool.connection() as conn:
            await conn.execute("SELECT 1")
