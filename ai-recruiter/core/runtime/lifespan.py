"""FastAPI lifespan that orchestrates ordered startup/shutdown across agents.

Heavy GPU loads stay lazy in their adapters; this just calls the agents'
`startup` (cheap init) and `shutdown` (release) hooks in a known order.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncIterator, List

from fastapi import FastAPI

from .registry import AgentRegistry

logger = logging.getLogger("runtime.lifespan")


def build_lifespan(registry: AgentRegistry, startup_order: List[str] | None = None):
    """Return a FastAPI lifespan context manager.

    `startup_order` controls the explicit order; missing names are appended.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Resolve order
        names = list(registry.names())
        ordered = []
        if startup_order:
            for n in startup_order:
                if n in names:
                    ordered.append(n)
                    names.remove(n)
        ordered.extend(names)

        app.state.registry = registry
        logger.info("Lifespan: starting agents in order: %s", ordered)

        # Cheap startups in parallel — heavy work happens lazily on first request
        async def _start_one(name: str):
            try:
                await registry.get(name).startup(app.state)
                logger.info("agent.startup ok: %s", name)
            except Exception as e:
                logger.exception("agent.startup failed: %s — %s", name, e)

        await asyncio.gather(*(_start_one(n) for n in ordered))

        try:
            yield
        finally:
            logger.info("Lifespan: shutting down agents…")
            for name in reversed(ordered):
                try:
                    await registry.get(name).shutdown()
                    logger.info("agent.shutdown ok: %s", name)
                except Exception as e:
                    logger.warning("agent.shutdown error %s: %s", name, e)

    return lifespan
