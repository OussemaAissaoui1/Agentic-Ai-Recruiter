"""Neo4j driver wrapper.

A single process-wide GraphDatabase driver is reused across requests. The
driver itself pools sessions internally — we just wrap it for ergonomics
and to keep the lazy-init pattern out of every caller.

Thread-safety: the neo4j-python driver is thread-safe; multiple FastAPI
workers can share one driver. We synchronize only the initial connect so
two concurrent first-requests don't spin up two drivers.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from neo4j import Driver, GraphDatabase, ManagedTransaction, Result, Session

from agents.jd_generation.config import JDConfig, Neo4jConfig, load_jd_config

log = logging.getLogger("agents.jd_generation.graph")


class Neo4jNotConfigured(RuntimeError):
    """Raised when graph operations are attempted but env vars are missing."""


class Neo4jClient:
    """Thin wrapper over a neo4j.Driver with read/write transaction helpers."""

    def __init__(self, cfg: Neo4jConfig) -> None:
        if not cfg.is_configured:
            raise Neo4jNotConfigured(
                "JD_NEO4J_URI / JD_NEO4J_USER / JD_NEO4J_PASSWORD must be set"
            )
        self._cfg = cfg
        self._driver: Driver = GraphDatabase.driver(
            cfg.uri,
            auth=(cfg.user, cfg.password),
            connection_timeout=cfg.connection_timeout_s,
        )
        log.info("neo4j driver opened uri=%s db=%s", cfg.uri, cfg.database)

    @property
    def database(self) -> str:
        return self._cfg.database

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    def verify(self) -> Dict[str, Any]:
        """Round-trip the driver to confirm credentials work.

        Returns the server's `RETURN 1 AS ok` plus the database name. Raises
        on failure — callers wrap in try/except to surface health state.
        """
        with self._driver.session(database=self._cfg.database) as session:
            record = session.run("RETURN 1 AS ok").single()
        return {"ok": record["ok"] if record else None, "database": self._cfg.database}

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._driver.session(database=self._cfg.database) as s:
            yield s

    def read(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        """Run a read-only query and return list of records as dicts."""

        def _work(tx: ManagedTransaction) -> List[Dict[str, Any]]:
            result: Result = tx.run(cypher, **params)
            return [r.data() for r in result]

        with self.session() as session:
            return session.execute_read(_work)

    def write(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        """Run a write query and return list of records as dicts."""

        def _work(tx: ManagedTransaction) -> List[Dict[str, Any]]:
            result: Result = tx.run(cypher, **params)
            return [r.data() for r in result]

        with self.session() as session:
            return session.execute_write(_work)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_client: Optional[Neo4jClient] = None


def get_client(cfg: Optional[JDConfig] = None) -> Neo4jClient:
    """Return the process-wide Neo4jClient, creating it on first call."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            cfg = cfg or load_jd_config()
            _client = Neo4jClient(cfg.neo4j)
        return _client


def close_client() -> None:
    """Drop the singleton — used at shutdown and in tests."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
            _client = None
