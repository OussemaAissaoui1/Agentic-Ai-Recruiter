"""Neo4j graph layer for the JD Generation agent."""

from __future__ import annotations

from .client import Neo4jClient, get_client, close_client
from .schema import apply_schema

__all__ = ["Neo4jClient", "apply_schema", "close_client", "get_client"]
