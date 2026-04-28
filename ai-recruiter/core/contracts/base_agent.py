"""BaseAgent — the abstract contract every agent in the platform implements.

An agent owns:
  - a `name` and `version` (class vars),
  - a FastAPI `router` that exposes its public HTTP/WS surface,
  - lifecycle hooks (`startup`, `shutdown`),
  - a `health()` probe,
  - a `card()` returning AgentCard for /.well-known discovery,
  - a `handle_task()` accepting an A2ATask for the orchestrator.

The unified app composes the platform by iterating over registered agents.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

from fastapi import APIRouter

from .a2a import A2ATask, A2ATaskResult, AgentCard
from .health import HealthStatus


class BaseAgent(abc.ABC):
    """Every concrete agent inherits this and is registered in core.runtime.registry."""

    name: ClassVar[str]
    version: ClassVar[str]

    @property
    @abc.abstractmethod
    def router(self) -> APIRouter:
        """The FastAPI router this agent exposes (mounted at /api/<name>)."""

    @abc.abstractmethod
    def card(self) -> AgentCard:
        """Discovery metadata. Should match agent_card.json on disk."""

    @abc.abstractmethod
    async def startup(self, app_state: Any) -> None:
        """Called once at app startup. Heavy model loads stay lazy."""

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Called once at app shutdown. Release GPU memory, close sockets."""

    @abc.abstractmethod
    async def health(self) -> HealthStatus:
        """Non-blocking readiness probe. Must return promptly."""

    @abc.abstractmethod
    async def handle_task(self, task: A2ATask) -> A2ATaskResult:
        """Dispatch a task by capability. Used by the orchestrator stub."""
