"""In-process agent registry + A2A dispatcher.

The registry holds every BaseAgent registered at startup. The orchestrator
calls `dispatch(task)` which resolves `task.target` to an agent and invokes
its `handle_task()`. When we move to a real network transport (gRPC / Pub
/Sub) only `dispatch()` changes — agents and contracts stay the same.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List

from core.contracts import A2ATask, A2ATaskResult, AgentCard, BaseAgent, HealthStatus


class AgentNotFound(KeyError):
    """Raised when an A2A task targets an unknown agent."""


class AgentRegistry:
    """Holds the live BaseAgent instances. One per name."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent already registered: {agent.name}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise AgentNotFound(name)
        return self._agents[name]

    def all(self) -> Iterable[BaseAgent]:
        return self._agents.values()

    def names(self) -> List[str]:
        return list(self._agents.keys())

    def cards(self) -> Dict[str, AgentCard]:
        return {a.name: a.card() for a in self._agents.values()}

    async def health_all(self) -> Dict[str, HealthStatus]:
        out: Dict[str, HealthStatus] = {}
        for a in self._agents.values():
            try:
                out[a.name] = await a.health()
            except Exception as e:  # health probes must never crash the app
                out[a.name] = HealthStatus(
                    agent=a.name, state="error", version=a.version, error=str(e)
                )
        return out

    async def dispatch(self, task: A2ATask) -> A2ATaskResult:
        """In-process dispatch. Future: replace body with network call."""
        if task.target not in self._agents:
            return A2ATaskResult(
                task_id=task.task_id,
                success=False,
                error=f"unknown target: {task.target}",
            )
        agent = self._agents[task.target]
        t0 = time.monotonic()
        try:
            result = await agent.handle_task(task)
            result.duration_ms = (time.monotonic() - t0) * 1000.0
            return result
        except Exception as e:
            return A2ATaskResult(
                task_id=task.task_id,
                success=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.monotonic() - t0) * 1000.0,
            )
