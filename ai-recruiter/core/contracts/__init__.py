"""Frozen wire contracts that every agent in the platform must satisfy."""

from .a2a import A2ATask, A2ATaskResult, AgentCard, Capability
from .base_agent import BaseAgent
from .health import HealthState, HealthStatus

__all__ = [
    "A2ATask",
    "A2ATaskResult",
    "AgentCard",
    "BaseAgent",
    "Capability",
    "HealthState",
    "HealthStatus",
]
