"""Runtime composition: agent registry, lifespan, dispatcher."""

from .lifespan import build_lifespan
from .registry import AgentRegistry

__all__ = ["AgentRegistry", "build_lifespan"]
