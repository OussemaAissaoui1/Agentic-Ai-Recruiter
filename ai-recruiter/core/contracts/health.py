"""Aggregated health contract used by every agent."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthState(str, Enum):
    """Lifecycle state surfaced to the registry and `/api/health`."""

    LOADING = "loading"
    READY = "ready"
    FALLBACK = "fallback"
    ERROR = "error"


class HealthStatus(BaseModel):
    """Per-agent readiness probe result."""

    agent: str
    state: HealthState
    version: str
    fallback_reasons: list[str] = Field(default_factory=list)
    detail: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
