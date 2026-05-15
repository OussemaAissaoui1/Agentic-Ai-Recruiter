"""Agent-to-Agent (A2A) wire contracts.

The transport is in-process today — a registry-based dispatcher in
`core.runtime.registry`. The schemas below are the *frozen* shape every
agent's `handle_task()` accepts and returns, so swapping the transport for
gRPC / Pub/Sub later is a routing change, not a contract change.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Capability(BaseModel):
    """A single capability an agent advertises in its card."""

    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


class AgentCard(BaseModel):
    """Discovery metadata returned at /.well-known/agents/<name>."""

    name: str
    version: str
    description: str
    capabilities: List[Capability] = Field(default_factory=list)
    routes: List[str] = Field(default_factory=list)
    transports: List[str] = Field(default_factory=lambda: ["http"])


class A2ATask(BaseModel):
    """Inbound task envelope. `capability` resolves to one of the agent's cards."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = "orchestrator"
    target: str
    capability: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    deadline_ms: Optional[int] = None
    created_at: float = Field(default_factory=time.time)


class A2ATaskResult(BaseModel):
    """Outbound task result. `success=False` always carries `error`."""

    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
