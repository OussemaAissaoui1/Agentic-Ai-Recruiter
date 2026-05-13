"""Module config loader.

Reads the `jd_generation:` block from configs/runtime.yaml and resolves the
`*_env` indirections against the live environment. The result is a frozen
JDConfig dataclass — startup() checks `neo4j_uri`/`neo4j_password` for None
to decide whether to come up as READY or ERROR.

This mirrors the parent's existing pyyaml + env-var pattern. No
pydantic-settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from configs import load_runtime


@dataclass(frozen=True)
class Neo4jConfig:
    uri: Optional[str]
    user: Optional[str]
    password: Optional[str]
    database: str
    connection_timeout_s: float

    @property
    def is_configured(self) -> bool:
        return bool(self.uri and self.user and self.password)


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    api_key: Optional[str]
    temperature_synthesis: float
    temperature_tool_select: float
    request_timeout_s: float

    @property
    def is_configured(self) -> bool:
        return self.backend == "groq" and bool(self.api_key)


@dataclass(frozen=True)
class AgentLoopConfig:
    max_tool_calls: int
    max_revisions: int
    rejection_lookback: int
    deterministic_preambles: List[str]


@dataclass(frozen=True)
class UploadConfig:
    max_file_size_mb: int
    allowed_extensions: List[str]
    prod_guard: bool


@dataclass(frozen=True)
class JDConfig:
    neo4j: Neo4jConfig
    llm: LLMConfig
    agent: AgentLoopConfig
    upload: UploadConfig
    raw: Dict[str, Any] = field(default_factory=dict)


def load_jd_config() -> JDConfig:
    """Load configs/runtime.yaml + env vars into a JDConfig.

    Returns a config even when env vars are missing; consumers check
    `.neo4j.is_configured` and `.llm.is_configured` before using the agent.
    """
    raw = load_runtime()
    block: Dict[str, Any] = raw.get("jd_generation", {})

    neo4j_block = block.get("neo4j", {})
    neo4j = Neo4jConfig(
        uri=os.environ.get(neo4j_block.get("uri_env", "JD_NEO4J_URI")) or None,
        user=os.environ.get(neo4j_block.get("user_env", "JD_NEO4J_USER")) or None,
        password=os.environ.get(neo4j_block.get("password_env", "JD_NEO4J_PASSWORD")) or None,
        database=neo4j_block.get("database", "neo4j"),
        connection_timeout_s=float(neo4j_block.get("connection_timeout_s", 5)),
    )

    llm_block = block.get("llm", {})
    llm = LLMConfig(
        backend=llm_block.get("backend", "groq"),
        model=llm_block.get("model", "llama-3.3-70b-versatile"),
        api_key=os.environ.get(llm_block.get("api_key_env", "GROQ_API_KEY")) or None,
        temperature_synthesis=float(llm_block.get("temperature_synthesis", 0.7)),
        temperature_tool_select=float(llm_block.get("temperature_tool_select", 0.0)),
        request_timeout_s=float(llm_block.get("request_timeout_s", 60)),
    )

    agent_block = block.get("agent", {})
    agent_cfg = AgentLoopConfig(
        max_tool_calls=int(agent_block.get("max_tool_calls", 15)),
        max_revisions=int(agent_block.get("max_revisions", 3)),
        rejection_lookback=int(agent_block.get("rejection_lookback", 10)),
        deterministic_preambles=list(
            agent_block.get("deterministic_preambles", ["find_role_family", "get_past_rejections"])
        ),
    )

    upload_block = block.get("upload", {})
    prod_guard_env = upload_block.get("prod_guard_env", "JD_PROD_GUARD")
    upload_cfg = UploadConfig(
        max_file_size_mb=int(upload_block.get("max_file_size_mb", 20)),
        allowed_extensions=list(upload_block.get("allowed_extensions", ["json", "csv", "zip"])),
        prod_guard=os.environ.get(prod_guard_env, "0").strip() not in {"", "0", "false", "False"},
    )

    return JDConfig(neo4j=neo4j, llm=llm, agent=agent_cfg, upload=upload_cfg, raw=block)
