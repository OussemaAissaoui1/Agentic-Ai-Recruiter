"""JSON-structured logging with an `agent` field bound on every record.

Usage:
    from core.observability import configure_logging, bind_agent_logger
    configure_logging(level="INFO")
    log = bind_agent_logger("matching")
    log.info("rank_request", n_resumes=12)
"""

from __future__ import annotations

import logging
import sys

try:
    import structlog
except ImportError:  # structlog optional
    structlog = None  # type: ignore


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Set up structlog if available, otherwise stdlib JSON-ish formatter."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            cache_logger_on_first_use=True,
        )
    _CONFIGURED = True


def bind_agent_logger(agent_name: str):
    """Return a logger pre-bound with the agent name field."""
    if not _CONFIGURED:
        configure_logging()
    if structlog is not None:
        return structlog.get_logger().bind(agent=agent_name)
    return logging.getLogger(f"agent.{agent_name}")
