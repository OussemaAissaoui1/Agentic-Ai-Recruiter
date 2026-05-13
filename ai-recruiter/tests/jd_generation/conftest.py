"""Shared fixtures for jd_generation tests.

The integration fixtures require live Neo4j + Groq credentials and are
skipped unless `pytest -m integration` is run.
"""

from __future__ import annotations

import os

import pytest

from agents.jd_generation.config import JDConfig, load_jd_config


def _neo4j_ready() -> bool:
    return all(os.environ.get(k) for k in ("JD_NEO4J_URI", "JD_NEO4J_USER", "JD_NEO4J_PASSWORD"))


def _groq_ready() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


@pytest.fixture(scope="session")
def jd_config() -> JDConfig:
    return load_jd_config()


@pytest.fixture(scope="session")
def neo4j_available() -> bool:
    return _neo4j_ready()


@pytest.fixture(scope="session")
def groq_available() -> bool:
    return _groq_ready()


@pytest.fixture
def requires_neo4j(neo4j_available: bool) -> None:
    if not neo4j_available:
        pytest.skip("JD_NEO4J_URI / USER / PASSWORD not set")


@pytest.fixture
def requires_groq(groq_available: bool) -> None:
    if not groq_available:
        pytest.skip("GROQ_API_KEY not set")
