"""JD Generation agent — graph-RAG over a Neo4j employee/role/skill graph
with a rejection-feedback loop.

Public API is materialized as submodules are built. End state (see
`docs/superpowers/specs/2026-05-13-jd-generation-graph-rag-design.md`):

    from agents.jd_generation import JDGenerationAgent      # main.py registry
    from agents.jd_generation import (
        generate_jd, reject_jd, approve_jd, get_jd,
        dump_graph, upload_graph,
    )

Build phases land Python-API entries one at a time; importers should
prefer submodule paths (e.g. `from agents.jd_generation.graph.client import
get_client`) during the build-out.
"""

from __future__ import annotations

__version__ = "0.1.0"
