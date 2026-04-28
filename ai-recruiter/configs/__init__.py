"""Configuration loaders for the unified app."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent  # ai-recruiter/


@lru_cache(maxsize=1)
def load_runtime() -> Dict[str, Any]:
    """Load configs/runtime.yaml and apply env-var overrides."""
    cfg_path = Path(__file__).parent / "runtime.yaml"
    with cfg_path.open() as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    server = cfg.setdefault("server", {})
    server["host"] = os.environ.get("AIR_HOST", server.get("host", "0.0.0.0"))
    server["port"] = int(os.environ.get("AIR_PORT", server.get("port", 8000)))
    server["log_level"] = os.environ.get("AIR_LOG_LEVEL", server.get("log_level", "INFO"))

    gpu = cfg.setdefault("gpu_policy", {})
    gpu["vllm_mem_fraction"] = float(
        os.environ.get("AIR_GPU_VLLM_FRACTION", gpu.get("vllm_mem_fraction", 0.55))
    )
    return cfg


def resolve(path: str) -> Path:
    """Resolve a config path relative to the ai-recruiter root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()
