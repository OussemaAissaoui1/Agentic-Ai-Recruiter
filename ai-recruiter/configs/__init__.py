"""Configuration loaders for the unified app."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # ai-recruiter/

# Load .env at first import of this module — every agent's config.py imports
# load_runtime() before reading os.environ, so this fires early enough.
# find_dotenv walks upward from CWD, picking up a .env placed at the project
# root OR one level up (e.g. /teamspace/studios/this_studio/.env on Lightning
# studios where the project lives in a subdirectory).
# override=False: real environment variables win over .env entries.
_DOTENV_PATH = find_dotenv(usecwd=True) or str(ROOT.parent / ".env")
load_dotenv(_DOTENV_PATH, override=False)


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
