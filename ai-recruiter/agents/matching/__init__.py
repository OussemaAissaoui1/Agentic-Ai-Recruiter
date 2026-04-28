"""Matching agent — pre-interview CV ↔ JD ranking and shortlist."""

# ---------------------------------------------------------------------------
# Pickle-portability shim.
#
# The TAPJFNN checkpoint and LDA `.joblib` files were saved when this package
# was importable as `backend.*` (Paper/cv-ranking-system layout). After the
# integration its real path is `agents.matching.backend.*`. joblib / pickle
# resolves classes by importing their *original* module path, so unpickling
# fails with `No module named 'backend'`.
#
# Fix: register the new package under the old names in sys.modules so the
# pickle resolver finds the same classes via the legacy import path. This
# runs once when `agents.matching` is first imported, before any LDA load.
# ---------------------------------------------------------------------------
import importlib as _importlib
import sys as _sys

_LEGACY_TO_REAL = {
    "backend": "agents.matching.backend",
    "backend.config": "agents.matching.backend.config",
    "backend.utils": "agents.matching.backend.utils",
    "backend.utils.topic_model": "agents.matching.backend.utils.topic_model",
    "backend.utils.pdf_reader": "agents.matching.backend.utils.pdf_reader",
    "backend.utils.ocr": "agents.matching.backend.utils.ocr",
    "backend.data": "agents.matching.backend.data",
    "backend.data.dataset": "agents.matching.backend.data.dataset",
    "backend.data.entities": "agents.matching.backend.data.entities",
    "backend.data.segmentation": "agents.matching.backend.data.segmentation",
    "backend.models": "agents.matching.backend.models",
    "backend.models.encoder": "agents.matching.backend.models.encoder",
    "backend.models.tapjfnn": "agents.matching.backend.models.tapjfnn",
    "backend.models.gnn": "agents.matching.backend.models.gnn",
    "backend.models.ga": "agents.matching.backend.models.ga",
}
for _legacy, _real in _LEGACY_TO_REAL.items():
    if _legacy not in _sys.modules:
        try:
            _sys.modules[_legacy] = _importlib.import_module(_real)
        except Exception:  # noqa: BLE001 — non-fatal; pickle will surface a real error
            pass

from .agent import MatchingAgent  # noqa: E402

__all__ = ["MatchingAgent"]
