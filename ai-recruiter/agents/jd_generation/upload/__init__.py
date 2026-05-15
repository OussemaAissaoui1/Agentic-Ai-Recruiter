"""Upload handlers for custom graph data."""

from __future__ import annotations

from .parsers import handle_upload
from .schema import UploadPayload

__all__ = ["UploadPayload", "handle_upload"]
