"""
Avatar Agent — 3D Avatar Lip Sync & Rendering Service

Production-grade agent that drives a Ready Player Me GLB avatar
with real-time lip synchronization from TTS audio output.

Key Responsibilities:
    - Serve the GLB avatar model to the browser
    - Extract viseme timelines from TTS audio chunks
    - Provide SSE viseme streams synchronized with audio
    - Manage avatar state (idle animations, expressions)

Architecture:
    NLP Agent (TTS) ─► Avatar Agent (viseme extraction) ─► Browser (Three.js)

The avatar uses Ready Player Me's ARKit-compatible blend shapes
for lip sync, mapped from audio spectral analysis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("avatar.agent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AVATAR_DIR = Path(__file__).parent
DEFAULT_GLB = AVATAR_DIR / "brunette.glb"


@dataclass
class AvatarConfig:
    """Configuration for the Avatar Agent."""
    glb_path: str = str(DEFAULT_GLB)
    viseme_frame_ms: int = 20
    viseme_smoothing: float = 0.6
    energy_threshold: float = 0.01
    enable_viseme_cache: bool = True
    cache_max_items: int = 100


@dataclass
class AvatarStatus:
    """Runtime status of the Avatar Agent."""
    ready: bool = False
    glb_loaded: bool = False
    glb_file_size_mb: float = 0.0
    viseme_engine_ready: bool = False
    total_viseme_extractions: int = 0
    avg_extraction_ms: float = 0.0
    error: Optional[str] = None


class AvatarAgent:
    """
    Production avatar agent that manages GLB model serving and
    real-time lip sync viseme extraction from audio.

    Usage:
        agent = AvatarAgent(config=AvatarConfig())
        await agent.initialize()

        # Get GLB file path for serving
        glb_path = agent.get_glb_path()

        # Extract visemes from audio
        timeline = agent.extract_visemes(wav_bytes)
    """

    def __init__(self, config: Optional[AvatarConfig] = None):
        self._config = config or AvatarConfig()
        self._viseme_engine = None
        self._status = AvatarStatus()
        self._extraction_times: list = []
        self._glb_path: Optional[Path] = None
        logger.info("AvatarAgent created with config: %s", self._config)

    async def initialize(self) -> None:
        """Initialize the avatar agent: validate GLB, load viseme engine."""
        logger.info("Initializing AvatarAgent...")
        start = time.monotonic()

        try:
            # Validate GLB file
            glb_path = Path(self._config.glb_path)
            if not glb_path.exists():
                raise FileNotFoundError(f"GLB avatar not found: {glb_path}")

            file_size = glb_path.stat().st_size
            self._glb_path = glb_path
            self._status.glb_loaded = True
            self._status.glb_file_size_mb = round(file_size / (1024 * 1024), 2)
            logger.info(
                "GLB avatar loaded: %s (%.2f MB)",
                glb_path.name, self._status.glb_file_size_mb,
            )

            # Initialize viseme engine
            from .viseme_engine import VisemeEngine
            self._viseme_engine = VisemeEngine(
                frame_duration_ms=self._config.viseme_frame_ms,
                smoothing_factor=self._config.viseme_smoothing,
                energy_threshold=self._config.energy_threshold,
            )
            self._status.viseme_engine_ready = True
            logger.info("Viseme engine initialized")

            self._status.ready = True
            elapsed = time.monotonic() - start
            logger.info("AvatarAgent ready in %.2fs", elapsed)

        except Exception as e:
            self._status.error = str(e)
            logger.error("AvatarAgent initialization failed: %s", e, exc_info=True)
            raise

    def get_glb_path(self) -> Path:
        """Return the path to the GLB avatar file."""
        if not self._glb_path:
            raise RuntimeError("AvatarAgent not initialized")
        return self._glb_path

    def extract_visemes(self, wav_bytes: bytes) -> list:
        """
        Extract viseme timeline from WAV audio bytes.

        Args:
            wav_bytes: Complete WAV file as bytes.

        Returns:
            List of viseme frame dicts for JSON serialization.
        """
        if not self._viseme_engine:
            raise RuntimeError("Viseme engine not initialized")

        start = time.monotonic()
        try:
            timeline = self._viseme_engine.extract_from_wav_bytes(wav_bytes)
            result = self._viseme_engine.timeline_to_dict(timeline)

            elapsed_ms = (time.monotonic() - start) * 1000
            self._extraction_times.append(elapsed_ms)
            if len(self._extraction_times) > 100:
                self._extraction_times = self._extraction_times[-100:]

            self._status.total_viseme_extractions += 1
            self._status.avg_extraction_ms = round(
                sum(self._extraction_times) / len(self._extraction_times), 2
            )

            logger.debug(
                "Viseme extraction: %d frames in %.1fms (avg: %.1fms)",
                len(result), elapsed_ms, self._status.avg_extraction_ms,
            )
            return result

        except Exception as e:
            logger.error("Viseme extraction failed: %s", e, exc_info=True)
            raise

    def get_status(self) -> dict:
        """Return current agent status as a dict."""
        return {
            "ready": self._status.ready,
            "glb_loaded": self._status.glb_loaded,
            "glb_file_size_mb": self._status.glb_file_size_mb,
            "viseme_engine_ready": self._status.viseme_engine_ready,
            "total_viseme_extractions": self._status.total_viseme_extractions,
            "avg_extraction_ms": self._status.avg_extraction_ms,
            "error": self._status.error,
        }
