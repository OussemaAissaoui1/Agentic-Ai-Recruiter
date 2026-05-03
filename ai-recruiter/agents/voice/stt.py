"""Server-side STT — Moonshine + Silero VAD.

Lazy-loaded singleton. The Moonshine model + Silero VAD are heavy enough that
we don't want them in the main FastAPI startup path; they load on the first
/api/transcribe or /ws/transcribe call.

Public API
----------
get_stt() -> SttBackend                 # idempotent
SttBackend.transcribe_wav(b64) -> dict  # {text, segments, language}
SttBackend.create_listener() -> WSListener
SttBackend.create_vad_iterator() -> VADIterator | None

The WS handler in main.py uses these directly — separating the Moonshine /
Silero plumbing from the route logic so we can unit-test each piece.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("voice.stt")

_lock = threading.Lock()
_singleton: Optional["SttBackend"] = None


def pcm_int16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit signed mono PCM bytes → float32 in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV blob → (mono float32 array, sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        # 8-bit and unusual widths — best-effort fallback
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels == 2:
        samples = samples[::2]  # take left channel only

    return samples, sr


@dataclass
class SttBackend:
    """Bundles the loaded Moonshine + Silero handles.

    Construction is expensive (~5–10 s first call). Use module-level
    `get_stt()` so we only pay it once per process.
    """

    transcriber: object               # moonshine_voice.Transcriber
    transcript_event_listener: type   # moonshine_voice.TranscriptEventListener
    vad_model: Optional[object]       # silero VAD torch module (may be None)

    def transcribe_wav(self, audio_b64: str, language: str = "en") -> dict:
        if not audio_b64:
            return {"text": "", "segments": [], "language": language}
        audio_bytes = base64.b64decode(audio_b64)
        if len(audio_bytes) < 100:
            return {"text": "", "segments": [], "language": language}

        audio_float, sr = wav_bytes_to_float32(audio_bytes)
        transcript = self.transcriber.transcribe_without_streaming(
            audio_float.tolist(), sr
        )

        segments, parts = [], []
        for line in transcript.lines:
            text = line.text.strip()
            if not text:
                continue
            segments.append({
                "start": round(line.start_time, 2),
                "end": round(line.start_time + line.duration, 2),
                "text": text,
            })
            parts.append(text)

        return {"text": " ".join(parts), "segments": segments, "language": language}

    def create_vad_iterator(self):
        """Per-WebSocket Silero VAD iterator (or None if VAD couldn't load)."""
        if self.vad_model is None:
            return None
        from silero_vad.utils_vad import VADIterator  # type: ignore
        return VADIterator(
            self.vad_model,
            sampling_rate=16000,
            threshold=0.5,
            min_silence_duration_ms=800,   # end-of-turn silence (≈ 1 s)
            speech_pad_ms=30,
        )


def get_stt() -> SttBackend:
    """Lazy-load Moonshine + Silero VAD. Idempotent + thread-safe."""
    global _singleton
    if _singleton is not None:
        return _singleton

    with _lock:
        if _singleton is not None:
            return _singleton

        logger.info("[stt] Loading Moonshine STT…")
        from moonshine_voice import (
            Transcriber,
            TranscriptEventListener,
            get_model_for_language,
        )
        model_path, model_arch = get_model_for_language("en")
        logger.info("[stt] Moonshine model path=%s arch=%s", model_path, model_arch)
        transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
        # Warmup with 1 s of silence so the first real call is responsive.
        transcriber.transcribe_without_streaming(
            np.zeros(16000, dtype=np.float32).tolist(), 16000
        )
        logger.info("[stt] Moonshine ready")

        vad_model = None
        try:
            import torch
            torch.set_num_threads(1)
            logger.info("[stt] Loading Silero VAD…")
            vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            logger.info("[stt] Silero VAD ready")
        except Exception as e:
            logger.warning("[stt] Silero VAD unavailable: %s", e)

        _singleton = SttBackend(
            transcriber=transcriber,
            transcript_event_listener=TranscriptEventListener,
            vad_model=vad_model,
        )
        return _singleton


def is_loaded() -> bool:
    """Cheap probe used by /api/status — does NOT trigger lazy load."""
    return _singleton is not None
