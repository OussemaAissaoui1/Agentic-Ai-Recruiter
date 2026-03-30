"""
TTS Engine — Kokoro Text-to-Speech

Converts recruiter text responses to natural-sounding speech audio.
Uses Kokoro TTS (Apache 2.0, 82M params) running entirely on CPU.

Performance:
- ~150x real-time on GPU, ~35x real-time on CPU
- A 20-word question takes ~50-150ms to synthesize
- Sample rate: 24000 Hz
"""
from __future__ import annotations
import io
import time
import asyncio
import numpy as np
from typing import Optional
from dataclasses import dataclass
@dataclass(frozen=True)
class TTSResult:
    """Result of text-to-speech synthesis."""
    audio_array: np.ndarray
    sample_rate: int
    duration_sec: float
    latency_ms: float

    def to_wav_bytes(self) -> bytes:
        """Convert audio to WAV format bytes for playback."""
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, self.audio_array, self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
class TTSEngine:
    """
    Kokoro-based TTS engine for the recruiter persona.
    Runs entirely on CPU — no GPU competition with the 8B recruiter model.
    Typical latency: 50-150ms for a short interview question.

    Voices:
        af_heart   — warm female (default, good for recruiter)
        af_bella   — professional female
        am_adam    — professional male
        am_michael — warm male
        bf_emma    — British female
        bm_george  — British male
    """
    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 0.95,
    ):
        self._voice = voice
        self._lang_code = lang_code
        self._speed = speed
        self._sample_rate = 24000
        self._ready = False
        self._pipeline = None

        try:
            import kokoro
            import torch
            print(f"[tts] Loading Kokoro TTS (voice={voice}, lang={lang_code})...")
            t0 = time.time()
            self._pipeline = kokoro.KPipeline(lang_code=lang_code, device=torch.device('cpu'))
            load_time = time.time() - t0
            print(f"[tts] Ready in {load_time:.1f}s")
            self._ready = True
        except ImportError:
            print("[tts] WARNING: kokoro not installed. Run: pip install kokoro>=0.9")
            print("[tts] TTS will be disabled. Text-only mode.")
        except Exception as e:
            print(f"[tts] WARNING: Failed to initialize Kokoro: {e}")
            print("[tts] TTS will be disabled. Text-only mode.")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize_sync(self, text: str) -> Optional[TTSResult]:
        """Synchronous synthesis — call from thread pool."""
        if not self._ready or not text.strip():
            return None

        t0 = time.perf_counter()
        chunks = []

        try:
            for _, _, audio in self._pipeline(
                text, voice=self._voice, speed=self._speed
            ):
                if audio is not None and len(audio) > 0:
                    chunks.append(audio)
        except Exception as e:
            print(f"[tts] Synthesis error: {e}")
            return None

        if not chunks:
            return None

        full_audio = np.concatenate(chunks)
        latency_ms = (time.perf_counter() - t0) * 1000
        duration_sec = len(full_audio) / self._sample_rate

        return TTSResult(
            audio_array=full_audio,
            sample_rate=self._sample_rate,
            duration_sec=duration_sec,
            latency_ms=latency_ms,
        )

    async def synthesize(self, text: str) -> Optional[TTSResult]:
        """Async synthesis — runs CPU work in thread pool."""
        if not self._ready:
            return None
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self.synthesize_sync, text),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            print("[tts] Synthesis timed out (>5s)")
            return None
        except Exception as e:
            print(f"[tts] Async synthesis error: {e}")
            return None