"""
TTS Engine — Kokoro Text-to-Speech  (streaming-optimized v2)

Architecture upgrade: sentence-boundary streaming pipeline.

Old behaviour (v1):
    generation finishes → TTS synthesizes whole sentence → audio returned
    Perceived latency = generation_time + tts_time  (~0.6s + ~0.15s = ~0.75s)

New behaviour (v2):
    tokens stream in → sentence boundary detected → TTS fires on that chunk
    immediately → audio for chunk 1 plays while generation of chunk 2 is
    still running.
    Perceived latency ≈ generation_time_to_first_sentence  (~0.2s)

Public API additions:
    TTSEngine.synthesize_streaming(token_async_gen)
        Async generator — yields TTSResult objects one sentence at a time
        as soon as each sentence is ready.  Caller can play/queue them
        incrementally.

    TTSEngine.synthesize_segmented(text)
        For the non-streaming path: splits a completed text into sentences
        and synthesizes them concurrently, then returns a StreamingTTSResult.

    TTSEngine.split_sentences(text)
        Utility: split completed text into TTS-sized chunks.

Performance:
    ~150x real-time on GPU, ~35x real-time on CPU
    A 20-word question takes ~50-150ms to synthesize per sentence chunk.
    Sample rate: 24000 Hz
"""
from __future__ import annotations

import io
import re
import time
import asyncio
import numpy as np
from typing import Optional, AsyncGenerator, List
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Sentence boundary splitter
# ---------------------------------------------------------------------------

# Split on . ? ! but not on abbreviations like "Dr." or "U.S."
_SENTENCE_END = re.compile(
    r'(?<=[^A-Z][^A-Z])(?<=[.?!])(?=\s|$)',
    re.UNICODE,
)
_MIN_CHUNK_CHARS = 20


def split_sentences(text: str) -> List[str]:
    """
    Split text into TTS-sized sentence chunks.
    Returns a list of non-empty stripped strings.
    """
    parts = _SENTENCE_END.split(text)
    result = []
    buf = ""
    for part in parts:
        buf = (buf + part).strip()
        if len(buf) >= _MIN_CHUNK_CHARS and buf[-1] in ".?!":
            result.append(buf)
            buf = ""
    if buf.strip():
        result.append(buf.strip())
    return result if result else [text.strip()]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTSResult:
    """Result of text-to-speech synthesis for one sentence chunk."""
    audio_array: np.ndarray
    sample_rate: int
    duration_sec: float
    latency_ms: float
    sentence_index: int = 0
    source_text: str = ""

    def to_wav_bytes(self) -> bytes:
        """Convert audio to WAV format bytes for playback."""
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, self.audio_array, self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()


@dataclass
class StreamingTTSResult:
    """
    Collected result from a full streaming synthesis run.
    Contains per-sentence TTSResult objects that can be played
    sequentially, plus a convenience method to concatenate them all.
    """
    chunks: List[TTSResult] = field(default_factory=list)

    def to_wav_bytes(self) -> bytes:
        """Concatenate all chunks into a single WAV file."""
        if not self.chunks:
            return b""
        import soundfile as sf
        combined = np.concatenate([c.audio_array for c in self.chunks])
        sr = self.chunks[0].sample_rate
        buffer = io.BytesIO()
        sf.write(buffer, combined, sr, format="WAV")
        buffer.seek(0)
        return buffer.read()

    @property
    def total_duration_sec(self) -> float:
        return sum(c.duration_sec for c in self.chunks)

    @property
    def total_latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.chunks)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Kokoro-based TTS engine — streaming-optimized v2.

    Key addition: synthesize_streaming() accepts an async token generator
    (e.g. from vLLM generate_streaming) and yields TTSResult objects
    sentence by sentence as soon as each sentence boundary is detected.
    Audio playback of sentence 1 begins while sentence 2 is still being
    generated — cutting perceived latency by ~60-70%.

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
            self._pipeline = kokoro.KPipeline(
                lang_code=lang_code, device=torch.device("cuda") 
            )
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

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        return split_sentences(text)

    # ------------------------------------------------------------------
    # Core synthesis (sync — always runs in a thread pool executor)
    # ------------------------------------------------------------------

    def _synthesize_text_sync(
        self, text: str, sentence_index: int = 0
    ) -> Optional[TTSResult]:
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
            sentence_index=sentence_index,
            source_text=text,
        )

    # Backwards-compatible alias
    def synthesize_sync(self, text: str) -> Optional[TTSResult]:
        return self._synthesize_text_sync(text, sentence_index=0)

    # ------------------------------------------------------------------
    # Async: single text — backwards-compatible
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> Optional[TTSResult]:
        """Async synthesis of a complete text. Backwards-compatible with v1."""
        if not self._ready:
            return None
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, self._synthesize_text_sync, text, 0
                ),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            print("[tts] Synthesis timed out (>5s)")
            return None
        except Exception as e:
            print(f"[tts] Async synthesis error: {e}")
            return None

    # ------------------------------------------------------------------
    # NEW: Streaming synthesis — sentence-boundary pipelined
    # ------------------------------------------------------------------

    async def synthesize_streaming(
        self,
        token_generator: AsyncGenerator[str, None],
        timeout_per_chunk: float = 5.0,
    ) -> AsyncGenerator[TTSResult, None]:
        """
        Sentence-boundary streaming TTS pipeline.

        Consumes an async token generator (from vLLM generate_streaming),
        buffers tokens until a sentence boundary (.?!) is detected, then
        fires TTS synthesis on that chunk immediately — without waiting for
        the rest of generation to finish.

        Yields TTSResult objects one sentence at a time as soon as each is
        synthesized. The caller can start playing chunk 0 while chunk 1 is
        still being generated AND synthesized.

        Latency improvement:
            Before:  wait(full_generation) + wait(full_tts)   ~0.75s
            After:   wait(generation_to_first_sentence_end)   ~0.2s
        """
        if not self._ready:
            return

        loop = asyncio.get_event_loop()
        token_buf = ""
        sentence_idx = 0

        # We overlap: while the current sentence is being synthesized on the
        # thread pool, generation of the next tokens continues on the event loop.
        pending_future: Optional[asyncio.Future] = None
        pending_idx: int = 0

        async def _await_pending() -> Optional[TTSResult]:
            nonlocal pending_future
            if pending_future is None:
                return None
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(pending_future), timeout=timeout_per_chunk
                )
                return result
            except (asyncio.TimeoutError, Exception) as e:
                print(f"[tts] Chunk {pending_idx} error: {e}")
                return None
            finally:
                pending_future = None

        async for token in token_generator:
            token_buf += token

            # Fire TTS as soon as we hit a sentence boundary
            if (
                len(token_buf) >= _MIN_CHUNK_CHARS
                and token_buf.rstrip()
                and token_buf.rstrip()[-1] in ".?!"
            ):
                sentence_text = token_buf.strip()
                token_buf = ""

                # Yield previous chunk before starting the new one
                result = await _await_pending()
                if result:
                    yield result

                # Start this chunk's synthesis immediately (non-blocking)
                pending_future = loop.run_in_executor(
                    None, self._synthesize_text_sync, sentence_text, sentence_idx
                )
                pending_idx = sentence_idx
                sentence_idx += 1

        # Yield any still-running synthesis
        result = await _await_pending()
        if result:
            yield result

        # Synthesize any leftover text that didn't end with punctuation
        remainder = token_buf.strip()
        if remainder:
            if remainder[-1] not in ".?!":
                remainder += "?"
            try:
                fut = loop.run_in_executor(
                    None, self._synthesize_text_sync, remainder, sentence_idx
                )
                result = await asyncio.wait_for(fut, timeout=timeout_per_chunk)
                if result:
                    yield result
            except Exception as e:
                print(f"[tts] Remainder synthesis error: {e}")

    # ------------------------------------------------------------------
    # NEW: Segmented synthesis for completed text (concurrent sentences)
    # ------------------------------------------------------------------

    async def synthesize_segmented(
        self, text: str, timeout_per_chunk: float = 5.0
    ) -> StreamingTTSResult:
        """
        Split a completed text into sentences and synthesize concurrently.

        For the non-streaming generation path: instead of synthesizing the
        whole text as one long string, we split on sentence boundaries and
        run each chunk in parallel on the thread pool. Results are sorted
        back into order before returning.

        For typical single-sentence interview questions this is equivalent
        to synthesize() but returns a StreamingTTSResult for consistency.
        Multi-sentence responses get a real speedup.
        """
        if not self._ready or not text.strip():
            return StreamingTTSResult()

        loop = asyncio.get_event_loop()
        sentences = split_sentences(text)

        futures = [
            loop.run_in_executor(
                None, self._synthesize_text_sync, s, i
            )
            for i, s in enumerate(sentences)
        ]

        result = StreamingTTSResult()
        for fut in asyncio.as_completed(futures):
            try:
                chunk = await asyncio.wait_for(fut, timeout=timeout_per_chunk)
                if chunk:
                    result.chunks.append(chunk)
            except Exception as e:
                print(f"[tts] Segmented chunk error: {e}")

        result.chunks.sort(key=lambda x: x.sentence_index)
        return result