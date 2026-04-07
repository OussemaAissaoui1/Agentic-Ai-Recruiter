"""
Viseme Engine — Audio-to-Viseme Extraction for GLB Avatar Lip Sync

Extracts viseme (mouth shape) weights from WAV audio data for driving
Ready Player Me avatar blend shapes. Uses spectral analysis to map
frequency content to the 16 standard RPM viseme targets.

Viseme Targets (Ready Player Me / ARKit compatible):
    visemeSil  — Silence / neutral mouth
    visemePP   — P, B, M (bilabial)
    visemeFF   — F, V (labiodental)
    visemeTH   — Th (dental fricative)
    visemeDD   — D, T, N (alveolar)
    visemeK    — K, G, Ng (velar)
    visemeCH   — Ch, J, Sh (postalveolar)
    visemeSS   — S, Z (sibilant)
    visemeNN   — N (nasal)
    visemeRR   — R (approximant)
    visemeAA   — A (open vowel)
    visemeE    — E (mid front vowel)
    visemeI    — I (close front vowel)
    visemeO    — O (mid back vowel)
    visemeU    — U (close back vowel)
    visemeOH   — Oh (open back vowel)
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger("avatar.viseme_engine")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

VISEME_NAMES = [
    "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD",
    "viseme_kk", "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR",
    "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U",
]


@dataclass
class VisemeFrame:
    """A single frame of viseme weights."""
    time: float  # seconds from start
    duration: float  # frame length in seconds
    weights: dict = field(default_factory=dict)


@dataclass
class VisemeTimeline:
    """Complete viseme timeline for an audio clip."""
    frames: List[VisemeFrame] = field(default_factory=list)
    duration: float = 0.0
    sample_rate: int = 24000
    frame_duration: float = 0.02  # 20ms per frame


class VisemeEngine:
    """
    Extracts viseme weights from WAV audio using spectral analysis.

    Uses FFT-based frequency band energy extraction to approximate
    articulatory phonetic features and map them to viseme targets.
    """

    def __init__(
        self,
        frame_duration_ms: int = 20,
        smoothing_factor: float = 0.35,
        energy_threshold: float = 0.01,
    ):
        self._frame_duration_ms = frame_duration_ms
        self._smoothing = smoothing_factor
        self._energy_threshold = energy_threshold
        logger.info(
            "VisemeEngine initialized | frame=%dms smooth=%.2f threshold=%.3f",
            frame_duration_ms, smoothing_factor, energy_threshold,
        )

    def extract_from_wav_bytes(self, wav_bytes: bytes) -> VisemeTimeline:
        """
        Extract viseme timeline from WAV audio bytes.

        Args:
            wav_bytes: Complete WAV file as bytes.

        Returns:
            VisemeTimeline with per-frame viseme weights.

        Raises:
            ValueError: If audio format is unsupported.
        """
        logger.debug("Extracting visemes from %d bytes of WAV data", len(wav_bytes))

        try:
            samples, sample_rate = self._decode_wav(wav_bytes)
        except Exception as e:
            logger.error("Failed to decode WAV: %s", e)
            raise ValueError(f"WAV decode failed: {e}") from e

        if len(samples) == 0:
            logger.warning("Empty audio — returning silent timeline")
            return VisemeTimeline(frames=[], duration=0.0, sample_rate=sample_rate)

        frame_samples = int(sample_rate * self._frame_duration_ms / 1000)
        num_frames = max(1, len(samples) // frame_samples)
        duration = len(samples) / sample_rate

        logger.debug(
            "Audio: %.2fs, %d Hz, %d samples, %d frames",
            duration, sample_rate, len(samples), num_frames,
        )

        frames = []
        prev_weights = {name: 0.0 for name in VISEME_NAMES}

        for i in range(num_frames):
            start = i * frame_samples
            end = min(start + frame_samples, len(samples))
            frame_data = samples[start:end]

            raw_weights = self._analyze_frame(frame_data, sample_rate)

            # Apply temporal smoothing
            smoothed = {}
            for name in VISEME_NAMES:
                raw = raw_weights.get(name, 0.0)
                prev = prev_weights.get(name, 0.0)
                smoothed[name] = prev * (1 - self._smoothing) + raw * self._smoothing
            prev_weights = smoothed

            frames.append(VisemeFrame(
                time=i * self._frame_duration_ms / 1000,
                duration=self._frame_duration_ms / 1000,
                weights={k: round(v, 4) for k, v in smoothed.items() if v > 0.005},
            ))

        timeline = VisemeTimeline(
            frames=frames,
            duration=duration,
            sample_rate=sample_rate,
            frame_duration=self._frame_duration_ms / 1000,
        )

        logger.info(
            "Viseme extraction complete: %d frames over %.2fs",
            len(frames), duration,
        )
        return timeline

    def _decode_wav(self, wav_bytes: bytes) -> tuple:
        """Decode WAV bytes to float32 samples normalized to [-1, 1]."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        if sample_width == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
        elif sample_width == 4:
            fmt = f"<{n_frames * n_channels}i"
            samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert to mono if stereo
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples, sample_rate

    def _analyze_frame(self, frame: np.ndarray, sample_rate: int) -> dict:
        """
        Analyze a single audio frame and return viseme weights.

        Uses FFT to extract frequency band energies, then maps to
        viseme weights using articulatory phonetics heuristics.
        """
        weights = {name: 0.0 for name in VISEME_NAMES}

        # RMS energy
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < self._energy_threshold:
            weights["viseme_sil"] = 1.0
            return weights

        # Normalize energy to [0, 1] range (log scale)
        energy = min(1.0, max(0.0, (np.log10(rms + 1e-10) + 2) / 2))

        # Apply Hann window and compute FFT
        windowed = frame * np.hanning(len(frame))
        fft_mag = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)

        # Extract energy in frequency bands
        bands = self._get_band_energies(fft_mag, freqs)

        # Map frequency bands to visemes
        # Low frequencies (80-300 Hz) — voice fundamental, indicates voicing
        voice_energy = bands.get("voice", 0.0)

        # First formant region (300-1000 Hz) — vowel height
        f1_energy = bands.get("f1", 0.0)

        # Second formant region (1000-3000 Hz) — vowel frontness
        f2_energy = bands.get("f2", 0.0)

        # Fricative region (3000-6000 Hz)
        fric_energy = bands.get("fricative", 0.0)

        # Sibilant region (6000-12000 Hz)
        sib_energy = bands.get("sibilant", 0.0)

        # Spectral centroid for overall brightness
        total_mag = fft_mag.sum()
        if total_mag > 0:
            centroid = float(np.sum(freqs * fft_mag) / total_mag)
        else:
            centroid = 0.0

        # --- Viseme mapping (Wolf3D / RPM underscore names) ---

        # Silence/neutral (inverse of energy)
        weights["viseme_sil"] = max(0.0, 1.0 - energy * 3)

        # Open vowels — high F1, moderate energy
        weights["viseme_aa"] = min(1.0, f1_energy * 1.8 * energy)

        # Front vowels — high F2
        weights["viseme_E"] = min(1.0, f2_energy * 1.2 * energy * (1 - f1_energy * 0.5))
        weights["viseme_I"] = min(1.0, f2_energy * 1.5 * (1 - f1_energy) * energy)

        # Back vowels — low F2, moderate F1
        weights["viseme_O"] = min(1.0, f1_energy * (1 - f2_energy * 0.8) * energy * 1.3)
        weights["viseme_U"] = min(1.0, (1 - f2_energy) * (1 - f1_energy * 0.5) * energy * 0.8)

        # Fricatives — high frequency content
        weights["viseme_SS"] = min(1.0, sib_energy * 2.0 * energy)
        weights["viseme_FF"] = min(1.0, fric_energy * 1.5 * (1 - sib_energy) * energy)
        weights["viseme_TH"] = min(1.0, fric_energy * 0.8 * energy)

        # Plosives/nasals — detected by energy transients (approximated)
        weights["viseme_PP"] = min(1.0, energy * 0.6 * (1 - voice_energy) * (1 - fric_energy))
        weights["viseme_DD"] = min(1.0, energy * 0.5 * voice_energy * (1 - f1_energy * 0.5))
        weights["viseme_kk"] = min(1.0, energy * 0.4 * (1 - voice_energy * 0.5) * fric_energy * 0.5)
        weights["viseme_CH"] = min(1.0, fric_energy * sib_energy * energy * 1.2)
        weights["viseme_nn"] = min(1.0, voice_energy * 0.5 * (1 - fric_energy) * energy * 0.6)
        weights["viseme_RR"] = min(1.0, f1_energy * f2_energy * energy * 0.8)

        # Normalize: ensure weights sum to reasonable range
        total = sum(weights.values())
        if total > 1.5:
             scale = 1.5 / total
             weights = {k: v * scale for k, v in weights.items()}

        return weights

    def _get_band_energies(self, fft_mag: np.ndarray, freqs: np.ndarray) -> dict:
        """Extract normalized energy in frequency bands."""
        bands = {}
        total_energy = float(np.sum(fft_mag ** 2)) + 1e-10

        band_ranges = {
            "voice": (80, 300),
            "f1": (300, 1000),
            "f2": (1000, 3000),
            "fricative": (3000, 6000),
            "sibilant": (6000, 12000),
        }

        for name, (low, high) in band_ranges.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy = float(np.sum(fft_mag[mask] ** 2))
            bands[name] = min(1.0, band_energy / total_energy * 3)

        return bands

    def timeline_to_dict(self, timeline: VisemeTimeline) -> list:
        """Convert VisemeTimeline to a JSON-serializable list of dicts."""
        return [
            {
                "time": f.time,
                "duration": f.duration,
                "weights": f.weights,
            }
            for f in timeline.frames
        ]
