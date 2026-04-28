"""
Real-Time Stress Detection Pipeline
=====================================
Orchestrates the full multimodal stress detection pipeline:

  Camera → Video Frames → FaceMesh/Pose → AU Features → Visual MLP ──┐
                                                                       ├→ Fusion → Stress Score
  Microphone → Audio Chunks → Wav2Vec2 → 512-D Embedding → StudentNet ┘

Supports:
  - Real-time mode (webcam + microphone, continuous inference)
  - Batch mode (process recorded video/audio files)
  - Visual-only or audio-only fallback modes
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .audio_model import AudioStressDetector, resample_audio
from .fusion import FusionStrategy, MultimodalFusion
from .visual_model import VisualStressDetector

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the stress detection pipeline."""

    # Video settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 15  # Processing FPS (downsample from camera)

    # Audio settings
    audio_sample_rate: int = 16000
    audio_chunk_seconds: float = 2.0  # Length of audio window for inference

    # Model settings
    studentnet_weights: Optional[str] = None
    visual_mlp_weights: Optional[str] = None
    wav2vec2_model: str = "facebook/wav2vec2-base"
    use_body_pose: bool = False
    temporal_window: int = 30  # Frames for temporal aggregation

    # Fusion settings
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE
    visual_weight: float = 0.5
    audio_weight: float = 0.5

    # Pipeline settings
    device: Optional[str] = None
    enable_video: bool = True
    enable_audio: bool = True
    show_preview: bool = True


@dataclass
class StressResult:
    """Container for a single pipeline inference result."""

    timestamp: float = 0.0
    visual_result: Optional[dict] = None
    audio_result: Optional[dict] = None
    fused_result: Optional[dict] = None
    frame: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
class StressDetectionPipeline:
    """
    Main orchestrator for real-time multimodal stress detection.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._running = False
        self._results_callback: Optional[Callable[[StressResult], None]] = None

        # Initialize models
        if config.enable_audio:
            logger.info("Initializing audio stress detector...")
            self.audio_detector = AudioStressDetector(
                studentnet_weights=config.studentnet_weights,
                wav2vec2_model=config.wav2vec2_model,
                device=config.device,
            )
        else:
            self.audio_detector = None

        if config.enable_video:
            logger.info("Initializing visual stress detector...")
            self.visual_detector = VisualStressDetector(
                weights_path=config.visual_mlp_weights,
                use_body_pose=config.use_body_pose,
                window_size=config.temporal_window,
                device=config.device,
            )
        else:
            self.visual_detector = None

        # Fusion
        self.fusion = MultimodalFusion(
            strategy=config.fusion_strategy,
            visual_weight=config.visual_weight,
            audio_weight=config.audio_weight,
            device=config.device,
        )

        # Audio buffer
        self._audio_buffer: list[np.ndarray] = []
        self._audio_lock = threading.Lock()

        # Latest results for cross-thread access
        self._latest_result = StressResult()
        self._result_lock = threading.Lock()

    def on_result(self, callback: Callable[[StressResult], None]):
        """Register a callback for each new stress result."""
        self._results_callback = callback

    # ──────────────────────────────────────────────────────────────────────
    # Batch Processing
    # ──────────────────────────────────────────────────────────────────────

    def process_video_file(self, video_path: str) -> list[StressResult]:
        """
        Process a recorded video file frame by frame.
        Returns a list of StressResult per processed frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(1, int(original_fps / self.config.fps))
        results = []
        frame_idx = 0

        logger.info("Processing video '%s' (%.1f fps, skip=%d)", video_path, original_fps, frame_skip)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                result = StressResult(timestamp=frame_idx / original_fps, frame=frame)

                if self.visual_detector:
                    result.visual_result = self.visual_detector.process_frame(frame)

                result.fused_result = self.fusion.fuse(
                    visual_result=result.visual_result,
                    audio_result=None,
                    visual_features=self.visual_detector.get_temporal_features() if self.visual_detector else None,
                )
                results.append(result)

            frame_idx += 1

        cap.release()
        logger.info("Processed %d frames, got %d results", frame_idx, len(results))
        return results

    def process_audio_file(self, audio_path: str) -> StressResult:
        """
        Process a recorded audio file.
        Returns a single StressResult with the audio prediction.
        """
        import wave

        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if sr != 16000:
            waveform = resample_audio(waveform, sr, 16000)

        result = StressResult(timestamp=0.0)
        if self.audio_detector:
            result.audio_result = self.audio_detector.predict(waveform, sampling_rate=16000)
        result.fused_result = self.fusion.fuse(audio_result=result.audio_result)
        return result

    def process_av_files(self, video_path: str, audio_path: str) -> list[StressResult]:
        """
        Process paired video and audio files with multimodal fusion.
        """
        # Process audio globally
        audio_result = self.process_audio_file(audio_path)

        # Process video frame-by-frame, fusing with global audio result
        video_results = self.process_video_file(video_path)

        fused_results = []
        for vr in video_results:
            fused = self.fusion.fuse(
                visual_result=vr.visual_result,
                audio_result=audio_result.audio_result,
                visual_features=self.visual_detector.get_temporal_features() if self.visual_detector else None,
            )
            vr.audio_result = audio_result.audio_result
            vr.fused_result = fused
            fused_results.append(vr)

        return fused_results

    # ──────────────────────────────────────────────────────────────────────
    # Real-Time Processing
    # ──────────────────────────────────────────────────────────────────────

    def _audio_capture_thread(self):
        """Capture audio from microphone in a background thread."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed; audio capture disabled.")
            return

        chunk_samples = int(self.config.audio_sample_rate * self.config.audio_chunk_seconds)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio status: %s", status)
            with self._audio_lock:
                self._audio_buffer.append(indata[:, 0].copy())

        logger.info("Starting audio capture at %d Hz", self.config.audio_sample_rate)
        with sd.InputStream(
            samplerate=self.config.audio_sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=chunk_samples,
        ):
            while self._running:
                time.sleep(0.1)

    def _get_audio_chunk(self) -> Optional[np.ndarray]:
        """Retrieve and clear the audio buffer."""
        with self._audio_lock:
            if not self._audio_buffer:
                return None
            chunk = np.concatenate(self._audio_buffer)
            self._audio_buffer.clear()
        return chunk

    def run_realtime(self):
        """
        Run the real-time stress detection pipeline using webcam and microphone.
        Press 'q' to stop.
        """
        self._running = True

        # Start audio thread
        audio_thread = None
        if self.config.enable_audio:
            audio_thread = threading.Thread(target=self._audio_capture_thread, daemon=True)
            audio_thread.start()

        # Open camera
        cap = None
        if self.config.enable_video:
            cap = cv2.VideoCapture(self.config.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            if not cap.isOpened():
                logger.error("Cannot open camera %d", self.config.camera_index)
                self._running = False
                return

        logger.info("Pipeline running. Press 'q' to stop.")
        frame_interval = 1.0 / self.config.fps

        try:
            while self._running:
                loop_start = time.time()
                result = StressResult(timestamp=time.time())

                # ── Video inference ──
                if cap is not None:
                    ret, frame = cap.read()
                    if ret:
                        result.frame = frame
                        result.visual_result = self.visual_detector.process_frame(frame)

                # ── Audio inference ──
                if self.audio_detector is not None:
                    audio_chunk = self._get_audio_chunk()
                    if audio_chunk is not None and len(audio_chunk) > self.config.audio_sample_rate * 0.5:
                        result.audio_result = self.audio_detector.predict(
                            audio_chunk, sampling_rate=self.config.audio_sample_rate
                        )

                # ── Fusion ──
                result.fused_result = self.fusion.fuse(
                    visual_result=result.visual_result,
                    audio_result=result.audio_result,
                    visual_features=(
                        self.visual_detector.get_temporal_features()
                        if self.visual_detector
                        else None
                    ),
                )

                # Store and dispatch result
                with self._result_lock:
                    self._latest_result = result
                if self._results_callback:
                    self._results_callback(result)

                # ── Preview ──
                if self.config.show_preview and result.frame is not None:
                    preview = self._draw_overlay(result)
                    cv2.imshow("Stress Detection Pipeline", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Throttle to target FPS
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        finally:
            self._running = False
            if cap is not None:
                cap.release()
            if self.config.show_preview:
                cv2.destroyAllWindows()
            logger.info("Pipeline stopped.")

    def stop(self):
        """Signal the pipeline to stop."""
        self._running = False

    def _draw_overlay(self, result: StressResult) -> np.ndarray:
        """Draw stress score overlay on the video frame."""
        frame = result.frame.copy()
        y = 30

        def put_text(text, color=(0, 255, 0)):
            nonlocal y
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

        put_text("=== Stress Detection ===", (255, 255, 255))

        if result.visual_result:
            v = result.visual_result
            color = (0, 0, 255) if v["label"] == "stressed" else (0, 255, 0)
            put_text(f"Visual: {v['label']} ({v['confidence']:.2f})", color)
        else:
            put_text("Visual: N/A", (128, 128, 128))

        if result.audio_result:
            a = result.audio_result
            color = (0, 0, 255) if a["label"] == "stressed" else (0, 255, 0)
            put_text(f"Audio: {a['label']} ({a['confidence']:.2f})", color)
        else:
            put_text("Audio: N/A", (128, 128, 128))

        if result.fused_result and result.fused_result.get("label") != "unknown":
            f = result.fused_result
            color = (0, 0, 255) if f["label"] == "stressed" else (0, 255, 0)
            put_text(f"Fused: {f['label']} ({f['confidence']:.2f})", color)
            modalities = ", ".join(f.get("modalities_used", []))
            put_text(f"Sources: {modalities}", (200, 200, 200))

        return frame
