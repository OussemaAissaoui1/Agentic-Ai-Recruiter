"""
Audio Stress Detection Module
==============================
Implements two pathways:
  1. **StudentNet** – A lightweight MLP (from HuggingFace forwarder1121/StudentNet)
     that takes 768-D Wav2Vec2 embeddings and outputs binary stress predictions.
  2. **Wav2Vec2 Feature Extractor** – Extracts 768-D embeddings from raw audio
     using facebook/wav2vec2-base.

Pipeline:
  raw audio (16 kHz) → Wav2Vec2 feature extraction → 768-D embedding → StudentNet MLP → stress score
"""

import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)
class Wav2Vec2FeatureExtractor:
    """
    Extracts 768-dimensional embeddings from raw audio waveforms
    using the facebook/wav2vec2-base model.
    """
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: Optional[str] = None,
    ):
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading Wav2Vec2 model '%s' on %s", model_name, self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
    @torch.no_grad()
    def extract(self, waveform: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """
        Extract a 768-D embedding from a mono audio waveform.
        Args:
            waveform: 1-D float numpy array of audio samples.
            sampling_rate: Sample rate (must be 16000 for wav2vec2-base).
        Returns:
            768-D numpy array (mean-pooled hidden states).
        """
        inputs = self.processor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        outputs = self.model(input_values)
        # Mean-pool over the time dimension → (1, 768)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding


class StudentNetMLP(nn.Module):
    """
    Lightweight MLP that mirrors the StudentNet architecture
    (forwarder1121/StudentNet on HuggingFace).

    Architecture: Linear(768 → 256) → ReLU → Dropout → Linear(256 → 2)

    This can be loaded from HuggingFace weights or initialized randomly
    for fine-tuning.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AudioStressDetector:
    """
    End-to-end audio stress detector.

    Combines Wav2Vec2FeatureExtractor + StudentNetMLP to go from
    raw audio → stress probability.
    """

    LABELS = ["not_stressed", "stressed"]

    def __init__(
        self,
        studentnet_weights: Optional[str] = None,
        wav2vec2_model: str = "facebook/wav2vec2-base",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            model_name=wav2vec2_model, device=self.device
        )

        # Classifier
        self.classifier = StudentNetMLP().to(self.device)
        if studentnet_weights:
            logger.info("Loading StudentNet weights from %s", studentnet_weights)
            state = torch.load(studentnet_weights, map_location=self.device, weights_only=True)
            self.classifier.load_state_dict(state, strict=False)
        else:
            logger.warning(
                "No StudentNet weights provided; classifier is randomly initialized. "
                "Download from HuggingFace: forwarder1121/StudentNet"
            )
        self.classifier.eval()

    @torch.no_grad()
    def predict(self, waveform: np.ndarray, sampling_rate: int = 16000) -> dict:
        """
        Predict stress from raw audio.

        Args:
            waveform: 1-D float array of audio samples (mono, 16 kHz recommended).
            sampling_rate: Sample rate of the waveform.

        Returns:
            dict with keys:
                label (str): "stressed" or "not_stressed"
                confidence (float): probability of the predicted class
                probabilities (dict): {"not_stressed": p0, "stressed": p1}
        """
        embedding = self.feature_extractor.extract(waveform, sampling_rate)
        tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.classifier(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        label_idx = int(np.argmax(probs))
        return {
            "label": self.LABELS[label_idx],
            "confidence": float(probs[label_idx]),
            "probabilities": {
                self.LABELS[i]: float(probs[i]) for i in range(len(self.LABELS))
            },
        }

    @torch.no_grad()
    def predict_embedding(self, embedding: np.ndarray) -> dict:
        """
        Predict stress from a pre-computed 768-D Wav2Vec2 embedding.
        """
        tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.classifier(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        label_idx = int(np.argmax(probs))
        return {
            "label": self.LABELS[label_idx],
            "confidence": float(probs[label_idx]),
            "probabilities": {
                self.LABELS[i]: float(probs[i]) for i in range(len(self.LABELS))
            },
        }


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to the target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return waveform
    ratio = target_sr / orig_sr
    n_samples = int(len(waveform) * ratio)
    indices = np.linspace(0, len(waveform) - 1, n_samples)
    return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)
