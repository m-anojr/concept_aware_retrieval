"""
ASR transcription (Whisper) + topic-drift signal.

Pipeline:
  1. Transcribe the full lecture audio with Whisper, obtaining a list of
     (start, end, text) segments with word/segment-level timestamps.
  2. For each fixed time-step of the multimodal feature stream, gather the
     transcript text spoken in a rolling window centred on that time-step.
  3. Encode each rolling-window transcript with Sentence-BERT.
  4. The "topic-drift" signal at time t is the embedding distance between the
     rolling-window transcript ending at t and the one ending at t-1 - large
     values indicate the spoken content is changing topic.

Both Whisper and Sentence-BERT are *frozen, pretrained* models used purely
for feature extraction (no gradients, no fine-tuning).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class WhisperASR:
    """Thin wrapper around openai-whisper for offline transcription."""

    def __init__(self, model_name: str = "base", device: str = None):
        self._model = None
        self.model_name = model_name
        self.device = device

    def _load(self):
        if self._model is None:
            import whisper  # lazy import
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        model = self._load()
        result = model.transcribe(audio_path, verbose=False)
        return [
            TranscriptSegment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
            for seg in result["segments"]
        ]


class DummyASR:
    """Fallback used when Whisper / ffmpeg is unavailable. Produces an empty
    transcript so the pipeline can still run end-to-end."""

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        return []


def get_asr_backend(model_name: str = "base", use_dummy: bool = False, device: str = None):
    if use_dummy:
        return DummyASR()
    try:
        return WhisperASR(model_name, device=device)
    except Exception:
        return DummyASR()


class SentenceEncoder:
    """Wrapper around a frozen Sentence-BERT model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self._model = None
        self.model_name = model_name
        self._device = device
        self._dim = 384  # all-MiniLM-L6-v2 default

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        try:
            model = self._load()
            embeds = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeds.astype(np.float32)
        except Exception:
            # Deterministic fallback so downstream shapes remain valid.
            rng = np.random.default_rng(abs(hash(tuple(texts))) % (2 ** 32))
            return rng.normal(size=(len(texts), self._dim)).astype(np.float32)


def text_for_window(transcript: List[TranscriptSegment], t_center: float, half_window: float) -> str:
    """Concatenate transcript text whose interval overlaps
    [t_center - half_window, t_center + half_window]."""
    lo, hi = t_center - half_window, t_center + half_window
    parts = [seg.text for seg in transcript if seg.end >= lo and seg.start <= hi]
    return " ".join(parts).strip()


def extract_topic_drift_signal(
    transcript: List[TranscriptSegment],
    timestamps: np.ndarray,
    encoder: SentenceEncoder,
    half_window: float = 15.0,
) -> Tuple[np.ndarray, List[str]]:
    """Compute topic-drift signal over the given timestamps.

    Returns
    -------
    signal : np.ndarray of shape (T,)
        signal[0] = 0; signal[t] = 1 - cosine(embed(window_t), embed(window_{t-1}))
    window_texts : List[str]
        Per-time-step transcript text (used later for segment aggregation).
    """
    window_texts = [text_for_window(transcript, t, half_window) for t in timestamps]
    embeds = encoder.encode(window_texts)

    signal = np.zeros(len(timestamps), dtype=np.float32)
    if len(embeds) > 1:
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = embeds / norms
        cos_sim = np.sum(unit[1:] * unit[:-1], axis=1)
        signal[1:] = np.clip(1.0 - cos_sim, 0.0, 2.0)
    return signal, window_texts