"""
OCR-based board/slide text-change signal.

For each sampled frame, OCR is run on the (cropped) board/slide region of the
frame. The signal fed into Stage 1 is the *rate of textual change* between
consecutive sampled frames - a proxy for "the instructor wrote/erased/changed
the slide", which is one of the strongest cues for a concept boundary in a
lecture video.

Two backends are supported:
  - EasyOCR (default, GPU-accelerated if available)
  - Tesseract via pytesseract (lighter, CPU-only)

Both are *frozen* - no training happens here. If neither OCR engine is
installed, a deterministic dummy backend is used so the rest of the pipeline
can still run (useful for smoke-testing on machines without OCR installed).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import difflib

import numpy as np


@dataclass
class OCRFrameResult:
    timestamp: float
    text: str


class BaseOCRBackend:
    def read_text(self, frame: np.ndarray) -> str:
        raise NotImplementedError


class EasyOCRBackend(BaseOCRBackend):
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        import easyocr  # lazy import

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def read_text(self, frame: np.ndarray) -> str:
        results = self.reader.readtext(frame, detail=0, paragraph=True)
        return " ".join(results)


class TesseractBackend(BaseOCRBackend):
    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def read_text(self, frame: np.ndarray) -> str:
        import pytesseract  # lazy import
        from PIL import Image

        img = Image.fromarray(frame)
        return pytesseract.image_to_string(img, lang=self.lang)


class DummyOCRBackend(BaseOCRBackend):
    """Fallback backend used only when no real OCR engine is available.

    Returns an empty string, which yields a zero text-change signal. This
    keeps the pipeline runnable end-to-end (e.g. for unit tests) but should
    NOT be used for real experiments.
    """

    def read_text(self, frame: np.ndarray) -> str:
        return ""


def get_ocr_backend(engine: str = "easyocr", gpu: Optional[bool] = None) -> BaseOCRBackend:
    # Auto-detect GPU if not explicitly provided
    if gpu is None:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False

    if engine == "easyocr":
        try:
            return EasyOCRBackend(gpu=gpu)
        except Exception:
            pass
    if engine == "tesseract":
        try:
            return TesseractBackend()
        except Exception:
            pass
    return DummyOCRBackend()


def board_region(frame: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Crop the board/slide region of a frame.

    `bbox` is given as fractional (x_min, y_min, x_max, y_max) in [0, 1].
    If not provided, the full frame is used. In a real deployment this bbox
    would be calibrated per camera setup (or detected automatically), but a
    sensible default is to crop out a thin strip of UI chrome at the edges.
    """
    h, w = frame.shape[:2]
    if bbox is None:
        # Tighter crop to exclude typical lecture video watermarks, channel
        # logos, and UI chrome that appear at the edges/corners (e.g.
        # "SAVE MORE", "NPTEL" banners).  This dramatically improves OCR
        # quality on real NPTEL / YouTube lecture recordings.
        bbox = (0.05, 0.10, 0.95, 0.85)
    x0, y0, x1, y1 = bbox
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def text_change_rate(prev_text: str, curr_text: str) -> float:
    """Compute a normalized text-change score in [0, 1].

    0 -> texts are identical (no board/slide change)
    1 -> texts are completely different
    """
    if not prev_text and not curr_text:
        return 0.0
    matcher = difflib.SequenceMatcher(None, prev_text, curr_text)
    similarity = matcher.ratio()
    return float(1.0 - similarity)


def extract_ocr_signal(
    frames: List[np.ndarray],
    backend: Optional[BaseOCRBackend] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Run OCR on a sequence of sampled frames and return the text-change
    signal alongside the raw OCR text for each frame (used later to build
    the segment-level OCR text for Stage 2).

    Returns
    -------
    signal : np.ndarray of shape (T,)
        signal[0] = 0 by convention; signal[t] = text_change_rate(t-1, t)
    texts : List[str]
        OCR text recognized for each frame.
    """
    backend = backend or get_ocr_backend()
    texts = [backend.read_text(board_region(f, bbox)) for f in frames]

    signal = np.zeros(len(frames), dtype=np.float32)
    for t in range(1, len(frames)):
        signal[t] = text_change_rate(texts[t - 1], texts[t])
    return signal, texts