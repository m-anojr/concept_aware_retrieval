import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2_retrieval.retrieval import LectureRetriever
from app.backend.main import app
print("Retrieval + Backend imports OK")

# Test pseudo-label generation handles weak signals
import numpy as np
from src.stage1_boundary.pseudo_labels import pseudo_boundaries_from_signals

# Simulate weak signals (almost flat)
T = 500
ocr = np.random.rand(T) * 0.05
drift = np.random.rand(T) * 0.05
visual = np.random.rand(T) * 0.05

boundaries = pseudo_boundaries_from_signals(ocr, drift, visual)
n_boundaries = int(boundaries.sum())
print(f"Pseudo-labels with weak signals: {n_boundaries} boundaries found for T={T}")
assert n_boundaries >= 1, "Expected at least 1 boundary for a 500-step video!"

# Test segment fallback
from src.stage1_boundary.segment import segments_from_scores
timestamps = np.arange(T, dtype=np.float32) * 3.0  # 3s steps
scores = np.random.rand(T - 1).astype(np.float32) * 0.1  # very low scores
segments = segments_from_scores(scores, timestamps)
print(f"Segmentation with low scores: {len(segments)} segments for {T} steps")
assert len(segments) >= 2, f"Expected >= 2 segments for a {T*3}s video, got {len(segments)}"

print("\nAll tests passed!")
