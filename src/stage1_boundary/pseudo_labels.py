"""
Pseudo-label generation for the self-supervised boundary objective.

Stage 1 is trained with a contrastive boundary loss that needs, for each
training video, a (noisy) set of "boundary" time-steps to define which
adjacent/random pairs of time-steps should be pulled together (same segment)
or pushed apart (different segments).

Two label sources are combined, exactly as described in the proposal:

  1. Pseudo-labels (automatic, used for ALL training videos): large *joint*
     peaks in the OCR text-change and topic-drift signals are treated as
     candidate concept-boundary points.

  2. Manual annotations (used for the small annotated subset, ~15-20
     lectures): ground-truth concept-boundary timestamps derived from the
     syllabus / textbook table of contents. These override / refine the
     pseudo-labels when available.

Both sources produce the same artifact: a binary array `is_boundary` of
shape (T-1,), where `is_boundary[t] = 1` means a concept boundary lies
between time-step t and t+1.
"""

from typing import Optional, List
import numpy as np
from scipy.signal import find_peaks

from config import STAGE1


def smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def pseudo_boundaries_from_signals(
    ocr_signal: np.ndarray,
    topic_drift_signal: np.ndarray,
    visual_signal: Optional[np.ndarray] = None,
    smoothing_window: int = STAGE1.peak_smoothing_window,
    min_distance: int = STAGE1.peak_min_distance_steps,
    prominence: float = STAGE1.peak_prominence,
    time_step_sec: float = 3.0,
    min_boundary_interval_sec: float = 300.0,
) -> np.ndarray:
    """Detect candidate concept-boundary time-steps from joint peaks in the
    OCR text-change and topic-drift signals (optionally also visual change).

    If no peaks are detected with the default settings, the function
    progressively lowers the threshold to ensure at least a minimum number
    of boundaries (approximately one per `min_boundary_interval_sec` seconds).
    This prevents the degenerate case where the model sees the entire lecture
    as one segment.

    Returns
    -------
    is_boundary : np.ndarray of shape (T-1,), dtype=int (0/1)
        1 if a boundary is hypothesized between time-step t and t+1.
    """
    T = len(ocr_signal)
    if T < 3:
        return np.zeros(max(T - 1, 0), dtype=np.int64)

    # Combine signals: average of (normalized) OCR-change and topic-drift.
    def _norm(s):
        s = smooth_signal(s.astype(np.float32), smoothing_window)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 1e-8 else np.zeros_like(s)

    combined = 0.5 * _norm(ocr_signal) + 0.5 * _norm(topic_drift_signal)
    if visual_signal is not None:
        combined = 0.7 * combined + 0.3 * _norm(visual_signal)

    # Minimum number of boundaries expected for a video of this length
    video_duration_sec = T * time_step_sec
    min_boundaries = max(1, int(video_duration_sec / min_boundary_interval_sec))

    # Try progressively lower prominence thresholds until we get enough peaks
    peaks = np.array([], dtype=np.int64)
    for prom_scale in [1.0, 0.5, 0.25, 0.1, 0.0]:
        current_prominence = prominence * prom_scale
        peaks, _ = find_peaks(combined, distance=min_distance,
                              prominence=max(current_prominence, 0.01))
        if len(peaks) >= min_boundaries:
            break

    # If still no peaks, fall back to evenly-spaced boundaries
    if len(peaks) == 0 and T > 10:
        step = max(min_distance + 1, T // (min_boundaries + 1))
        peaks = np.arange(step, T - 1, step)

    is_boundary = np.zeros(T - 1, dtype=np.int64)
    for p in peaks:
        idx = min(max(p - 1, 0), T - 2)  # boundary lies between idx and idx+1
        is_boundary[idx] = 1
    return is_boundary


def boundaries_from_annotations(
    timestamps: np.ndarray,
    annotated_boundary_times: List[float],
) -> np.ndarray:
    """Convert manually-annotated boundary timestamps (e.g., from a syllabus
    alignment) into the same (T-1,) binary representation, by snapping each
    annotated time to the nearest inter-step gap.

    Returns
    -------
    is_boundary : np.ndarray of shape (T-1,), dtype=int (0/1)
    """
    T = len(timestamps)
    is_boundary = np.zeros(max(T - 1, 0), dtype=np.int64)
    if T < 2:
        return is_boundary

    midpoints = (timestamps[:-1] + timestamps[1:]) / 2.0  # (T-1,)
    for bt in annotated_boundary_times:
        idx = int(np.argmin(np.abs(midpoints - bt)))
        is_boundary[idx] = 1
    return is_boundary


def merge_boundary_labels(pseudo: np.ndarray, manual: Optional[np.ndarray]) -> np.ndarray:
    """Manual annotations (when available) refine the pseudo-labels: any
    boundary present in either source is kept (logical OR), which is a
    simple but effective way to combine a noisy automatic signal with a
    sparse, high-precision manual signal."""
    if manual is None:
        return pseudo
    return np.clip(pseudo + manual, 0, 1)


def boundaries_to_segment_ids(is_boundary: np.ndarray) -> np.ndarray:
    """Convert a (T-1,) boundary indicator into a (T,) segment-id array,
    e.g. [0,0,0,1,1,2,2,2] for boundaries at indices 2 and 4."""
    T = len(is_boundary) + 1
    seg_ids = np.zeros(T, dtype=np.int64)
    current = 0
    for t in range(1, T):
        if is_boundary[t - 1]:
            current += 1
        seg_ids[t] = current
    return seg_ids
