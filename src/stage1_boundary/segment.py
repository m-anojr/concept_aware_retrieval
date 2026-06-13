"""
Convert per-time-step boundary scores from the trained Pedagogical Boundary
Detector into a list of discrete, concept-coherent segments
(start_time, end_time), with light post-processing:

  - peak-pick the boundary-score sequence using the same `find_peaks`
    machinery used for pseudo-labels (but on the *learned* score, not the
    raw signals)
  - merge segments shorter than `min_segment_len_steps` into a neighbour,
    so the output is robust to spurious single-step boundaries.
"""

from typing import List, Dict
import numpy as np
from scipy.signal import find_peaks

from config import STAGE1


def boundary_scores_to_boundaries(
    boundary_scores: np.ndarray,
    threshold: float = None,
    min_distance: int = STAGE1.peak_min_distance_steps,
    adaptive_std: float = STAGE1.adaptive_threshold_std,
) -> np.ndarray:
    """boundary_scores: (T-1,) array in [0, 1] (output of model.boundary_scores).

    If `threshold` is None (default), an adaptive threshold is used:
        threshold = mean(boundary_scores) + adaptive_std * std(boundary_scores)
    This is more robust than a fixed absolute cosine-similarity threshold,
    since the absolute scale of "1 - cos_sim" depends on the embedding
    dimensionality and how long/strongly the model has been trained.

    Returns indices t such that a boundary lies between time-step t and t+1.
    """
    if len(boundary_scores) == 0:
        return np.array([], dtype=np.int64)

    if threshold is None:
        mu, sigma = float(boundary_scores.mean()), float(boundary_scores.std())
        threshold = mu + adaptive_std * sigma

    peaks, _ = find_peaks(boundary_scores, height=threshold, distance=min_distance)
    return peaks


def boundaries_to_segments(
    boundary_indices: np.ndarray,
    timestamps: np.ndarray,
    min_segment_len_steps: int = STAGE1.min_segment_len_steps,
) -> List[Dict]:
    """Build a list of {start_idx, end_idx, start_time, end_time} segments
    from boundary indices, merging short segments into their neighbours.
    """
    T = len(timestamps)
    if T == 0:
        return []

    cut_points = sorted(set([0] + [int(b) + 1 for b in boundary_indices] + [T]))

    # Merge adjacent segments shorter than the minimum length.
    merged = [cut_points[0]]
    for cp in cut_points[1:]:
        if cp - merged[-1] < min_segment_len_steps and cp != T:
            continue  # skip this cut point, effectively merging
        merged.append(cp)
    if merged[-1] != T:
        merged[-1] = T

    segments = []
    for i in range(len(merged) - 1):
        start_idx, end_idx = merged[i], merged[i + 1]
        if start_idx >= end_idx:
            continue
        start_time = float(timestamps[start_idx])
        # end time = timestamp of the time-step after the last one in this
        # segment (or, for the final segment, extrapolate one step).
        if end_idx < T:
            end_time = float(timestamps[end_idx])
        else:
            step = float(timestamps[1] - timestamps[0]) if T > 1 else 1.0
            end_time = float(timestamps[-1] + step)

        segments.append({
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_time": start_time,
            "end_time": end_time,
        })
    return segments


def segments_from_scores(
    boundary_scores: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = None,
    min_distance: int = STAGE1.peak_min_distance_steps,
    min_segment_len_steps: int = STAGE1.min_segment_len_steps,
    adaptive_std: float = STAGE1.adaptive_threshold_std,
    time_step_sec: float = 3.0,
    max_segment_duration_sec: float = 300.0,
) -> List[Dict]:
    """Convenience wrapper: scores -> boundary indices -> segment list.

    `threshold=None` (default) uses the adaptive mean+std threshold — see
    `boundary_scores_to_boundaries`.

    If the model produces zero boundaries (e.g. undertrained model), a
    fallback kicks in that splits the video at the top score peaks to
    guarantee at most `max_segment_duration_sec` per segment.  This prevents
    the degenerate case of one giant segment per video.
    """
    boundary_idx = boundary_scores_to_boundaries(boundary_scores, threshold, min_distance, adaptive_std)

    # Fallback: if no boundaries found, force-split at the top score peaks
    T = len(timestamps)
    video_duration_sec = T * time_step_sec if T > 1 else 0
    min_segments = max(1, int(video_duration_sec / max_segment_duration_sec))

    if len(boundary_idx) == 0 and min_segments > 1 and len(boundary_scores) > 0:
        # Pick the top-scoring positions as forced boundaries
        n_needed = min_segments - 1
        # Only consider positions that are far enough apart
        sorted_idx = np.argsort(-boundary_scores)
        forced = []
        for idx in sorted_idx:
            # Check this position is far enough from all already-selected boundaries
            if all(abs(int(idx) - f) >= min_distance for f in forced):
                forced.append(int(idx))
            if len(forced) >= n_needed:
                break
        boundary_idx = np.array(sorted(forced), dtype=np.int64)

    return boundaries_to_segments(boundary_idx, timestamps, min_segment_len_steps)
