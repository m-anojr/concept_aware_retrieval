"""
Baseline 1 (Stage-1 comparison): fixed-window segmentation.

Splits a lecture video into uniform `window_sec`-second windows, ignoring
all content signals. Used as the simplest baseline for boundary F1 / IoU
evaluation of the Pedagogical Boundary Detector.
"""

from typing import Dict, List
import numpy as np


def fixed_window_segments(timestamps: np.ndarray, window_sec: float = 60.0) -> List[Dict]:
    """Produce segments of approximately `window_sec` duration.

    Returns the same {start_idx, end_idx, start_time, end_time} schema as
    `src.stage1_boundary.segment.boundaries_to_segments`, so it can be fed
    directly into the same evaluation code.
    """
    T = len(timestamps)
    if T == 0:
        return []

    duration = float(timestamps[-1] - timestamps[0])
    step = float(timestamps[1] - timestamps[0]) if T > 1 else 1.0
    n_windows = max(1, int(np.ceil((duration + step) / window_sec)))

    segments = []
    for w in range(n_windows):
        win_start_time = timestamps[0] + w * window_sec
        win_end_time = win_start_time + window_sec

        start_idx = int(np.searchsorted(timestamps, win_start_time, side="left"))
        end_idx = int(np.searchsorted(timestamps, win_end_time, side="left"))
        end_idx = max(end_idx, start_idx + 1)
        end_idx = min(end_idx, T)
        if start_idx >= T:
            break

        segments.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": float(timestamps[start_idx]),
            "end_time": float(timestamps[min(end_idx, T - 1)]) if end_idx < T else float(timestamps[-1] + step),
        })

    return segments
