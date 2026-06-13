"""
Visual-change signal via dense optical flow / frame differencing.

For lecture videos the camera is largely static, so large optical-flow
magnitude usually corresponds to the instructor moving / gesturing / walking
to a new part of the board, which weakly correlates with topic transitions.
This is the third (and weakest on its own, but still useful) signal fused
into the Stage-1 feature stream - the ablation study in the report quantifies
its individual contribution.
"""

from typing import List, Tuple
import numpy as np


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        # Simple luma conversion, avoids a hard cv2 dependency for this step.
        return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.uint8)
    return frame


def extract_visual_change_signal(frames: List[np.ndarray]) -> np.ndarray:
    """Compute a normalized visual-change score per frame using optical flow.

    Falls back to mean absolute frame differencing if OpenCV's optical-flow
    implementation is unavailable.

    Returns
    -------
    signal : np.ndarray of shape (T,), values roughly in [0, 1] after
             min-max normalization across the video.
    """
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float32)

    raw = np.zeros(len(frames), dtype=np.float32)

    try:
        import cv2

        prev_gray = _to_gray(frames[0])
        for t in range(1, len(frames)):
            curr_gray = _to_gray(frames[t])
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            raw[t] = float(np.mean(mag))
            prev_gray = curr_gray
    except Exception:
        prev = _to_gray(frames[0]).astype(np.float32)
        for t in range(1, len(frames)):
            curr = _to_gray(frames[t]).astype(np.float32)
            raw[t] = float(np.mean(np.abs(curr - prev)))
            prev = curr

    # Min-max normalize to [0, 1] for stable fusion with the other signals.
    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin > 1e-8:
        signal = (raw - rmin) / (rmax - rmin)
    else:
        signal = np.zeros_like(raw)
    return signal.astype(np.float32)
