"""
Baseline 2 (Stage-1 comparison): shot-boundary / scene-change detection.

Uses PySceneDetect's content-aware detector to find visual scene cuts in the
raw video. For lecture videos (single static camera, slow visual change),
this typically produces very few or very irregular boundaries, illustrating
why generic shot detection is a poor fit for lecture segmentation - exactly
the gap motivating the Pedagogical Boundary Detector.

If PySceneDetect (or the video file) is unavailable, falls back to detecting
boundaries from large jumps in the (already-computed) visual-change signal,
so the comparison can still run on precomputed features.
"""

from typing import Dict, List, Optional
import numpy as np

from src.stage1_boundary.segment import boundaries_to_segments


def shot_segments_from_video(video_path: str, timestamps: np.ndarray,
                              threshold: float = 27.0) -> List[Dict]:
    """Run PySceneDetect's ContentDetector on `video_path` and map detected
    cut times onto the nearest feature time-step boundaries."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        cut_times = [scene[0].get_seconds() for scene in scene_list[1:]]
        return _segments_from_cut_times(cut_times, timestamps)
    except Exception:
        return shot_segments_from_visual_signal(timestamps, None)


def shot_segments_from_visual_signal(timestamps: np.ndarray,
                                      visual_signal: Optional[np.ndarray],
                                      z_thresh: float = 1.5) -> List[Dict]:
    """Fallback: treat large z-scored jumps in the precomputed visual-change
    signal as shot boundaries."""
    T = len(timestamps)
    if visual_signal is None or T < 3:
        # No signal available: return one big segment covering the video.
        return boundaries_to_segments(np.array([], dtype=np.int64), timestamps, min_segment_len_steps=1)

    mu, sigma = visual_signal.mean(), visual_signal.std() + 1e-8
    z = (visual_signal - mu) / sigma
    boundary_idx = np.where(z[1:] > z_thresh)[0]
    return boundaries_to_segments(boundary_idx, timestamps, min_segment_len_steps=1)


def _segments_from_cut_times(cut_times: List[float], timestamps: np.ndarray) -> List[Dict]:
    boundary_idx = []
    for ct in cut_times:
        idx = int(np.searchsorted(timestamps, ct)) - 1
        idx = max(0, min(idx, len(timestamps) - 2))
        boundary_idx.append(idx)
    return boundaries_to_segments(np.array(sorted(set(boundary_idx)), dtype=np.int64),
                                   timestamps, min_segment_len_steps=1)
