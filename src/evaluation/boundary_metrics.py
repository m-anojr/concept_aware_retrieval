"""
Evaluation metrics for Stage 1 (Pedagogical Boundary Detector) against
manually annotated ground-truth concept boundaries.

  - Boundary F1: precision/recall of predicted boundary points against
    ground-truth boundary points, with a tolerance window (in time-steps).
  - Segment IoU: for each ground-truth segment, the Intersection-over-Union
    with its best-matching predicted segment, averaged (mean IoU).
"""

from typing import Dict, List
import numpy as np

from config import EVAL


def boundary_f1(pred_boundary_idx: np.ndarray, gt_boundary_idx: np.ndarray,
                 tolerance: int = EVAL.boundary_tolerance_steps) -> Dict[str, float]:
    """Compute precision, recall and F1 for predicted vs. ground-truth
    boundary indices, matching within +/- `tolerance` time-steps (each
    ground-truth boundary can be matched at most once).
    """
    pred = sorted(int(p) for p in pred_boundary_idx)
    gt = sorted(int(g) for g in gt_boundary_idx)

    matched_gt = set()
    tp = 0
    for p in pred:
        for g in gt:
            if g in matched_gt:
                continue
            if abs(p - g) <= tolerance:
                matched_gt.add(g)
                tp += 1
                break

    fp = len(pred) - tp
    fn = len(gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(gt) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(pred) == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def segment_iou(seg_a: Dict, seg_b: Dict) -> float:
    """IoU between two segments given as {"start_time": ..., "end_time": ...}."""
    inter = max(0.0, min(seg_a["end_time"], seg_b["end_time"]) - max(seg_a["start_time"], seg_b["start_time"]))
    union = max(seg_a["end_time"], seg_b["end_time"]) - min(seg_a["start_time"], seg_b["start_time"])
    return inter / union if union > 0 else 0.0


def mean_segment_iou(pred_segments: List[Dict], gt_segments: List[Dict]) -> float:
    """For each ground-truth segment, find the predicted segment with the
    highest IoU and average these best-match IoUs across all ground-truth
    segments."""
    if not gt_segments:
        return 0.0
    if not pred_segments:
        return 0.0

    ious = []
    for gt_seg in gt_segments:
        best = max(segment_iou(gt_seg, pred_seg) for pred_seg in pred_segments)
        ious.append(best)
    return float(np.mean(ious))


def evaluate_boundaries(pred_segments: List[Dict], gt_segments: List[Dict],
                         total_steps: int, tolerance: int = EVAL.boundary_tolerance_steps) -> Dict[str, float]:
    """Convenience wrapper combining boundary F1 and mean segment IoU.

    `pred_segments` / `gt_segments` use the {start_idx, end_idx, start_time,
    end_time} schema produced by `boundaries_to_segments`. Internal
    boundaries are the `end_idx` of every segment except the last.
    """
    pred_boundaries = np.array([s["end_idx"] for s in pred_segments[:-1]], dtype=np.int64) if len(pred_segments) > 1 else np.array([], dtype=np.int64)
    gt_boundaries = np.array([s["end_idx"] for s in gt_segments[:-1]], dtype=np.int64) if len(gt_segments) > 1 else np.array([], dtype=np.int64)

    f1_metrics = boundary_f1(pred_boundaries, gt_boundaries, tolerance)
    miou = mean_segment_iou(pred_segments, gt_segments)

    return {**f1_metrics, "mean_iou": miou}
