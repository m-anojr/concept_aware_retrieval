"""
Evaluation metrics for Stage 2 (Cross-Modal Segment Retrieval) against
manually annotated (query, ground-truth segment) pairs.

  - Recall@k: fraction of queries for which at least one of the top-k
    retrieved segments overlaps the ground-truth segment with IoU >=
    `iou_threshold`.
  - Mean IoU: average IoU between the top-1 retrieved segment and the
    ground-truth segment, across all queries.
"""

from typing import Dict, List
import numpy as np

from config import EVAL


def temporal_iou(a_start, a_end, b_start, b_end) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def recall_at_k(retrieved: List[List[Dict]], ground_truth: List[Dict],
                 k: int, iou_threshold: float = EVAL.retrieval_iou_threshold) -> float:
    """
    retrieved: for each query, a ranked list of result dicts with
               video_id/start_time/end_time (already truncated/sliced to top-k
               by the caller, or full list - this function takes [:k]).
    ground_truth: for each query, a dict with video_id/start_time/end_time.
    """
    if not retrieved:
        return 0.0

    hits = 0
    for results, gt in zip(retrieved, ground_truth):
        hit = False
        for r in results[:k]:
            if r["video_id"] != gt["video_id"]:
                continue
            iou = temporal_iou(r["start_time"], r["end_time"], gt["start_time"], gt["end_time"])
            if iou >= iou_threshold:
                hit = True
                break
        hits += int(hit)
    return hits / len(retrieved)


def mean_iou_at_1(retrieved: List[List[Dict]], ground_truth: List[Dict]) -> float:
    if not retrieved:
        return 0.0
    ious = []
    for results, gt in zip(retrieved, ground_truth):
        if not results:
            ious.append(0.0)
            continue
        top = results[0]
        if top["video_id"] != gt["video_id"]:
            ious.append(0.0)
            continue
        ious.append(temporal_iou(top["start_time"], top["end_time"], gt["start_time"], gt["end_time"]))
    return float(np.mean(ious))


def evaluate_retrieval(retrieved: List[List[Dict]], ground_truth: List[Dict],
                        ks: List[int] = None, iou_threshold: float = EVAL.retrieval_iou_threshold) -> Dict[str, float]:
    ks = ks or EVAL.retrieval_ks
    metrics = {f"recall@{k}": recall_at_k(retrieved, ground_truth, k, iou_threshold) for k in ks}
    metrics["mean_iou@1"] = mean_iou_at_1(retrieved, ground_truth)
    return metrics
