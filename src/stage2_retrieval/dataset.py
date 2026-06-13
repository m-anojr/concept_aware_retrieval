"""
PyTorch Dataset for Stage-2 (query, segment) contrastive training.

Expected annotation format (`data/annotations/<video_id>_queries.json`):

    {
      "pairs": [
        {"query": "explain backpropagation with an example",
         "start_time": 612.0, "end_time": 730.0},
        ...
      ]
    }

For each (query, start_time, end_time) pair, the dataset finds the segment
(from `data/segments/<video_id>_segment_features.npz`, produced by Stage 1 +
`segment_features.py`) with maximum temporal IoU against [start_time,
end_time], and pairs the query with that segment's precomputed
(ocr_emb, transcript_emb, visual_emb).

Query text is encoded once at dataset construction time with the frozen
Sentence-BERT text encoder.
"""

import glob
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from config import PATHS
from src.utils.io_utils import load_json
from src.stage2_retrieval.segment_features import load_segment_features


def temporal_iou(a_start, a_end, b_start, b_end) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


class RetrievalPairDataset(Dataset):
    def __init__(self, video_ids: List[str], text_encoder,
                 annotations_dir: str = PATHS.annotations_dir,
                 segments_dir: str = PATHS.segments_dir):
        self.samples = []  # list of dicts: ocr_emb, transcript_emb, visual_emb, query_emb

        seg_cache = {}
        for video_id in video_ids:
            ann_path = os.path.join(annotations_dir, f"{video_id}_queries.json")
            if not os.path.exists(ann_path):
                continue
            ann = load_json(ann_path)

            if video_id not in seg_cache:
                seg_cache[video_id] = load_segment_features(video_id, segments_dir)
            seg_feats = seg_cache[video_id]

            starts, ends = seg_feats["start_times"], seg_feats["end_times"]
            if len(starts) == 0:
                continue

            queries = [p["query"] for p in ann["pairs"]]
            query_embs = text_encoder.encode(queries)

            for pair, q_emb in zip(ann["pairs"], query_embs):
                ious = [temporal_iou(pair["start_time"], pair["end_time"], s, e)
                        for s, e in zip(starts, ends)]
                best_idx = int(np.argmax(ious))

                self.samples.append({
                    "ocr_emb": seg_feats["ocr_emb"][best_idx],
                    "transcript_emb": seg_feats["transcript_emb"][best_idx],
                    "visual_emb": seg_feats["visual_emb"][best_idx],
                    "query_emb": q_emb,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "ocr_emb": torch.from_numpy(s["ocr_emb"].astype(np.float32)),
            "transcript_emb": torch.from_numpy(s["transcript_emb"].astype(np.float32)),
            "visual_emb": torch.from_numpy(s["visual_emb"].astype(np.float32)),
            "query_emb": torch.from_numpy(s["query_emb"].astype(np.float32)),
        }


def list_annotated_video_ids(annotations_dir: str = PATHS.annotations_dir) -> List[str]:
    paths = sorted(glob.glob(os.path.join(annotations_dir, "*_queries.json")))
    return [os.path.basename(p).replace("_queries.json", "") for p in paths]
