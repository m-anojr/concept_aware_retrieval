"""
PyTorch Dataset for Stage-1 boundary-detector training.

Each item corresponds to one lecture video's feature stream
(`data/features/<video_id>.npz`, produced by
`src.feature_extraction.feature_pipeline`). Pseudo-labels are computed on the
fly from the OCR / topic-drift / visual signals; if a manual annotation file
exists for the video (`data/annotations/<video_id>_boundaries.json`,
containing a list of boundary timestamps in seconds), it is merged in.

Because lecture videos vary in length, the dataset returns full sequences and
a custom `collate_fn` pads them to the same length within a batch (with a
padding mask passed to the Transformer encoder).
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import PATHS
from src.feature_extraction.feature_pipeline import load_feature_stream
from src.utils.io_utils import load_json
from src.stage1_boundary.pseudo_labels import (
    pseudo_boundaries_from_signals,
    boundaries_from_annotations,
    merge_boundary_labels,
    boundaries_to_segment_ids,
)


class BoundaryDataset(Dataset):
    def __init__(self, video_ids: List[str], features_dir: str = PATHS.features_dir,
                 annotations_dir: str = PATHS.annotations_dir):
        self.video_ids = video_ids
        self.features_dir = features_dir
        self.annotations_dir = annotations_dir

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        feats = load_feature_stream(os.path.join(self.features_dir, f"{video_id}.npz"))

        features = feats["features"]  # (T, D)
        timestamps = feats["timestamps"].astype(np.float32)
        ocr_signal = feats["ocr_signal"].astype(np.float32)
        topic_drift = feats["topic_drift_signal"].astype(np.float32)
        visual_signal = feats["visual_signal"].astype(np.float32)

        pseudo = pseudo_boundaries_from_signals(ocr_signal, topic_drift, visual_signal)

        manual = None
        ann_path = os.path.join(self.annotations_dir, f"{video_id}_boundaries.json")
        if os.path.exists(ann_path):
            ann = load_json(ann_path)
            manual = boundaries_from_annotations(timestamps, ann.get("boundary_times_sec", []))

        is_boundary = merge_boundary_labels(pseudo, manual)
        seg_ids = boundaries_to_segment_ids(is_boundary)

        return {
            "video_id": video_id,
            "features": torch.from_numpy(features),
            "is_boundary": torch.from_numpy(is_boundary).long(),
            "seg_ids": torch.from_numpy(seg_ids).long(),
            "timestamps": torch.from_numpy(timestamps),
        }


def collate_fn(batch):
    max_T = max(item["features"].shape[0] for item in batch)
    D = batch[0]["features"].shape[1]
    B = len(batch)

    features = torch.zeros(B, max_T, D)
    is_boundary = torch.zeros(B, max_T - 1, dtype=torch.long)
    seg_ids = torch.zeros(B, max_T, dtype=torch.long)
    key_padding_mask = torch.ones(B, max_T, dtype=torch.bool)  # True = pad
    lengths = torch.zeros(B, dtype=torch.long)
    video_ids = []

    for i, item in enumerate(batch):
        T = item["features"].shape[0]
        features[i, :T] = item["features"]
        is_boundary[i, : T - 1] = item["is_boundary"]
        # Pad seg_ids with a unique segment id so padded steps never count as
        # "same segment" with real ones during InfoNCE sampling.
        seg_ids[i, :T] = item["seg_ids"]
        if T < max_T:
            seg_ids[i, T:] = seg_ids[i, :T].max() + 100
        key_padding_mask[i, :T] = False
        lengths[i] = T
        video_ids.append(item["video_id"])

    return {
        "video_ids": video_ids,
        "features": features,
        "is_boundary": is_boundary,
        "seg_ids": seg_ids,
        "key_padding_mask": key_padding_mask,
        "lengths": lengths,
    }
