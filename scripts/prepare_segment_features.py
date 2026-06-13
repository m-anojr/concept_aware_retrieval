"""
Compute per-segment OCR/transcript/visual embeddings
(`data/segments/<video_id>_segment_features.npz`) for every video that has
both a feature stream (`data/features/<video_id>.npz`) and a segmentation
(`data/segments/<video_id>.json`, produced by `segment_videos.py`).

These per-segment features are the input to Stage 2's SegmentFusionEncoder
(for both training-pair construction and FAISS index building).

Usage
-----
    python scripts/prepare_segment_features.py
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import PATHS, ensure_dirs
from src.stage2_retrieval.encoders import TextEncoder
from src.stage2_retrieval.segment_features import compute_segment_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=PATHS.features_dir)
    parser.add_argument("--segments-dir", default=PATHS.segments_dir)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run GPU-accelerated models on. Use cuda or cpu.")
    args = parser.parse_args()

    ensure_dirs()

    seg_paths = sorted(glob.glob(os.path.join(args.segments_dir, "*.json")))
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in seg_paths
                  if not os.path.basename(p).endswith("_segment_features.json")]

    if not video_ids:
        print(f"No segmentation files found in {args.segments_dir}. Run segment_videos.py first.")
        return

    text_encoder = TextEncoder(device=args.device)

    for video_id in video_ids:
        result = compute_segment_features(video_id, text_encoder, args.features_dir, args.segments_dir)
        print(f"{video_id}: {len(result['start_times'])} segment feature vectors")


if __name__ == "__main__":
    main()
