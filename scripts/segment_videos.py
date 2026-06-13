"""
Run the trained Pedagogical Boundary Detector over every video in
`data/features/` and write concept-coherent segments to `data/segments/`.

Usage
-----
    python scripts/segment_videos.py \
        --checkpoint checkpoints/stage1_boundary_detector.pt
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import PATHS, STAGE1, ensure_dirs
from src.stage1_boundary.inference import load_stage1_model, segment_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=PATHS.features_dir)
    parser.add_argument("--segments-dir", default=PATHS.segments_dir)
    parser.add_argument("--checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt"))
    parser.add_argument("--threshold", type=float, default=None,
                         help="Fixed boundary-score threshold. Default: adaptive (mean + std).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_dirs()

    model = load_stage1_model(args.checkpoint, args.device)

    paths = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    if not video_ids:
        print(f"No feature files found in {args.features_dir}.")
        return

    for video_id in video_ids:
        segments = segment_and_save(video_id, model, args.features_dir, args.segments_dir, args.device,
                                     threshold=args.threshold)
        print(f"{video_id}: {len(segments)} concept-coherent segments")


if __name__ == "__main__":
    main()
