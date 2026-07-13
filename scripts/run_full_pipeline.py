"""
One-command pipeline runner: extracts features, segments videos, computes
segment features, generates pseudo-queries, trains Stage 2, and builds the
FAISS index — everything needed to go from raw videos to a working search.

Usage
-----
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --force          # re-do every step
    python scripts/run_full_pipeline.py --skip-training   # skip model training
"""

import argparse
import glob
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import PATHS, ensure_dirs

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def _elapsed(start):
    return f"{time.time() - start:.1f}s"


def step_extract_features(args):
    from src.feature_extraction.feature_pipeline import build_feature_stream

    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        video_paths.extend(glob.glob(os.path.join(PATHS.raw_videos_dir, f"*{ext}")))
    video_paths = sorted(video_paths)

    if not video_paths:
        print(f"  [!] No videos found in {PATHS.raw_videos_dir}")
        return False

    for vp in video_paths:
        vid_id = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(PATHS.features_dir, f"{vid_id}.npz")
        if os.path.exists(out_path) and not args.force:
            print(f"  [OK] {vid_id} — features already exist, skipping")
            continue
        t0 = time.time()
        result = build_feature_stream(vp, save=True, device=args.device)
        print(f"  [OK] {vid_id} — {len(result['timestamps'])} steps ({_elapsed(t0)})")
    return True


def step_segment_videos(args):
    from src.stage1_boundary.inference import load_stage1_model, segment_and_save

    ckpt = os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt")
    if not os.path.exists(ckpt):
        print(f"  [!] No Stage 1 checkpoint at {ckpt} — using pseudo-label segmentation")
        # Fallback: use pseudo-label-based segmentation directly
        _segment_with_pseudo_labels(args)
        return True

    model = load_stage1_model(ckpt, args.device)

    paths = sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    for vid_id in video_ids:
        out_path = os.path.join(PATHS.segments_dir, f"{vid_id}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"  [OK] {vid_id} — segments already exist, skipping")
            continue
        segments = segment_and_save(vid_id, model, PATHS.features_dir,
                                    PATHS.segments_dir, args.device)
        print(f"  [OK] {vid_id} — {len(segments)} segments")
    return True


def _segment_with_pseudo_labels(args):
    """Segment videos using pseudo-label boundaries when no trained model exists."""
    import numpy as np
    from src.utils.io_utils import load_features, save_json, segments_path_for
    from src.stage1_boundary.pseudo_labels import (
        pseudo_boundaries_from_signals, boundaries_to_segment_ids,
    )
    from src.stage1_boundary.segment import boundaries_to_segments

    paths = sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))
    for path in paths:
        vid_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(PATHS.segments_dir, f"{vid_id}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"  [OK] {vid_id} — segments already exist, skipping")
            continue

        feats = load_features(path)
        timestamps = feats["timestamps"].astype(np.float32)
        ocr_signal = feats["ocr_signal"].astype(np.float32)
        topic_drift = feats["topic_drift_signal"].astype(np.float32)
        visual_signal = feats["visual_signal"].astype(np.float32)

        is_boundary = pseudo_boundaries_from_signals(
            ocr_signal, topic_drift, visual_signal,
        )
        boundary_idx = np.where(is_boundary == 1)[0]
        segments = boundaries_to_segments(boundary_idx, timestamps)

        save_json({"video_id": vid_id, "segments": segments},
                  segments_path_for(vid_id, PATHS.segments_dir))
        print(f"  [OK] {vid_id} — {len(segments)} segments (pseudo-label)")


def step_segment_features(args):
    from src.stage2_retrieval.encoders import TextEncoder
    from src.stage2_retrieval.segment_features import compute_segment_features

    seg_paths = sorted(glob.glob(os.path.join(PATHS.segments_dir, "*.json")))
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in seg_paths
                 if not os.path.basename(p).endswith("_segment_features.json")]

    if not video_ids:
        print("  [!] No segmentation files found")
        return False

    text_encoder = TextEncoder(device=args.device)
    for vid_id in video_ids:
        out_path = os.path.join(PATHS.segments_dir, f"{vid_id}_segment_features.npz")
        if os.path.exists(out_path) and not args.force:
            print(f"  [OK] {vid_id} — segment features exist, skipping")
            continue
        t0 = time.time()
        result = compute_segment_features(vid_id, text_encoder,
                                          PATHS.features_dir, PATHS.segments_dir)
        print(f"  [OK] {vid_id} — {len(result['start_times'])} segment vectors ({_elapsed(t0)})")
    return True


def step_pseudo_queries(args):
    from src.stage2_retrieval.segment_features import load_segment_features
    from src.utils.io_utils import save_json

    paths = sorted(glob.glob(os.path.join(PATHS.segments_dir, "*_segment_features.npz")))
    video_ids = [os.path.basename(p).replace("_segment_features.npz", "") for p in paths]

    if not video_ids:
        print("  [!] No segment feature files found")
        return False

    total = 0
    for vid_id in video_ids:
        out_path = os.path.join(PATHS.annotations_dir, f"{vid_id}_queries.json")
        if os.path.exists(out_path) and not args.force:
            print(f"  [OK] {vid_id} — queries already exist, skipping")
            continue

        feats = load_segment_features(vid_id, PATHS.segments_dir)
        starts, ends = feats["start_times"], feats["end_times"]
        transcript_text = feats["transcript_text"]
        ocr_text = feats["ocr_text"]

        pairs = []
        for i in range(len(starts)):
            text = str(transcript_text[i]).strip() or str(ocr_text[i]).strip()
            if not text:
                continue
            words = text.split()
            query = " ".join(words[:12]).strip()
            if not query:
                continue
            pairs.append({
                "query": query,
                "start_time": float(starts[i]),
                "end_time": float(ends[i]),
            })
        save_json({"pairs": pairs}, out_path)
        total += len(pairs)
        print(f"  [OK] {vid_id} — {len(pairs)} pseudo-queries")

    print(f"  Total pseudo-query pairs: {total}")
    return total > 0


def step_train_stage2(args):
    """Train Stage 2 model."""
    from src.stage2_retrieval.dataset import RetrievalPairDataset, list_annotated_video_ids
    from src.stage2_retrieval.encoders import TextEncoder
    from src.stage2_retrieval.fusion_model import CrossModalRetrievalModel
    from src.stage2_retrieval.train import info_nce_loss, run_epoch
    from config import STAGE2
    import random

    video_ids = list_annotated_video_ids(PATHS.annotations_dir)
    if not video_ids:
        print("  [!] No annotation files found, skipping training")
        return False

    ckpt_path = os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt")
    if os.path.exists(ckpt_path) and not args.force:
        print(f"  [OK] Stage 2 checkpoint exists, skipping training")
        return True

    random.seed(42)
    torch.manual_seed(42)

    text_encoder = TextEncoder(device=args.device)
    full_ds = RetrievalPairDataset(video_ids, text_encoder,
                                   PATHS.annotations_dir, PATHS.segments_dir)
    if len(full_ds) == 0:
        print("  [!] No (query, segment) pairs, skipping training")
        return False

    from torch.utils.data import DataLoader, random_split

    n_val = max(1, int(len(full_ds) * 0.15)) if len(full_ds) > 4 else 0
    n_train = len(full_ds) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))
    else:
        train_ds, val_ds = full_ds, None

    batch_size = min(STAGE2.batch_size, max(2, n_train))
    pin_mem = args.device.startswith("cuda") if isinstance(args.device, str) else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=(n_train > batch_size), pin_memory=pin_mem)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
                  if val_ds else None)

    sample = full_ds[0]
    ocr_dim = sample["ocr_emb"].shape[0]
    transcript_dim = sample["transcript_emb"].shape[0]
    visual_dim = sample["visual_emb"].shape[0]
    query_dim = sample["query_emb"].shape[0]

    model = CrossModalRetrievalModel(
        ocr_dim=ocr_dim, transcript_dim=transcript_dim,
        visual_dim=visual_dim, query_dim=query_dim,
        hidden_dim=STAGE2.fusion_hidden_dim, out_dim=STAGE2.fusion_out_dim,
        num_heads=STAGE2.num_heads, dropout=STAGE2.dropout,
        fusion_mode="cross_attention",
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE2.lr)

    epochs = min(STAGE2.epochs, 20)
    best_val = float("inf")
    
    # Track losses for AI/ML presentation plotting
    history_train = []
    history_val = []
    
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device,
                               STAGE2.temperature, train=True)
        val_loss = (run_epoch(model, val_loader, optimizer, args.device,
                              STAGE2.temperature, train=False)
                    if val_loader else train_loss)

        history_train.append(train_loss)
        history_val.append(val_loss)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:3d}/{epochs} — train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "ocr_dim": ocr_dim, "transcript_dim": transcript_dim,
                    "visual_dim": visual_dim, "query_dim": query_dim,
                    "hidden_dim": STAGE2.fusion_hidden_dim,
                    "out_dim": STAGE2.fusion_out_dim,
                    "num_heads": STAGE2.num_heads,
                    "dropout": STAGE2.dropout,
                    "fusion_mode": "cross_attention",
                },
            }, ckpt_path)

    # Generate a beautiful matplotlib plot for the presentation
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), history_train, label="Train Loss (InfoNCE)", marker='o')
        if val_loader:
            plt.plot(range(1, epochs + 1), history_val, label="Validation Loss", marker='s')
        plt.title("Stage 2: Cross-Modal Retrieval Model Training")
        plt.xlabel("Epoch")
        plt.ylabel("Contrastive Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_path = os.path.join(PATHS.data_dir, "training_loss_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved training loss curve plot to: {plot_path} (Perfect for your presentation!)")
    except Exception as e:
        print(f"  [!] Could not plot loss curve: {e}")

    print(f"  [OK] Stage 2 trained — best val loss: {best_val:.4f}")
    return True


def step_build_index(args):
    from src.stage2_retrieval.index_builder import build_index

    ckpt = os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt")
    if not os.path.exists(ckpt):
        print("  [!] No Stage 2 checkpoint — cannot build FAISS index")
        return False

    t0 = time.time()
    build_index(PATHS.segments_dir, ckpt, PATHS.index_dir, args.device)
    print(f"  [OK] Index built ({_elapsed(t0)})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full Lecture Retrieval pipeline.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_dirs()
    total_start = time.time()

    steps = [
        ("1/6  Extract features", step_extract_features),
        ("2/6  Segment videos", step_segment_videos),
        ("3/6  Compute segment features", step_segment_features),
        ("4/6  Generate pseudo-queries", step_pseudo_queries),
    ]

    if not args.skip_training:
        steps.append(("5/6  Train Stage 2 model", step_train_stage2))
    else:
        print("\n[SKIP] Skipping training (--skip-training)")

    steps.append(("6/6  Build FAISS index", step_build_index))

    for label, fn in steps:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        t0 = time.time()
        ok = fn(args)
        print(f"  [TIME] Step took {_elapsed(t0)}")
        if not ok:
            print(f"  [!] Step had issues, continuing...")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! Total time: {_elapsed(total_start)}")
    print(f"{'='*60}")
    print(f"\nTo start the search app:")
    print(f"  uvicorn app.backend.main:app --reload --port 8000")
    print(f"  Then open http://localhost:8000")


if __name__ == "__main__":
    main()
