"""
Training loop for the Cross-Modal Segment Fusion + Retrieval model (Stage 2).

Trains the SegmentFusionEncoder and QueryProjector jointly with a symmetric
InfoNCE loss over (query, segment) pairs, using in-batch negatives.

Usage
-----
    python -m src.stage2_retrieval.train \
        --annotations-dir data/annotations \
        --segments-dir data/segments \
        --checkpoint checkpoints/stage2_retrieval_model.pt \
        --fusion-mode cross_attention
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from config import FEATURES, STAGE2, PATHS, ensure_dirs
from src.stage2_retrieval.dataset import RetrievalPairDataset, list_annotated_video_ids
from src.stage2_retrieval.encoders import TextEncoder
from src.stage2_retrieval.fusion_model import CrossModalRetrievalModel


def info_nce_loss(query_z: torch.Tensor, seg_z: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of (query, segment) pairs using
    in-batch negatives.

    query_z, seg_z: (B, D), both L2-normalized.
    """
    logits = query_z @ seg_z.t() / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_q2s = F.cross_entropy(logits, labels)
    loss_s2q = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q2s + loss_s2q)


def run_epoch(model, loader, optimizer, device, temperature, train=True):
    model.train(train)
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        ocr_emb = batch["ocr_emb"].to(device)
        transcript_emb = batch["transcript_emb"].to(device)
        visual_emb = batch["visual_emb"].to(device)
        query_emb = batch["query_emb"].to(device)

        with torch.set_grad_enabled(train):
            seg_z = model.encode_segments(ocr_emb, transcript_emb, visual_emb)
            query_z = model.encode_queries(query_emb)
            loss = info_nce_loss(query_z, seg_z, temperature)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", default=PATHS.annotations_dir)
    parser.add_argument("--segments-dir", default=PATHS.segments_dir)
    parser.add_argument("--checkpoint", default=os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))
    parser.add_argument("--epochs", type=int, default=STAGE2.epochs)
    parser.add_argument("--batch-size", type=int, default=STAGE2.batch_size)
    parser.add_argument("--lr", type=float, default=STAGE2.lr)
    parser.add_argument("--temperature", type=float, default=STAGE2.temperature)
    parser.add_argument("--fusion-mode", default="cross_attention", choices=["cross_attention", "concat"])
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_ids = list_annotated_video_ids(args.annotations_dir)
    if len(video_ids) == 0:
        raise RuntimeError(f"No '*_queries.json' annotation files found in {args.annotations_dir}.")

    text_encoder = TextEncoder(device=args.device)
    full_ds = RetrievalPairDataset(video_ids, text_encoder, args.annotations_dir, args.segments_dir)
    if len(full_ds) == 0:
        raise RuntimeError("No (query, segment) pairs could be constructed. "
                            "Check that segment_features.py has been run for these videos.")

    n_val = max(1, int(len(full_ds) * args.val_split)) if len(full_ds) > 4 else 0
    n_train = len(full_ds) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = full_ds, None

    batch_size = min(args.batch_size, max(2, n_train))
    pin_memory = args.device.startswith('cuda') if isinstance(args.device, str) else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=(n_train > batch_size), pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory) if val_ds else None

    # Infer dimensionalities from the actual precomputed segment features
    # (Sentence-BERT dim for text, CLIP dim for visual) rather than assuming
    # config.py defaults. This keeps the training script robust to real data.
    sample = full_ds[0]
    ocr_dim = sample["ocr_emb"].shape[0]
    transcript_dim = sample["transcript_emb"].shape[0]
    visual_dim = sample["visual_emb"].shape[0]
    query_dim = sample["query_emb"].shape[0]
    print(f"Detected dims: ocr={ocr_dim} transcript={transcript_dim} visual={visual_dim} query={query_dim}")

    model = CrossModalRetrievalModel(
        ocr_dim=ocr_dim,
        transcript_dim=transcript_dim,
        visual_dim=visual_dim,
        query_dim=query_dim,
        hidden_dim=STAGE2.fusion_hidden_dim,
        out_dim=STAGE2.fusion_out_dim,
        num_heads=STAGE2.num_heads,
        dropout=STAGE2.dropout,
        fusion_mode=args.fusion_mode,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device, args.temperature, train=True)
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, args.device, args.temperature, train=False)
        else:
            val_loss = train_loss

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "ocr_dim": ocr_dim,
                    "transcript_dim": transcript_dim,
                    "visual_dim": visual_dim,
                    "query_dim": query_dim,
                    "hidden_dim": STAGE2.fusion_hidden_dim,
                    "out_dim": STAGE2.fusion_out_dim,
                    "num_heads": STAGE2.num_heads,
                    "dropout": STAGE2.dropout,
                    "fusion_mode": args.fusion_mode,
                },
            }, args.checkpoint)
            print(f"  -> saved best checkpoint to {args.checkpoint} (val_loss={val_loss:.4f})")

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    main()
