"""
Baseline 3 (Stage-2 comparison): sliding-window retrieval without
concept-aware segmentation.

Instead of using Stage-1's learned concept-coherent segments, this baseline
splits each video into fixed-length sliding windows (with optional overlap),
embeds each window directly with the frozen CLIP/Sentence-BERT encoders
(simple mean-pool / concatenation, no trainable fusion), and performs the
same cosine-similarity search.

This isolates the contribution of (a) concept-aware segmentation and (b) the
learned Segment Fusion Encoder - both ablations described in the proposal.
"""

import os
from typing import Dict, List

import numpy as np
import torch

from config import PATHS, STAGE2
from src.baselines.fixed_window import fixed_window_segments
from src.stage2_retrieval.encoders import TextEncoder
from src.utils.io_utils import load_features


def build_sliding_window_index(video_ids: List[str], window_sec: float = 60.0,
                                stride_sec: float = 30.0,
                                features_dir: str = PATHS.features_dir,
                                device: str = None):
    """Build an in-memory index of sliding-window segments.

    Each window's representation is the simple concatenation of:
      - mean Sentence-BERT embedding of OCR text in the window
      - mean Sentence-BERT embedding of transcript text in the window
      - mean CLIP embedding of the window
    L2-normalized after concatenation, so cosine similarity in this
    concatenated space is the retrieval score (no trainable fusion).

    Returns
    -------
    embeddings : np.ndarray (N, D)
    metadata   : List[Dict] with video_id, start_time, end_time
    """
    text_encoder = TextEncoder(device=device)
    embeddings, metadata = [], []

    for video_id in video_ids:
        feats = load_features(os.path.join(features_dir, f"{video_id}.npz"))
        timestamps = feats["timestamps"]
        ocr_texts = feats["ocr_texts"]
        transcript_texts = feats["transcript_texts"]
        clip_embeds = feats["clip_embeds"]

        T = len(timestamps)
        if T == 0:
            continue
        step = float(timestamps[1] - timestamps[0]) if T > 1 else 1.0
        window_steps = max(1, int(round(window_sec / step)))
        stride_steps = max(1, int(round(stride_sec / step)))

        for start_idx in range(0, T, stride_steps):
            end_idx = min(start_idx + window_steps, T)
            if start_idx >= end_idx:
                continue

            ocr_text = " ".join(str(t) for t in ocr_texts[start_idx:end_idx] if t)
            transcript_text = " ".join(str(t) for t in transcript_texts[start_idx:end_idx] if t)

            ocr_emb = text_encoder.encode([ocr_text])[0]
            transcript_emb = text_encoder.encode([transcript_text])[0]
            visual_emb = clip_embeds[start_idx:end_idx].mean(axis=0)

            concat = np.concatenate([ocr_emb, transcript_emb, visual_emb]).astype(np.float32)
            concat = concat / (np.linalg.norm(concat) + 1e-8)

            embeddings.append(concat)
            metadata.append({
                "video_id": video_id,
                "start_time": float(timestamps[start_idx]),
                "end_time": float(timestamps[end_idx - 1] + step),
            })

            if end_idx >= T:
                break

    embeddings = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 1), dtype=np.float32)
    return embeddings, metadata


def search_sliding_window(query: str, embeddings: np.ndarray, metadata: List[Dict],
                           top_k: int = 5,
                           device: str = None) -> List[Dict]:
    """Encode the query in the same concatenated space and rank windows by
    cosine similarity (the embedding dimensions for query text are matched by
    concatenating [query_emb, query_emb, zeros(clip_dim)] -> this mirrors the
    common "embed query with text tower only" setup of CLIP-style sliding
    window baselines)."""
    from config import FEATURES

    text_encoder = TextEncoder(device=device)
    q_emb = text_encoder.encode([query])[0]
    clip_dim = embeddings.shape[1] - 2 * FEATURES.text_embed_dim
    q_concat = np.concatenate([q_emb, q_emb, np.zeros(clip_dim, dtype=np.float32)])
    q_concat = q_concat / (np.linalg.norm(q_concat) + 1e-8)

    scores = embeddings @ q_concat
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_idx:
        meta = metadata[idx]
        results.append({**meta, "score": float(scores[idx])})
    return results
