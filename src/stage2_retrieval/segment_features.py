"""
Aggregate per-time-step features (from `data/features/<video_id>.npz`) into
per-segment features (using the segment boundaries produced by Stage 1, in
`data/segments/<video_id>.json`):

  - OCR text: concatenate the OCR text of every time-step in the segment,
    then encode with Sentence-BERT -> e_ocr (text_embed_dim,)
  - Transcript text: concatenate the ASR transcript text of every time-step
    in the segment, then encode with Sentence-BERT -> e_transcript (text_embed_dim,)
  - Visual: mean-pool the CLIP embeddings of the time-steps in the segment
    (a cheap stand-in for "representative keyframe(s)") -> e_visual (clip_dim,)

The resulting per-segment feature bundle is saved to
`data/segments/<video_id>_segment_features.npz` and is consumed by both
`index_builder.py` (to build the searchable FAISS index) and
`dataset.py` (to build (query, segment) training pairs).
"""

import os
from typing import Dict, List

import numpy as np

from config import FEATURES, PATHS
from src.feature_extraction.feature_pipeline import project_clip_embeddings
from src.utils.io_utils import load_features, load_json, save_features


def compute_segment_features(
    video_id: str,
    text_encoder,
    features_dir: str = PATHS.features_dir,
    segments_dir: str = PATHS.segments_dir,
) -> Dict:
    feats = load_features(os.path.join(features_dir, f"{video_id}.npz"))
    seg_data = load_json(os.path.join(segments_dir, f"{video_id}.json"))
    segments: List[Dict] = seg_data["segments"]

    ocr_texts = feats["ocr_texts"]
    transcript_texts = feats["transcript_texts"]
    clip_embeds = project_clip_embeddings(feats["clip_embeds"], FEATURES.clip_dim)

    ocr_agg, transcript_agg, visual_agg = [], [], []
    starts, ends = [], []

    for seg in segments:
        s, e = seg["start_idx"], seg["end_idx"]
        s, e = max(0, s), min(len(ocr_texts), max(e, s + 1))

        ocr_agg.append(" ".join(str(t) for t in ocr_texts[s:e] if t))
        transcript_agg.append(" ".join(str(t) for t in transcript_texts[s:e] if t))
        visual_agg.append(clip_embeds[s:e].mean(axis=0) if e > s else clip_embeds[s])
        starts.append(seg["start_time"])
        ends.append(seg["end_time"])

    ocr_emb = text_encoder.encode(ocr_agg)
    transcript_emb = text_encoder.encode(transcript_agg)
    visual_emb = np.stack(visual_agg, axis=0) if visual_agg else np.zeros((0, clip_embeds.shape[1]), dtype=np.float32)

    result = dict(
        video_id=video_id,
        start_times=np.array(starts, dtype=np.float32),
        end_times=np.array(ends, dtype=np.float32),
        ocr_emb=ocr_emb,
        transcript_emb=transcript_emb,
        visual_emb=visual_emb,
        ocr_text=np.array(ocr_agg, dtype=object),
        transcript_text=np.array(transcript_agg, dtype=object),
    )

    out_path = os.path.join(segments_dir, f"{video_id}_segment_features.npz")
    save_features(out_path, **result)
    return result


def load_segment_features(video_id: str, segments_dir: str = PATHS.segments_dir) -> Dict:
    feats = load_features(os.path.join(segments_dir, f"{video_id}_segment_features.npz"))
    if feats["visual_emb"].shape[1] != FEATURES.clip_dim:
        feats["visual_emb"] = project_clip_embeddings(feats["visual_emb"], FEATURES.clip_dim)
    return feats
