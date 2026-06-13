"""
Query-time retrieval: encode a free-text student query, search the FAISS
index of segment embeddings, and return the top-k ranked
(video_id, start_time, end_time, ...) results with "jump to moment" links.

When the FAISS model returns low-confidence results (which happens when the
Stage 2 model hasn't been well-trained), a fallback text-similarity search
is used: direct Sentence-BERT cosine similarity between the query and each
segment's stored transcript/OCR text.  The two scores are combined for the
final ranking, ensuring search works reasonably even before the model is
properly trained.
"""

import os
from typing import List, Dict

import numpy as np
import torch

from config import PATHS
from src.stage2_retrieval.encoders import TextEncoder
from src.stage2_retrieval.index_builder import load_stage2_model
from src.utils.io_utils import load_json, format_timestamp


class LectureRetriever:
    def __init__(self, index_dir: str = PATHS.index_dir,
                 checkpoint_path: str = None,
                 device: str = "cpu"):
        import faiss

        checkpoint_path = checkpoint_path or os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt")

        self.device = device
        self.text_encoder = TextEncoder(device=device)

        # Load FAISS index and metadata
        self.index = faiss.read_index(os.path.join(index_dir, "segments.index"))
        self.metadata: List[Dict] = load_json(os.path.join(index_dir, "segments_metadata.json"))

        # Try to load the trained Stage 2 model; if missing, use text-only search
        self._model = None
        if os.path.exists(checkpoint_path):
            try:
                self._model = load_stage2_model(checkpoint_path, device)
            except Exception as e:
                print(f"[LectureRetriever] Could not load Stage 2 model: {e}")
                print("[LectureRetriever] Falling back to text-similarity search.")

    def _text_similarity_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback: rank segments by direct Sentence-BERT cosine similarity
        between the query and each segment's transcript + OCR text."""
        query_emb = self.text_encoder.encode([query])[0]  # (D,)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        scores = []
        for meta in self.metadata:
            # Combine transcript and OCR text for matching
            segment_text = (meta.get("transcript_text", "") + " " +
                            meta.get("ocr_text", "")).strip()
            if not segment_text:
                scores.append(0.0)
                continue
            seg_emb = self.text_encoder.encode([segment_text])[0]
            seg_emb = seg_emb / (np.linalg.norm(seg_emb) + 1e-8)
            scores.append(float(np.dot(query_emb, seg_emb)))

        scores = np.array(scores)
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            meta = self.metadata[idx]
            results.append({
                "video_id": meta["video_id"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "start_time_str": format_timestamp(meta["start_time"]),
                "end_time_str": format_timestamp(meta["end_time"]),
                "ocr_text": meta.get("ocr_text", ""),
                "transcript_text": meta.get("transcript_text", ""),
                "score": float(scores[idx]),
                "search_mode": "text_similarity",
            })
        return results

    def search(self, query: str, top_k: int = 5,
               video_url_template: str = "{video_id}#t={start}") -> List[Dict]:
        top_k = min(top_k, max(1, self.index.ntotal))

        # --- FAISS model-based search ---
        faiss_results = []
        if self._model is not None:
            query_emb = self.text_encoder.encode([query])  # (1, text_embed_dim)
            query_tensor = torch.from_numpy(query_emb.astype(np.float32)).to(self.device)

            with torch.no_grad():
                query_z = self._model.encode_queries(query_tensor)
            query_z_np = query_z.cpu().numpy().astype(np.float32)

            scores, indices = self.index.search(query_z_np, top_k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                meta = self.metadata[idx]
                faiss_results.append({
                    "video_id": meta["video_id"],
                    "start_time": meta["start_time"],
                    "end_time": meta["end_time"],
                    "start_time_str": format_timestamp(meta["start_time"]),
                    "end_time_str": format_timestamp(meta["end_time"]),
                    "ocr_text": meta.get("ocr_text", ""),
                    "transcript_text": meta.get("transcript_text", ""),
                    "score": float(score),
                    "search_mode": "model",
                })

        # --- Text-similarity fallback search ---
        text_results = self._text_similarity_search(query, top_k)

        # If the model scores are very low (< 0.15 mean), prefer text search.
        # Otherwise, combine: weight model scores higher but merge text matches in.
        if not faiss_results:
            return self._finalize_results(text_results, video_url_template)

        avg_faiss_score = np.mean([r["score"] for r in faiss_results]) if faiss_results else 0
        if avg_faiss_score < 0.15:
            # Model is not confident — use text similarity as primary
            return self._finalize_results(text_results, video_url_template)

        # Merge: model results are primary, boosted by text similarity
        faiss_lookup = {}
        for r in faiss_results:
            key = (r["video_id"], r["start_time"], r["end_time"])
            faiss_lookup[key] = r

        text_lookup = {}
        for r in text_results:
            key = (r["video_id"], r["start_time"], r["end_time"])
            text_lookup[key] = r["score"]

        # Boost FAISS scores with text similarity
        for key, r in faiss_lookup.items():
            text_boost = text_lookup.get(key, 0.0)
            r["score"] = 0.7 * r["score"] + 0.3 * text_boost
            r["search_mode"] = "hybrid"

        merged = sorted(faiss_lookup.values(), key=lambda x: -x["score"])
        return self._finalize_results(merged[:top_k], video_url_template)

    def _finalize_results(self, results: List[Dict], video_url_template: str) -> List[Dict]:
        for r in results:
            r["jump_link"] = video_url_template.format(
                video_id=r["video_id"], start=int(r["start_time"])
            )
        return results
