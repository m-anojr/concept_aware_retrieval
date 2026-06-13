"""
Auto-generate (query, segment) training pairs WITHOUT manual annotation.

For every video that has segment-level features
(`data/segments/<video_id>_segment_features.npz`, produced by
`prepare_segment_features.py`), this script builds one "pseudo-query" per
segment by taking a short snippet of that segment's own ASR transcript text
(falling back to its OCR/slide text if the transcript is empty), and writes
it to `data/annotations/<video_id>_queries.json` in the same format expected
by `src.stage2_retrieval.train`.

This is a SELF-SUPERVISED substitute: Stage 2 learns to map a short piece of
a segment's own spoken/written content back to that segment's fused
embedding (a denoising / paraphrase-retrieval-style objective). It lets the
full pipeline run end-to-end without hand-written student queries, but real
student queries (see README §4 Step 2) will give a much more meaningful
retrieval model and evaluation - replace/augment these files with manual
ones whenever you can.

Usage
-----
    python scripts/generate_pseudo_queries.py
    python scripts/generate_pseudo_queries.py --snippet-words 10
"""

import argparse
import glob
import os

from config import PATHS, ensure_dirs
from src.stage2_retrieval.segment_features import load_segment_features
from src.utils.io_utils import save_json


def make_pseudo_query(text: str, snippet_words: int) -> str:
    words = text.split()
    return " ".join(words[:snippet_words]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", default=PATHS.segments_dir)
    parser.add_argument("--annotations-dir", default=PATHS.annotations_dir)
    parser.add_argument("--snippet-words", type=int, default=12,
                         help="Number of leading words from the segment's "
                              "transcript/OCR text used as the pseudo-query.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Overwrite existing *_queries.json files. "
                              "By default, videos that already have a "
                              "queries file (e.g. manually written) are left untouched.")
    args = parser.parse_args()

    ensure_dirs()

    paths = sorted(glob.glob(os.path.join(args.segments_dir, "*_segment_features.npz")))
    video_ids = [os.path.basename(p).replace("_segment_features.npz", "") for p in paths]

    if not video_ids:
        print(f"No '*_segment_features.npz' files found in {args.segments_dir}. "
              f"Run segment_videos.py and prepare_segment_features.py first.")
        return

    total_pairs = 0
    for video_id in video_ids:
        out_path = os.path.join(args.annotations_dir, f"{video_id}_queries.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{video_id}: queries file already exists, skipping (use --overwrite to replace).")
            continue

        feats = load_segment_features(video_id, args.segments_dir)
        starts, ends = feats["start_times"], feats["end_times"]
        transcript_text = feats["transcript_text"]
        ocr_text = feats["ocr_text"]

        pairs = []
        for i in range(len(starts)):
            text = str(transcript_text[i]).strip() or str(ocr_text[i]).strip()
            if not text:
                continue
            query = make_pseudo_query(text, args.snippet_words)
            if not query:
                continue
            pairs.append({
                "query": query,
                "start_time": float(starts[i]),
                "end_time": float(ends[i]),
            })

        save_json({"pairs": pairs}, out_path)
        total_pairs += len(pairs)
        print(f"{video_id}: {len(pairs)} pseudo (query, segment) pairs -> {out_path}")

    print(f"\nTotal pseudo-query pairs written: {total_pairs}")
    if total_pairs == 0:
        print("WARNING: 0 pairs generated. This usually means both OCR and ASR "
              "transcript text are empty for every segment - check that ffmpeg/"
              "Whisper/EasyOCR ran successfully during feature extraction.")


if __name__ == "__main__":
    main()
