# Concept-Aware Retrieval of Lecture Video Moments
### using a Pedagogical Boundary Detector and Cross-Modal Segment Fusion

B.Tech Final Year Project — Department of Computer Science and Engineering (AI & ML)
Domain: Multimodal Video Retrieval / Educational Technology

---

## 1. What this project does

Given a long, single-camera lecture video (or a whole course playlist of them), this
system:

1. **Segments** each lecture into *concept-coherent* moments (e.g. "gradient
   descent derivation", "backpropagation worked example") using a trainable
   **Pedagogical Boundary Detector** (Stage 1), instead of fixed 60-second
   windows or generic shot-cut detection.
2. **Indexes** every segment with a trainable **Cross-Modal Segment Fusion
   Encoder** (Stage 2) that fuses on-screen board/slide text (OCR), spoken
   transcript (ASR), and visual keyframe content (CLIP) into one embedding
   per segment, stored in a FAISS vector index.
3. **Serves free-text queries** — a student types *"explain backpropagation
   with an example"* and gets back a ranked list of (video, timestamp range)
   results with a "jump to moment" link, via a FastAPI backend + simple web
   frontend.

The two trainable components are:

| Component | File | Objective |
|---|---|---|
| Pedagogical Boundary Detector | `src/stage1_boundary/model.py` | Self-supervised contrastive boundary loss (margin loss + segment InfoNCE) |
| Segment Fusion Encoder + Query Projector | `src/stage2_retrieval/fusion_model.py` | Symmetric InfoNCE contrastive loss on (query, segment) pairs |

Everything else (OCR, ASR/Whisper, Sentence-BERT, CLIP, FAISS) is **frozen,
pretrained** and used purely for feature extraction.

---

## 2. Project structure

```
lecture_retrieval/
├── config.py                      # central configuration (paths, dims, hyperparameters)
├── requirements.txt
├── data/
│   ├── raw_videos/                 # <-- put your .mp4/.mkv lecture files here
│   ├── features/                   # per-video fused feature streams (.npz)
│   ├── segments/                   # Stage-1 segment boundaries (.json) + Stage-2 segment features (.npz)
│   ├── annotations/                # manual ground-truth: boundaries + (query, segment) pairs
│   └── index/                      # FAISS index + metadata
├── checkpoints/                    # trained model weights
├── src/
│   ├── feature_extraction/         # OCR, ASR+topic-drift, visual-change, CLIP
│   ├── stage1_boundary/            # Pedagogical Boundary Detector: model, losses, pseudo-labels, train, segment
│   ├── stage2_retrieval/           # Segment Fusion Encoder, Query Projector, InfoNCE training, FAISS index, retrieval
│   ├── baselines/                  # fixed-window, shot-detection, sliding-window retrieval
│   ├── evaluation/                 # boundary F1/IoU, Recall@k / mean IoU
│   └── utils/                      # I/O helpers
├── app/
│   ├── backend/main.py             # FastAPI search API
│   └── frontend/index.html         # search UI
└── scripts/                        # orchestration scripts (see below)
```

---

## 3. Setup

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note:** `transformers` (CLIP) and `sentence-transformers` download pretrained
> weights from the internet on first use. Real lecture-video experiments need
> the actual pretrained encoders.

All scripts assume the project root is on `PYTHONPATH`. Either run them with
`python -m ...` from the project root, or:

```bash
# macOS / Linux
export PYTHONPATH=$(pwd)
# Windows PowerShell
$env:PYTHONPATH = Resolve-Path .
```

---

## 4. End-to-end pipeline (real lecture videos)

### Step 0 — Place videos
Put your NPTEL / YouTube lecture `.mp4` files in `data/raw_videos/`.

### Step 1 — Feature extraction
```bash
python scripts/extract_features.py
```
For each video, samples frames every `FEATURES.time_step_sec` (default 3s)
and computes:
- OCR text-change rate (EasyOCR/Tesseract on the board/slide region)
- ASR transcript + topic-drift signal (Whisper + Sentence-BERT)
- Visual-change rate (optical flow)
- CLIP image embedding of the sampled frame

Output: `data/features/<video_id>.npz`

### Step 2 — Annotate ground truth (for training/eval)
For each video you want to use for training/evaluation, create:

`data/annotations/<video_id>_boundaries.json`
```json
{ "boundary_times_sec": [185.0, 612.0, 1450.0] }
```

`data/annotations/<video_id>_queries.json`
```json
{
  "pairs": [
    {"query": "explain backpropagation with an example", "start_time": 612.0, "end_time": 730.0},
    {"query": "where is the cost function for linear regression defined?", "start_time": 60.0, "end_time": 185.0}
  ]
}
```

### Step 3 — Train Stage 1 (Pedagogical Boundary Detector)
```bash
python -m src.stage1_boundary.train --epochs 30 --encoder-type transformer
```
Saves the best checkpoint to `checkpoints/stage1_boundary_detector.pt`.

### Step 4 — Segment all videos
```bash
python scripts/segment_videos.py
```
Writes `data/segments/<video_id>.json` with `[{start_idx, end_idx,
start_time, end_time}, ...]`.

### Step 5 — Compute per-segment Stage-2 features
```bash
python scripts/prepare_segment_features.py
```
Aggregates OCR/transcript text and CLIP embeddings per segment into
`data/segments/<video_id>_segment_features.npz`.

### (No manual annotations yet?) Generate pseudo-queries
Stage 2 training requires `data/annotations/<video_id>_queries.json`. If you
haven't written manual (query, segment) pairs yet, generate self-supervised
pseudo-queries from each segment's own transcript/OCR text:
```bash
python scripts/generate_pseudo_queries.py
```
This lets the full pipeline run end-to-end with zero manual work. It is a
weaker training/eval signal than real student queries - replace these files
with hand-written ones (Step 2 format above) whenever possible for a
meaningful Recall@k evaluation. The script will not overwrite an existing
`*_queries.json` (e.g. one you wrote by hand) unless you pass `--overwrite`.

### Step 6 — Train Stage 2 (Segment Fusion Encoder + Query Projector)
```bash
python -m src.stage2_retrieval.train --epochs 20 --fusion-mode cross_attention
```
Saves the best checkpoint to `checkpoints/stage2_retrieval_model.pt`.
Use `--fusion-mode concat` to run the concatenation-vs-cross-attention
ablation described in the proposal.

### Step 7 — Build the FAISS index
```bash
python -m src.stage2_retrieval.index_builder
```

### Step 8 — Evaluate against baselines
```bash
python scripts/run_evaluation.py
```
Prints boundary F1 / mean IoU (Stage 1 vs. fixed-window vs. shot-detection)
and Recall@1/5/10 / mean IoU@1 (Stage 2 vs. sliding-window retrieval).

### Step 9 — Run the demo web app
```bash
uvicorn app.backend.main:app --reload --port 8000
```
Open `http://localhost:8000` and search.

---

## 5. Notes on running your own lecture videos

Use the real video pipeline with files placed in `data/raw_videos/`.

1. Place your `.mp4/.mkv/.avi/.mov/.webm` files in `data/raw_videos/`
2. Extract features:
```bash
python scripts/extract_features.py
```
3. Segment videos with the trained Stage 1 model:
```bash
python scripts/segment_videos.py
```
4. Compute per-segment Stage-2 features:
```bash
python scripts/prepare_segment_features.py
```
5. Build the FAISS index:
```bash
python -m src.stage2_retrieval.index_builder
```
6. Run the demo app:
```bash
uvicorn app.backend.main:app --reload --port 8000
```

> If you do not have a trained model yet, train Stage 1 and Stage 2 first,
> or obtain the checkpoints `checkpoints/stage1_boundary_detector.pt` and
> `checkpoints/stage2_retrieval_model.pt`.

---

## 6. Mapping to the proposal

| Proposal section | Implementation |
|---|---|
| §4.1 Stage 1 — feature streams (OCR-Δ, transcript topic-drift, visual-Δ, CLIP) | `src/feature_extraction/*` |
| §4.1 Pedagogical Boundary Detector (1D-CNN / Transformer, contrastive boundary loss) | `src/stage1_boundary/model.py`, `losses.py` |
| §4.1 Pseudo-labels from signal peaks + manual annotation refinement | `src/stage1_boundary/pseudo_labels.py` |
| §4.2 Segment Fusion Encoder (cross-attention) | `src/stage2_retrieval/fusion_model.py` |
| §4.2 InfoNCE retrieval training | `src/stage2_retrieval/train.py` |
| §4.2 FAISS index + "jump to moment" | `src/stage2_retrieval/index_builder.py`, `retrieval.py`, `app/` |
| §5.2 Baselines (fixed-window, shot-detection, sliding-window retrieval) | `src/baselines/*` |
| §5.3 Ablations (signal removal, segmentation method, fusion strategy) | drop columns from `features` before Stage-1 training; swap `--fusion-mode` for Stage 2 |
| §5 Metrics (boundary F1/IoU, Recall@k, mean IoU) | `src/evaluation/*` |

### Running the ablations

- **Signal-removal ablation (Stage 1):** zero out one of the three scalar
  columns (`features[:, 0]`=OCR, `features[:, 1]`=topic-drift,
  `features[:, 2]`=visual) before training, and compare boundary F1/IoU.
- **Segmentation-method ablation (Stage 2):** run `prepare_segment_features.py`
  + Stage-2 training/indexing once using Stage-1 segments and once using
  `src/baselines/fixed_window.fixed_window_segments` as the segment
  boundaries, holding the retrieval architecture fixed.
- **Fusion-strategy ablation (Stage 2):** train with `--fusion-mode
  cross_attention` vs `--fusion-mode concat` and compare Recall@k.

---

## 7. Notes on scaling to real NPTEL/YouTube data

- `FEATURES.time_step_sec` (default 3s) and `FEATURES.clip_dim` /
  `FEATURES.text_embed_dim` in `config.py` should match your chosen
  EasyOCR/Whisper/CLIP/Sentence-BERT models. The training scripts also infer
  dimensions directly from the saved `.npz`/segment-feature files, so they
  remain correct even if you change these.
- For a 1–2 hour lecture at a 3-second step, expect `T ≈ 1200–2400`
  time-steps per video — the Transformer encoder in Stage 1 handles this
  comfortably on CPU for inference; for training on many such videos, a GPU
  is recommended.
- OCR is the most expensive step at scale; consider running it only every
  N-th sampled frame and interpolating, or cropping tightly to the
  board/slide region via `src/feature_extraction/ocr_extractor.board_region`.
