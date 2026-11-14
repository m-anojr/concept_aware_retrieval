# ConceptBound JSON evaluation suite

Put the completed annotation JSON files in `data/annotations/` using the lecture basename, for example `1_boundaries.json` and `1_queries.json`. Then double-click `Run Evaluation.bat` (or run the command below):

```powershell
.\.venv\Scripts\python.exe evaluation_suite\evaluate.py
```

The suite checks video, feature, segment, and annotation consistency; computes Table II's boundary F1 at 15/30 seconds, Pk, WindowDiff, and segment counts for fixed windows, signal peaks, shot-change fallback, and the saved Stage 1 model when available; and computes Table III R@1, R@5, MRR, and mIoU for the existing cross-modal retriever and sliding-window retriever. It writes `output/evaluation_suite/report.json` plus CSV files for the tables and per-lecture results.

The current project does not implement BM25, BGE, RRF/reranking, or ground-truth-segmentation + RRF, so these rows are explicitly listed as unavailable. The existing Stage 2 checkpoint may have trained on these queries; if so, retrieval scores are descriptive and must not be presented as held-out generalization results. Human review of whether timestamps and queries are truly correct is still required.

Annotation formats:

`<id>_boundaries.json`:

```json
{
  "teaching_start_sec": 90.0,
  "teaching_end_sec": 3480.0,
  "boundary_times_sec": [512.0, 947.0, 1385.0]
}
```

`<id>_queries.json`:

```json
{
  "pairs": [
    {"query": "How does the instructor explain gradient descent?", "start_time": 612.0, "end_time": 730.0}
  ]
}
```

Example times/questions are placeholders. Boundary F1 matching is one-to-one within the named tolerance. Pk and WindowDiff use a window of half the mean gold segment length, rounded to steps. Table II aggregation is macro average by lecture.
