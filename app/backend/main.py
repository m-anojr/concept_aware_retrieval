"""
FastAPI backend for the Concept-Aware Lecture Retrieval demo.

Endpoints
---------
GET  /api/health       — system health + diagnostics
GET  /api/search?q=<query>&top_k=<k>  — search the lecture corpus
GET  /api/stats        — index statistics

Run with:
    uvicorn app.backend.main:app --reload --port 8000

The frontend (app/frontend/index.html) is also served as a static page at
"/" for convenience.
"""

import os
import traceback
from email.utils import formatdate
from typing import Tuple

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import PATHS
from src.stage2_retrieval.retrieval import LectureRetriever

app = FastAPI(title="Concept-Aware Lecture Video Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever = None
_retriever_error = None


def get_retriever() -> LectureRetriever:
    global _retriever, _retriever_error
    if _retriever is None and _retriever_error is None:
        try:
            _retriever = LectureRetriever(index_dir=PATHS.index_dir)
        except Exception as e:
            _retriever_error = str(e)
            print(f"[API] Failed to load retriever: {e}")
            traceback.print_exc()
    return _retriever


@app.get("/api/health")
def health():
    index_exists = os.path.exists(os.path.join(PATHS.index_dir, "segments.index"))
    metadata_exists = os.path.exists(os.path.join(PATHS.index_dir, "segments_metadata.json"))
    stage1_exists = os.path.exists(os.path.join(PATHS.checkpoints_dir, "stage1_boundary_detector.pt"))
    stage2_exists = os.path.exists(os.path.join(PATHS.checkpoints_dir, "stage2_retrieval_model.pt"))

    retriever = get_retriever()
    num_segments = 0
    if retriever is not None:
        num_segments = retriever.index.ntotal

    return {
        "status": "ok" if index_exists else "index_missing",
        "index_ready": index_exists and metadata_exists,
        "num_segments": num_segments,
        "stage1_model_ready": stage1_exists,
        "stage2_model_ready": stage2_exists,
        "retriever_loaded": retriever is not None,
        "retriever_error": _retriever_error,
    }


@app.get("/api/stats")
def stats():
    retriever = get_retriever()
    if retriever is None:
        return {"error": _retriever_error or "Retriever not loaded. Build the index first."}

    # Count unique videos
    video_ids = set()
    for meta in retriever.metadata:
        video_ids.add(meta["video_id"])

    return {
        "num_segments": retriever.index.ntotal,
        "num_videos": len(video_ids),
        "video_ids": sorted(video_ids),
        "has_model": retriever._model is not None,
    }


@app.get("/api/search")
def search(q: str = Query(..., description="Free-text concept query"),
           top_k: int = Query(5, ge=1, le=20)):
    retriever = get_retriever()
    if retriever is None:
        return {
            "query": q,
            "results": [],
            "error": _retriever_error or "Index not built. Run the pipeline first.",
        }
    try:
        # Pass a URL template that points to the mounted raw videos directory
        results = retriever.search(q, top_k=top_k, video_url_template="/videos/{video_id}.mp4#t={start}")
        response = {"query": q, "results": results}
        if not results:
            response["message"] = "Not found in video archive."
        return response
    except Exception as e:
        traceback.print_exc()
        return {"query": q, "results": [], "error": str(e)}


def _parse_byte_range(range_header: str, file_size: int) -> Tuple[int, int]:
    unit, separator, value = range_header.partition("=")
    if unit.strip().lower() != "bytes" or not separator or "," in value:
        raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    try:
        start_text, end_text = value.strip().split("-", 1)
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_text)
            end = min(int(end_text), file_size - 1) if end_text else file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    if file_size == 0 or start < 0 or start >= file_size or end < start:
        raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    return start, end


def _stream_file_range(file_path: str, start: int, end: int):
    with open(file_path, "rb") as video_file:
        video_file.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = video_file.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.api_route("/videos/{video_name}", methods=["GET", "HEAD"])
def video(video_name: str, request: Request):
    if os.path.basename(video_name) != video_name or video_name in {".", ".."}:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = os.path.realpath(os.path.join(PATHS.raw_videos_dir, video_name))
    video_root = os.path.realpath(PATHS.raw_videos_dir)
    if os.path.commonpath((video_root, video_path)) != video_root or not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    stat = os.stat(video_path)
    file_size = stat.st_size
    last_modified = formatdate(stat.st_mtime, usegmt=True)
    etag = f'"{stat.st_mtime_ns:x}-{file_size:x}"'
    range_header = request.headers.get("range")
    if_range = request.headers.get("if-range")
    if if_range and if_range not in {etag, last_modified}:
        range_header = None

    if range_header:
        start, end = _parse_byte_range(range_header, file_size)
        status_code = 206
    else:
        start, end = 0, file_size - 1
        status_code = 200

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(max(0, end - start + 1)),
        "Last-Modified": last_modified,
        "ETag": etag,
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

    if request.method == "HEAD":
        return Response(status_code=status_code, headers=headers)

    return StreamingResponse(
        _stream_file_range(video_path, start, end),
        status_code=status_code,
        media_type="video/mp4",
        headers=headers,
    )


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Keep the static mount as a fallback for nested video-directory paths.
if os.path.isdir(PATHS.raw_videos_dir):
    app.mount("/videos", StaticFiles(directory=PATHS.raw_videos_dir), name="videos")
