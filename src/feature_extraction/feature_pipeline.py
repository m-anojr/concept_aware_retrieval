"""
End-to-end feature pipeline for a single lecture video.

For a raw lecture video this module:
  1. Samples frames at `time_step_sec` intervals.
  2. Extracts the audio track and runs Whisper ASR once for the whole video.
  3. Computes, per time-step:
       - OCR text-change rate            (ocr_extractor)
       - ASR transcript topic-drift      (asr_extractor)
       - Visual-change rate              (visual_extractor)
       - CLIP image embedding            (clip_extractor)
  4. Concatenates the three scalar signals + CLIP embedding into a single
     per-time-step feature vector and writes everything to a compressed
     .npz file (one per video) for downstream Stage-1/Stage-2 use.

Frame/audio extraction uses OpenCV + ffmpeg. Both are optional - if missing,
a clear error is raised explaining what to install. Real lecture videos are
processed from `data/raw_videos/` and written to `data/features/`.
"""

import os
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np

from config import FEATURES, MODELS, PATHS
from src.feature_extraction.ocr_extractor import extract_ocr_signal, get_ocr_backend
from src.feature_extraction.asr_extractor import (
    SentenceEncoder,
    extract_topic_drift_signal,
    get_asr_backend,
)
from src.feature_extraction.visual_extractor import extract_visual_change_signal
from src.feature_extraction.clip_extractor import CLIPEncoder
from src.utils.io_utils import load_features, save_features, video_id_from_path, feature_path_for


def project_clip_embeddings(clip_embeds: np.ndarray, out_dim: int, seed: int = 42) -> np.ndarray:
    """Deterministically reduce raw CLIP visual features to a lower dimension."""
    clip_embeds = clip_embeds.astype(np.float32)
    if clip_embeds.ndim != 2:
        raise ValueError(f"Expected clip_embeds shape (T, D), got {clip_embeds.shape}")
    if clip_embeds.shape[1] == out_dim:
        return clip_embeds

    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((clip_embeds.shape[1], out_dim), dtype=np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
    reduced = clip_embeds @ proj
    return reduced.astype(np.float32)


def load_feature_stream(path: str) -> dict:
    """Load saved per-time-step feature data and ensure the CLIP embedding dimension matches config."""
    feats = load_features(path)
    clip_embeds = feats.get("clip_embeds")
    if clip_embeds is None:
        raise RuntimeError(f"Feature file {path} is missing 'clip_embeds'.")

    if feats["features"].shape[1] == 3 + FEATURES.clip_dim:
        return feats

    # Rebuild the fused feature stream from the raw scalar signals and the
    # saved CLIP embeddings so legacy 512-d CLIP features become 32-d.
    ocr_signal = feats["ocr_signal"].astype(np.float32)
    topic_drift = feats["topic_drift_signal"].astype(np.float32)
    visual_signal = feats["visual_signal"].astype(np.float32)
    clip_embeds = project_clip_embeddings(clip_embeds, FEATURES.clip_dim)
    scalars = np.stack([ocr_signal, topic_drift, visual_signal], axis=1)
    feats["features"] = np.concatenate([scalars, clip_embeds], axis=1).astype(np.float32)
    feats["clip_embeds"] = clip_embeds
    return feats


def sample_frames(video_path: str, time_step_sec: float) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """Sample frames from `video_path` every `time_step_sec` seconds.

    Returns (frames, timestamps, duration_sec).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps > 0 else 0.0

    step_frames = max(1, int(round(time_step_sec * fps)))
    frames, timestamps = [], []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(idx / fps)
        idx += 1
    cap.release()

    return frames, np.array(timestamps, dtype=np.float32), duration


def extract_audio(video_path: str) -> str:
    """Extract a mono 16kHz WAV track to a temp file using ffmpeg."""
    out_path = os.path.join(tempfile.gettempdir(), video_id_from_path(video_path) + ".wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _ocr_with_skip(frames, backend, ocr_every_n: int = 4, bbox=None):
    """Run OCR on every `ocr_every_n`-th frame and interpolate for the rest.

    This cuts OCR processing time by (1 - 1/ocr_every_n) — typically 75%.
    For lecture videos the board/slide changes slowly, so this is safe.
    """
    from src.feature_extraction.ocr_extractor import extract_ocr_signal, board_region, text_change_rate

    T = len(frames)
    if T == 0:
        return np.zeros(0, dtype=np.float32), []

    # Run OCR only on sampled frames
    ocr_indices = list(range(0, T, ocr_every_n))
    if ocr_indices[-1] != T - 1:
        ocr_indices.append(T - 1)  # always include last frame

    sampled_texts = []
    for idx in ocr_indices:
        sampled_texts.append(backend.read_text(board_region(frames[idx], bbox)))
        if len(sampled_texts) % 20 == 0:
            print(f"    OCR: {len(sampled_texts)}/{len(ocr_indices)} frames processed")

    # Interpolate texts for skipped frames (carry forward from last OCR'd frame)
    all_texts = [""] * T
    sample_ptr = 0
    for t in range(T):
        if sample_ptr + 1 < len(ocr_indices) and t >= ocr_indices[sample_ptr + 1]:
            sample_ptr += 1
        all_texts[t] = sampled_texts[sample_ptr]

    # Compute text-change signal
    signal = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        signal[t] = text_change_rate(all_texts[t - 1], all_texts[t])

    return signal, all_texts


def build_feature_stream(video_path: str, save: bool = True, device: str = None,
                         ocr_every_n: int = 4) -> dict:
    """Run the full feature-extraction pipeline for one lecture video.

    Parameters
    ----------
    ocr_every_n : int
        Only run OCR on every N-th sampled frame (default 4 = 75% speedup).
        The text-change signal and OCR texts are interpolated for skipped frames.

    Returns a dict with keys:
      video_id, timestamps, duration,
      ocr_signal, topic_drift_signal, visual_signal, clip_embeds,
      features (T x D fused matrix),
      ocr_texts, transcript_texts (per time-step strings, used by Stage 2)
    """
    video_id = video_id_from_path(video_path)
    print(f"  [{video_id}] Sampling frames...")
    frames, timestamps, duration = sample_frames(video_path, FEATURES.time_step_sec)
    print(f"  [{video_id}] {len(frames)} frames sampled ({duration:.0f}s video)")

    # --- OCR text-change (with frame skipping for speed) ---
    print(f"  [{video_id}] Running OCR (every {ocr_every_n}-th frame)...")
    ocr_gpu = None if device is None else (device == "cuda")
    ocr_backend = get_ocr_backend(MODELS.ocr_engine, gpu=ocr_gpu)
    ocr_signal, ocr_texts = _ocr_with_skip(frames, ocr_backend, ocr_every_n=ocr_every_n)
    print(f"  [{video_id}] OCR done.")

    # --- ASR + topic drift ---
    print(f"  [{video_id}] Running ASR (Whisper)...")
    try:
        audio_path = extract_audio(video_path)
        asr = get_asr_backend(MODELS.whisper_model, device=device)
        transcript = asr.transcribe(audio_path)
        print(f"  [{video_id}] ASR done — {len(transcript)} segments.")
    except Exception as e:
        print(f"  [{video_id}] ASR failed ({e}), using empty transcript.")
        transcript = []
    sbert = SentenceEncoder(MODELS.sentence_bert, device=device)
    topic_drift_signal, transcript_texts = extract_topic_drift_signal(transcript, timestamps, sbert)

    # --- Visual change ---
    print(f"  [{video_id}] Computing visual-change signal...")
    visual_signal = extract_visual_change_signal(frames)

    # --- CLIP image embeddings (batched for memory efficiency) ---
    print(f"  [{video_id}] Encoding CLIP embeddings...")
    clip = CLIPEncoder(MODELS.clip_model, device=device)
    clip_embeds = clip.encode_images(frames)  # now internally batched
    clip_embeds = project_clip_embeddings(clip_embeds, FEATURES.clip_dim)
    print(f"  [{video_id}] CLIP done.")

    # --- Fuse scalar signals + CLIP embedding into one feature vector ---
    scalars = np.stack([ocr_signal, topic_drift_signal, visual_signal], axis=1)  # (T, 3)
    features = np.concatenate([scalars, clip_embeds], axis=1)  # (T, 3 + clip_dim)

    result = dict(
        video_id=video_id,
        timestamps=timestamps,
        duration=np.array([duration], dtype=np.float32),
        ocr_signal=ocr_signal,
        topic_drift_signal=topic_drift_signal,
        visual_signal=visual_signal,
        clip_embeds=clip_embeds,
        features=features,
        ocr_texts=np.array(ocr_texts, dtype=object),
        transcript_texts=np.array(transcript_texts, dtype=object),
    )

    if save:
        out_path = feature_path_for(video_id, PATHS.features_dir)
        save_features(out_path, **result)
        print(f"  [{video_id}] Saved to {out_path}")

    return result
