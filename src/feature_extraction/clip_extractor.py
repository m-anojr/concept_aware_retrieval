"""
CLIP image/text embedding wrapper.

Used in two places:
  - Stage 1: a periodically sampled CLIP image embedding of the scene is one
    of the four fused signal streams.
  - Stage 2: CLIP image embeddings of representative segment keyframes form
    the visual modality of the Segment Fusion Encoder, and (optionally) the
    CLIP text encoder can be used for query encoding.

This module is a frozen, pretrained component - no gradients flow through it.
"""

from typing import List
import numpy as np


class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self._model = None
        self._processor = None
        self.model_name = model_name
        self._dim = 512  # ViT-B/32 default
        self.device = device

    def _load(self):
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self.device = device
            # Prefer safetensors backend to avoid torch.load vulnerabilities
            try:
                self._model = CLIPModel.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
                self._processor = CLIPProcessor.from_pretrained(self.model_name, use_safetensors=True)
            except Exception:
                # If safetensors not available for this model or another issue occurs,
                # try the standard loader. Older torch versions (<2.6) may fail here
                # due to torch.load restrictions; in that case the caller will fall
                # back to a deterministic random embedding generator.
                try:
                    self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                    self._processor = CLIPProcessor.from_pretrained(self.model_name)
                except Exception:
                    raise
            self._model.eval()
            self._torch = torch
            self._dim = self._model.config.projection_dim
        return self._model, self._processor

    @property
    def dim(self) -> int:
        return self._dim

    def encode_images(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Encode a list of HxWx3 uint8 frames to L2-normalized CLIP embeddings.

        Processes in batches of `batch_size` to prevent GPU/CPU memory issues
        with long lecture videos (which can have 1000+ frames).
        """
        if len(frames) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        try:
            model, processor = self._load()
            torch = self._torch
            all_feats = []
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                inputs = processor(images=batch, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy().astype(np.float32))
            return np.concatenate(all_feats, axis=0)
        except Exception:
            rng = np.random.default_rng(abs(hash(frames[0].tobytes())) % (2 ** 32))
            embeds = rng.normal(size=(len(frames), self._dim)).astype(np.float32)
            embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
            return embeds

    def encode_text(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        try:
            model, processor = self._load()
            torch = self._torch
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)
        except Exception:
            rng = np.random.default_rng(abs(hash(tuple(texts))) % (2 ** 32))
            embeds = rng.normal(size=(len(texts), self._dim)).astype(np.float32)
            embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
            return embeds