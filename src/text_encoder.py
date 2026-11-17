"""Transformer-based text encoder (DistilBERT / BERT via Hugging Face).

A frozen pretrained encoder converts a post into a 768-dimensional contextual
embedding (masked mean pooling of the last hidden state). The embedding is what
the fusion network and the text-only baseline consume - there is no keyword
matching anywhere in the classification path.
"""
from __future__ import annotations

import threading
from typing import List, Sequence

import numpy as np
import torch

import config


class ModelUnavailable(RuntimeError):
    """Raised when the pretrained weights cannot be loaded (e.g. no network)."""


class TextEncoder:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = config.TEXT_MODEL_NAME,
                 max_length: int = config.TEXT_MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self.hidden_size = 768
        self.n_layers = None
        self.n_heads = None
        self.quantized = False

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def instance(cls) -> "TextEncoder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise ModelUnavailable(
                    "Hugging Face Transformers is not installed. Run "
                    "`pip install -r requirements.txt`."
                ) from exc
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
            except Exception as exc:  # network / cache errors
                raise ModelUnavailable(
                    f"Could not load transformer '{self.model_name}': {exc}. "
                    "The first run needs internet access to download the weights."
                ) from exc

            model.eval()
            if config.QUANTIZE_TEXT_MODEL:
                try:
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    self.quantized = True
                except Exception:  # pragma: no cover - quantisation is optional
                    self.quantized = False
            for param in model.parameters():
                param.requires_grad_(False)

            self._model = model
            self.hidden_size = int(model.config.dim)
            self.n_layers = int(model.config.n_layers)
            self.n_heads = int(model.config.n_heads)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def describe(self) -> dict:
        return {
            "model": self.model_name,
            "architecture": "DistilBertModel (transformer encoder, frozen)",
            "hidden_size": self.hidden_size,
            "layers": self.n_layers,
            "attention_heads": self.n_heads,
            "max_tokens": self.max_length,
            "pooling": "masked mean of last hidden state",
            "quantised_int8": self.quantized,
        }

    # -- encoding ----------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        self.load()
        enc = self._tokenizer(text, truncation=True, max_length=self.max_length,
                              return_tensors=None)
        return self._tokenizer.convert_ids_to_tokens(enc["input_ids"])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, 768) float32 matrix of contextual text embeddings."""
        self.load()
        if not texts:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        chunks = []
        batch_size = 16 if config.QUANTIZE_TEXT_MODEL else 24
        for start in range(0, len(texts), batch_size):
            batch = [t if t and t.strip() else "empty" for t in texts[start:start + batch_size]]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            with torch.inference_mode():
                out = self._model(**enc).last_hidden_state  # (b, t, h)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def embedding_stats(self, vector: np.ndarray) -> dict:
        v = np.asarray(vector, dtype=np.float32).ravel()
        return {
            "dimensions": int(v.size),
            "l2_norm": round(float(np.linalg.norm(v)), 4),
            "mean": round(float(v.mean()), 4),
            "std": round(float(v.std()), 4),
            "nonzero_dims": int((np.abs(v) > 1e-6).sum()),
        }
