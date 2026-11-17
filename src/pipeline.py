"""End-to-end inference pipeline: encoders -> fusion -> classification -> explanation."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

import config
from src import consistency as consistency_mod
from src import explain, image_stats
from src.fusion_model import IMAGE_DIM, TEXT_DIM, FusionNet, ImageOnlyNet, TextOnlyNet
from src.image_encoder import ImageEncoder
from src.text_encoder import TextEncoder

_BUNDLE = None
_BUNDLE_LOCK = threading.Lock()


class ModelsNotTrained(RuntimeError):
    pass


class Bundle:
    def __init__(self) -> None:
        self.fusion = FusionNet()
        self.text_only = TextOnlyNet()
        self.image_only = ImageOnlyNet()
        self.threshold = config.DEFAULT_DECISION_THRESHOLD
        self.metrics = {}
        self.temperatures = {}
        self.metrics_path = config.MODELS_DIR / "metrics.json"

    def load(self) -> None:
        for kind, net in (("fusion", self.fusion), ("text_only", self.text_only),
                          ("image_only", self.image_only)):
            path = config.MODELS_DIR / f"{kind}.pt"
            if not path.exists():
                raise ModelsNotTrained(
                    f"{path.name} is missing. Run `python -m src.train` and "
                    "`python -m src.evaluate` first."
                )
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()
            for param in net.parameters():
                param.requires_grad_(False)
        if self.metrics_path.exists():
            self.metrics = json.loads(self.metrics_path.read_text())
            self.threshold = float(self.metrics.get("chosen_threshold",
                                                    config.DEFAULT_DECISION_THRESHOLD))
            self.temperatures = self.metrics.get("temperature", {})

    def temp(self, head: str) -> float:
        return float(self.temperatures.get(head, 1.0)) or 1.0


def load_bundle() -> Bundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _BUNDLE_LOCK:
        if _BUNDLE is None:
            bundle = Bundle()
            bundle.load()
            _BUNDLE = bundle
    return _BUNDLE


def warm_up() -> None:
    """Load every model once so the first request is fast."""
    load_bundle()
    TextEncoder.instance().load()
    ImageEncoder.instance().load()


def _fusion_probs(bundle: Bundle, text_vec: np.ndarray, image_vec: np.ndarray) -> dict:
    with torch.inference_mode():
        logits, cons_logits, cos, gates = bundle.fusion(
            torch.from_numpy(text_vec), torch.from_numpy(image_vec)
        )
        return {
            "prob": torch.softmax(logits / bundle.temp("fusion"), dim=-1).numpy()[0],
            "consistency": torch.softmax(
                cons_logits / bundle.temp("consistency"), dim=-1).numpy()[0],
            "cosine": float(cos.item()),
            "gates": gates.numpy()[0],
        }


def analyze(
    text: str,
    image: Optional[Image.Image] = None,
    source: Optional[str] = None,
    platform: Optional[str] = None,
    image_meta: Optional[dict] = None,
    explain_tokens: bool = True,
) -> dict:
    started = time.time()
    bundle = load_bundle()
    text_encoder = TextEncoder.instance()

    text_vec = text_encoder.encode([text]).astype(np.float32)
    has_image = image is not None

    if has_image:
        image_encoder = ImageEncoder.instance()
        image_vec = image_encoder.encode([image]).astype(np.float32)
        modality = "multimodal"
    else:
        image_vec = np.zeros((1, IMAGE_DIM), dtype=np.float32)
        modality = "text"

    result = _fusion_probs(bundle, text_vec, image_vec)
    p_misleading = float(result["prob"][1])

    # --- single-modality baseline scores ---------------------------------
    with torch.inference_mode():
        p_text_only = float(torch.softmax(
            bundle.text_only(torch.from_numpy(text_vec)) / bundle.temp("text_only"),
            dim=-1).numpy()[0][1])
        if has_image:
            p_image_only = float(torch.softmax(
                bundle.image_only(torch.from_numpy(image_vec)) / bundle.temp("image_only"),
                dim=-1).numpy()[0][1])
        else:
            p_image_only = None

    # --- modality contribution by ablation --------------------------------
    if has_image:
        zero_img = np.zeros_like(image_vec)
        zero_txt = np.zeros_like(text_vec)
        p_no_image = float(_fusion_probs(bundle, text_vec, zero_img)["prob"][1])
        p_no_text = float(_fusion_probs(bundle, zero_txt, image_vec)["prob"][1])
        delta_text = abs(p_misleading - p_no_text)
        delta_image = abs(p_misleading - p_no_image)
        total = delta_text + delta_image + 1e-9
        contributions = {
            "text": round(100 * delta_text / total, 1),
            "image": round(100 * delta_image / total, 1),
            "text_delta": round(p_misleading - p_no_text, 4),
            "image_delta": round(p_misleading - p_no_image, 4),
            "p_without_text": round(p_no_text, 4),
            "p_without_image": round(p_no_image, 4),
            "method": "leave-one-modality-out ablation of the fusion network",
        }
    else:
        contributions = {
            "text": 100.0, "image": 0.0,
            "text_delta": round(p_misleading - 0.5, 4), "image_delta": None,
            "p_without_text": None, "p_without_image": None,
            "method": "text-only inference (no image supplied)",
        }

    # --- cross-modal consistency -----------------------------------------
    consistency = None
    if has_image:
        consistency = consistency_mod.evaluate(
            result["cosine"], float(result["consistency"][1]))

    # --- decision ---------------------------------------------------------
    is_misleading = p_misleading >= bundle.threshold
    label = "Misleading" if is_misleading else "Credible"
    confidence = p_misleading if is_misleading else 1.0 - p_misleading

    # --- text analysis ----------------------------------------------------
    tokens = text_encoder.tokenize(text)
    linguistic = explain.linguistic_features(text)
    attribution = {"words": [], "method": "disabled"}
    if explain_tokens:
        def score_fn(texts):
            vecs = text_encoder.encode(list(texts))
            with torch.inference_mode():
                logits, _, _, _ = bundle.fusion(
                    torch.from_numpy(vecs), torch.from_numpy(np.repeat(image_vec, len(texts), axis=0))
                )
            return torch.softmax(logits, dim=-1).numpy()[:, 1]

        attribution = explain.token_attributions(text, score_fn)

    text_analysis = {
        "status": "analysed",
        "encoder": text_encoder.describe(),
        "tokens": len(tokens),
        "truncated": len(tokens) >= config.TEXT_MAX_LENGTH - 2,
        "embedding": text_encoder.embedding_stats(text_vec[0]),
        "prob_misleading": round(p_text_only, 4),
        "score": round(p_text_only, 4),
        "features": linguistic,
        "attribution": attribution,
    }

    # --- image analysis ---------------------------------------------------
    image_analysis = None
    if has_image:
        stats = image_stats.analyse(image)
        if image_meta:
            stats.update({k: v for k, v in image_meta.items()
                          if k in ("filename", "sha256", "url", "mime")})
        image_analysis = {
            "status": "analysed",
            "encoder": ImageEncoder.instance().describe(),
            "embedding": {
                "dimensions": int(image_vec.shape[1]),
                "l2_norm": round(float(np.linalg.norm(image_vec[0])), 4),
                "mean": round(float(image_vec[0].mean()), 4),
                "std": round(float(image_vec[0].std()), 4),
                "active_features": int((image_vec[0] > 0).sum()),
            },
            "prob_misleading": round(p_image_only, 4),
            "score": round(p_image_only, 4),
            "statistics": stats,
        }

    explanation = explain.summarise(
        linguistic,
        (image_analysis or {}).get("statistics", {}).get("artefact_flags", []),
        consistency, p_misleading, modality,
    )

    return {
        "text": text,
        "source": (source or "").strip() or None,
        "platform": (platform or "").strip() or None,
        "modality": modality,
        "label": label,
        "confidence": round(confidence, 4),
        "misleading_prob": round(p_misleading, 4),
        "threshold": bundle.threshold,
        "contributions": contributions,
        "gate_weights": {
            "text": round(float(result["gates"][0]), 4),
            "image": round(float(result["gates"][1]), 4),
        },
        "text_analysis": text_analysis,
        "image_analysis": image_analysis,
        "consistency": consistency,
        "explanation": explanation,
        "latency_ms": int((time.time() - started) * 1000),
        "encoders": {
            "text": text_encoder.describe(),
            "image": ImageEncoder.instance().describe() if has_image else None,
        },
        "disclaimer": config.DISCLAIMER,
    }


def hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
