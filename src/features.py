"""Feature cache: runs both frozen encoders once and stores the embeddings."""
from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image

import config
from src import benchmark
from src.image_encoder import ImageEncoder
from src.text_encoder import TextEncoder

FEATURE_PATH = config.ARTIFACTS_DIR / "features.npz"
META_PATH = config.ARTIFACTS_DIR / "benchmark.json"
SCENE_DIR = config.ARTIFACTS_DIR / "scenes"


def build(force: bool = False, seed: int = 42) -> dict:
    """Create the cached feature matrix (idempotent)."""
    if FEATURE_PATH.exists() and META_PATH.exists() and not force:
        return load()

    started = time.time()
    samples = benchmark.build_benchmark(seed=seed)
    benchmark.materialise(samples, SCENE_DIR)
    print(f"[features] rendered {len(samples)} sample images")

    text_encoder = TextEncoder.instance()
    image_encoder = ImageEncoder.instance()

    texts = [s["text"] for s in samples]
    print("[features] encoding text with", text_encoder.model_name)
    text_feats = text_encoder.encode(texts)

    images = [Image.open(s["image_path"]).convert("RGB") for s in samples]
    print("[features] encoding images with ResNet18")
    image_feats = image_encoder.encode(images)

    labels = np.array([s["label"] for s in samples], dtype=np.int64)
    consistency = np.array([s["consistency"] for s in samples], dtype=np.int64)
    np.savez_compressed(
        FEATURE_PATH,
        text=text_feats.astype(np.float32),
        image=image_feats.astype(np.float32),
        label=labels,
        consistency=consistency,
    )
    meta = {
        "samples": samples,
        "dataset": benchmark.describe(),
        "seed": seed,
        "built_in_seconds": round(time.time() - started, 1),
        "text_model": text_encoder.describe(),
        "image_model": image_encoder.describe(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[features] cache written in {meta['built_in_seconds']}s")
    return load()


def load() -> dict:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Feature cache missing. Run `python -m src.train` to build it."
        )
    data = np.load(FEATURE_PATH)
    meta = json.loads(META_PATH.read_text())
    return {
        "text": data["text"],
        "image": data["image"],
        "label": data["label"],
        "consistency": data["consistency"],
        "meta": meta,
    }
