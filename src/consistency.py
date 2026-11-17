"""Cross-modal consistency checking in the shared projection space.

Two independent signals are combined:

1. Cosine similarity between the L2-normalised text and image projections of the
   trained fusion network (learned shared 256-d space).
2. The dedicated consistency head of the fusion network, trained with the
   pair-level consistency labels from the benchmark.

The calibration statistics (mean cosine of consistent vs inconsistent pairs,
measured on the training split) are used to render an interpretable score
instead of a raw cosine value.
"""
from __future__ import annotations

import json

import config


def load_calibration() -> dict:
    path = config.MODELS_DIR / "split.json"
    if not path.exists():
        return {"consistent_mean": 0.35, "consistent_std": 0.1,
                "inconsistent_mean": 0.05, "inconsistent_std": 0.1,
                "midpoint": 0.2}
    return json.loads(path.read_text()).get("consistency_cosine", {
        "consistent_mean": 0.35, "consistent_std": 0.1,
        "inconsistent_mean": 0.05, "inconsistent_std": 0.1, "midpoint": 0.2,
    })


def evaluate(cosine: float, prob_consistent: float, calibration: dict | None = None) -> dict:
    cal = calibration or load_calibration()
    consistent = prob_consistent >= 0.5
    label = "Consistent" if consistent else "Inconsistent"

    mid = float(cal.get("midpoint", 0.2))
    spread = max(0.05, float(cal.get("consistent_std", 0.1)))
    relative = (cosine - mid) / spread  # positive => closer to consistent centroid

    if consistent and cosine >= mid:
        verdict = ("The image content is semantically aligned with the claim; this pair "
                   "shows no cross-modal contradiction.")
    elif not consistent and cosine < mid:
        verdict = ("The image does not support the claim in the shared embedding space. "
                   "Inconsistent pairs are a strong misinformation signal.")
    else:
        verdict = ("The similarity and the consistency head disagree for this pair, so the "
                   "cross-modal signal is weak and the decision relies more on the text.")

    return {
        "score": round(100 * prob_consistent, 1),
        "probability_consistent": round(prob_consistent, 4),
        "cosine_similarity": round(cosine, 4),
        "relative_to_calibration": round(float(relative), 3),
        "label": label,
        "verdict": verdict,
        "calibration": {
            "consistent_mean_cosine": cal.get("consistent_mean"),
            "inconsistent_mean_cosine": cal.get("inconsistent_mean"),
        },
    }
