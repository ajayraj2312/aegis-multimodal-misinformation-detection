"""Evaluate all three models on the held-out test split.

Usage:  python -m src.evaluate
Writes models/metrics.json, which the /performance page renders verbatim.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import config
from src import features
from src.fusion_model import FusionNet, ImageOnlyNet, TextOnlyNet

MODEL_NAMES = {
    "text_only": "Text-only Transformer (DistilBERT)",
    "image_only": "Image-only CNN (ResNet18)",
    "fusion": "Multimodal Fusion Network",
}


def _load_models():
    fusion = FusionNet()
    fusion.load_state_dict(torch.load(config.MODELS_DIR / "fusion.pt"))
    fusion.eval()
    text_only = TextOnlyNet()
    text_only.load_state_dict(torch.load(config.MODELS_DIR / "text_only.pt"))
    text_only.eval()
    image_only = ImageOnlyNet()
    image_only.load_state_dict(torch.load(config.MODELS_DIR / "image_only.pt"))
    image_only.eval()
    return {"fusion": fusion, "text_only": text_only, "image_only": image_only}


def _predict(models: dict, data: dict, idxs):
    """Return raw logits per head (calibration is applied on top of these)."""
    t = torch.from_numpy(data["text"][idxs])
    i = torch.from_numpy(data["image"][idxs])
    out = {}
    with torch.inference_mode():
        out["text_only"] = models["text_only"](t).numpy()
        out["image_only"] = models["image_only"](i).numpy()
        logits, cons_logits, cos, gates = models["fusion"](t, i)
        out["fusion"] = logits.numpy()
        out["consistency"] = cons_logits.numpy()
        out["fusion_cosine"] = cos.numpy()
        out["fusion_gates"] = gates.numpy()
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Post-hoc temperature scaling: minimise validation NLL over a temperature grid.

    Trained heads on a small corpus can produce saturated logits, which overstates
    certainty. Only softening temperatures (T >= 1) are allowed: sharpening a model
    post-hoc would inflate confidence rather than calibrate it. AUC is unaffected.
    """
    best_t, best_nll = 1.0, float("inf")
    for temperature in np.arange(1.0, 12.01, 0.25):
        probs = _softmax(logits / temperature)
        nll = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best_nll:
            best_t, best_nll = float(temperature), float(nll)
    return round(best_t, 2)


def _best_threshold(y_true, scores) -> float:
    """Threshold that maximises F1 on the *validation* split."""
    thresholds = np.unique(np.round(scores, 4))
    best_t, best_f1 = 0.5, -1.0
    for thr in thresholds:
        f1 = f1_score(y_true, (scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(thr)
    return round(best_t, 4)


def _metrics(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, probs)
    keep = np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, probs)), 4),
        "average_precision": round(float(average_precision_score(y_true, probs)), 4),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": {"fpr": [round(float(v), 4) for v in fpr[keep]],
                "tpr": [round(float(v), 4) for v in tpr[keep]]},
        "n": int(len(y_true)),
    }


def main() -> None:
    data = features.load()
    split = json.loads((config.MODELS_DIR / "split.json").read_text())
    models = _load_models()

    test_idx = split["test"]
    val_idx = split["val"]
    y_test = data["label"][test_idx]
    y_val = data["label"][val_idx]

    val_logits = _predict(models, data, val_idx)
    test_logits = _predict(models, data, test_idx)

    temperatures = {head: fit_temperature(val_logits[head], y_val)
                    for head in ("text_only", "image_only", "fusion", "consistency")}
    y_cons_val = data["consistency"][val_idx]
    temperatures["consistency"] = fit_temperature(val_logits["consistency"], y_cons_val)

    def calibrated(logits_by_head: dict) -> dict:
        out = {}
        for head in ("text_only", "image_only", "fusion", "consistency"):
            out[head] = _softmax(logits_by_head[head] / temperatures[head])
        out["fusion_cosine"] = logits_by_head["fusion_cosine"]
        out["fusion_gates"] = logits_by_head["fusion_gates"]
        return out

    val_probs = calibrated(val_logits)
    test_probs = calibrated(test_logits)

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            **data["meta"]["dataset"],
            "split": {"train": len(split["train"]), "validation": len(val_idx),
                      "test": len(test_idx)},
            "test_class_balance": {"credible": int((y_test == 0).sum()),
                                   "misleading": int((y_test == 1).sum())},
            "seed": split.get("seed", 42),
        },
        "encoders": split.get("encoders", {}),
        "temperature": temperatures,
        "models": {},
    }

    for kind, name in MODEL_NAMES.items():
        threshold = _best_threshold(y_val, val_probs[kind][:, 1])
        metrics = _metrics(y_test, test_probs[kind][:, 1], threshold)
        metrics["name"] = name
        report["models"][kind] = metrics

    # Cross-modal consistency head quality on the test split.
    cons_pred = test_probs["consistency"].argmax(1)
    y_cons = data["consistency"][test_idx]
    report["cross_modal_consistency"] = {
        "accuracy": round(float((cons_pred == y_cons).mean()), 4),
        "auc": round(float(roc_auc_score(y_cons, test_probs["consistency"][:, 1])), 4),
        "mean_cosine_consistent": round(float(
            test_probs["fusion_cosine"][y_cons == 1].mean()), 4),
        "mean_cosine_inconsistent": round(float(
            test_probs["fusion_cosine"][y_cons == 0].mean()), 4),
        "modality_gates": {
            "text": round(float(test_probs["fusion_gates"][:, 0].mean()), 4),
            "image": round(float(test_probs["fusion_gates"][:, 1].mean()), 4),
        },
    }

    fusion_metrics = report["models"]["fusion"]
    report["chosen_threshold"] = fusion_metrics["threshold"]
    report["comparison"] = {
        "fusion_vs_text_only_accuracy": round(
            fusion_metrics["accuracy"] - report["models"]["text_only"]["accuracy"], 4),
        "fusion_vs_image_only_accuracy": round(
            fusion_metrics["accuracy"] - report["models"]["image_only"]["accuracy"], 4),
        "fusion_vs_text_only_f1": round(
            fusion_metrics["f1"] - report["models"]["text_only"]["f1"], 4),
    }
    report["training"] = split.get("training", {})
    report["history"] = split.get("history", {})

    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    for kind, m in report["models"].items():
        print(f"{kind:<11} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
              f"prec={m['precision']:.3f}  rec={m['recall']:.3f}  auc={m['auc']:.3f}")
    print("consistency:", report["cross_modal_consistency"])
    print("comparison:", report["comparison"])


if __name__ == "__main__":
    main()
