"""Train the text-only, image-only and multimodal fusion models.

Usage:  python -m src.train
Artifacts written to models/: fusion.pt, text_only.pt, image_only.pt, split.json
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

import config
from src import features
from src.fusion_model import FusionNet, ImageOnlyNet, TextOnlyNet

SEED = 42
EPOCHS = 120
BATCH_SIZE = 48
LR = 9e-4
WEIGHT_DECAY = 1e-3
CONSISTENCY_LOSS_WEIGHT = 0.4
CONTRASTIVE_LOSS_WEIGHT = 0.3
LABEL_SMOOTHING = 0.08
PATIENCE = 25


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.TORCH_THREADS:
        torch.set_num_threads(config.TORCH_THREADS)


def stratified_split(label: np.ndarray, consistency: np.ndarray, seed: int = SEED):
    """70/15/15 train/val/test split stratified over (label, consistency)."""
    rng = np.random.RandomState(seed)
    strata = {}
    for idx in range(len(label)):
        key = (int(label[idx]), int(consistency[idx]))
        strata.setdefault(key, []).append(idx)

    train, val, test = [], [], []
    for key, idxs in sorted(strata.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        test.extend(idxs[:n_test].tolist())
        val.extend(idxs[n_test:n_test + n_val].tolist())
        train.extend(idxs[n_test + n_val:].tolist())
    return sorted(train), sorted(val), sorted(test)


def _tensors(data, idxs):
    return (
        torch.from_numpy(data["text"][idxs]),
        torch.from_numpy(data["image"][idxs]),
        torch.from_numpy(data["label"][idxs]),
        torch.from_numpy(data["consistency"][idxs]),
    )


class _Bundle:
    """Uniform fit/predict wrapper so all three models train identically."""

    def __init__(self, kind: str):
        self.kind = kind
        if kind == "fusion":
            self.model = FusionNet()
        elif kind == "text_only":
            self.model = TextOnlyNet()
        else:
            self.model = ImageOnlyNet()

    def _forward(self, t, i):
        if self.kind == "fusion":
            logits, consistency_logits, cos, _gates = self.model(t, i)
            return logits, consistency_logits, cos
        if self.kind == "text_only":
            return self.model(t), None, None
        return self.model(i), None, None


def train_model(kind: str, data, split) -> dict:
    set_seed()
    train_idx, val_idx, _ = split
    bundle = _Bundle(kind)
    model = bundle.model
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    t_tr, i_tr, y_tr, c_tr = _tensors(data, train_idx)
    t_va, i_va, y_va, c_va = _tensors(data, val_idx)

    counts = torch.bincount(y_tr, minlength=2).float()
    class_weight = (counts.sum() / (2 * counts.clamp(min=1))).to(torch.float32)

    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    history = []
    rng = np.random.RandomState(SEED)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(t_tr))
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            logits, cons_logits, cos = bundle._forward(t_tr[batch], i_tr[batch])
            # Label smoothing keeps the heads from saturating at P ~ 0.9999, which would
            # turn every stored confidence score into 100%.
            loss = nn.functional.cross_entropy(logits, y_tr[batch], weight=class_weight,
                                               label_smoothing=LABEL_SMOOTHING)
            if cons_logits is not None:
                loss = loss + CONSISTENCY_LOSS_WEIGHT * nn.functional.cross_entropy(
                    cons_logits, c_tr[batch]
                )
            if cos is not None:
                # Contrastive objective: pull matching text/image projections together and
                # push mismatched pairs apart in the shared 256-d space.
                target = c_tr[batch].float()
                loss = loss + CONTRASTIVE_LOSS_WEIGHT * nn.functional.binary_cross_entropy(
                    (cos + 1.0) / 2.0, target
                )
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss) * len(batch)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            val_logits, val_cons, _ = bundle._forward(t_va, i_va)
            val_loss = float(nn.functional.cross_entropy(val_logits, y_va))
            val_acc = float((val_logits.argmax(1) == y_va).float().mean())
            entry = {
                "epoch": epoch,
                "train_loss": round(total / len(t_tr), 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
            }
            if val_cons is not None:
                entry["val_consistency_accuracy"] = round(
                    float((val_cons.argmax(1) == c_va).float().mean()), 4)
        history.append(entry)

        if val_loss < best["val_loss"] - 1e-5:
            best = {"val_loss": val_loss, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] > PATIENCE:
            break

    model.load_state_dict(best["state"])
    model.eval()
    torch.save(best["state"], config.MODELS_DIR / f"{kind}.pt")
    print(f"[train] {kind:<10} best epoch {best['epoch']:>3} "
          f"val_loss {best['val_loss']:.4f}")
    return {"best_epoch": best["epoch"], "best_val_loss": round(best["val_loss"], 5),
            "history": history}


def consistency_calibration(data, split) -> dict:
    """Cosine statistics of consistent vs inconsistent pairs (trained projector)."""
    fusion = FusionNet()
    fusion.load_state_dict(torch.load(config.MODELS_DIR / "fusion.pt"))
    fusion.eval()
    idxs = split[0] + split[1]
    t, i, _, c = _tensors(data, idxs)
    with torch.inference_mode():
        _, _, cos, _ = fusion(t, i)
    cos = cos.numpy()
    consistent = cos[c.numpy() == 1]
    inconsistent = cos[c.numpy() == 0]
    return {
        "consistent_mean": round(float(consistent.mean()), 4),
        "consistent_std": round(float(consistent.std()), 4),
        "inconsistent_mean": round(float(inconsistent.mean()), 4),
        "inconsistent_std": round(float(inconsistent.std()), 4),
        "midpoint": round(float((consistent.mean() + inconsistent.mean()) / 2), 4),
    }


def main() -> None:
    started = time.time()
    set_seed()
    data = features.build()
    label = data["label"]
    split = stratified_split(label, data["consistency"])

    print(f"[train] {len(label)} samples "
          f"(train {len(split[0])} / val {len(split[1])} / test {len(split[2])})")

    results = {}
    for kind in ("text_only", "image_only", "fusion"):
        results[kind] = train_model(kind, data, split)

    calibration = consistency_calibration(data, split)
    print(f"[train] consistency cosine calibration: {calibration}")

    (config.MODELS_DIR / "split.json").write_text(json.dumps({
        "train": split[0], "val": split[1], "test": split[2],
        "seed": SEED, "stratified_by": ["label", "consistency"],
        "consistency_cosine": calibration,
        "training": {k: {"best_epoch": v["best_epoch"],
                         "best_val_loss": v["best_val_loss"],
                         "epochs_run": len(v["history"])} for k, v in results.items()},
        "history": {k: v["history"] for k, v in results.items()},
        "dataset": data["meta"]["dataset"],
        "encoders": {"text": data["meta"]["text_model"],
                     "image": data["meta"]["image_model"]},
    }, indent=2))
    print(f"[train] done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
