"""Threshold sweep for the pneumonia classifier — finds the best operating point.

Loads the same validation split as evaluate.py (seed 42), computes metrics at
multiple thresholds, and saves a sweep table + recommended threshold.

Output: evaluation/threshold_sweep.json
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from app.core.config import DATA_DIR, MODEL_DIR, ROOT_DIR, IMAGE_SIZE
from app.core.dataset import RSNADataset
from app.core.preprocessing import get_val_transform
from app.models.classifier import load_model

EVAL_DIR = os.path.join(ROOT_DIR, "evaluation")
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(os.path.join(MODEL_DIR, "best_model.pth"), device=device).to(device)

    csv_path = os.path.join(DATA_DIR, "raw", "stage_2_train_labels.csv")
    img_dir = os.path.join(DATA_DIR, "raw", "stage_2_train_images")
    ds = RSNADataset(csv_path, img_dir, transform=get_val_transform(IMAGE_SIZE))

    n_val = int(0.15 * len(ds))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(42)
    _, val_idx = torch.utils.data.random_split(range(len(ds)), [n_train, n_val], generator=g)
    val_idx = list(val_idx)
    print(f"Validation images: {len(val_idx)}")

    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for i in val_idx:
            img, label, _ = ds[i]
            probs.append(model(img.unsqueeze(0).to(device)).item())
            labels.append(int(label))
    probs = np.array(probs)
    labels = np.array(labels)

    rows = []
    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(labels, preds)
        row = {
            "threshold": t,
            "accuracy": round(float(accuracy_score(labels, preds)), 4),
            "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
            "specificity": round(float(cm[0, 0] / (cm[0, 0] + cm[0, 1])), 4),
            "TP": int(cm[1, 1]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TN": int(cm[0, 0]),
        }
        rows.append(row)

    # Recommendation: maximise F1 as a balanced operating point
    best = max(rows, key=lambda r: r["f1"])
    out = {"rows": rows, "recommended_threshold": best["threshold"], "basis": "max F1"}
    with open(os.path.join(EVAL_DIR, "threshold_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'thr':>5} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'spec':>7} {'FP':>5} {'FN':>5}")
    for r in rows:
        print(f"{r['threshold']:>5} {r['accuracy']:>7} {r['precision']:>7} {r['recall']:>7} "
              f"{r['f1']:>7} {r['specificity']:>7} {r['FP']:>5} {r['FN']:>5}")
    print(f"\nRecommended threshold: {best['threshold']} (max F1 = {best['f1']})")
    print(f"Saved to {os.path.join(EVAL_DIR, 'threshold_sweep.json')}")


if __name__ == "__main__":
    main()
