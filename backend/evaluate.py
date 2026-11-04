"""
Evaluation script for the trained pneumonia classifier.
Loads models/weights/best_model.pth, runs inference on the validation split,
and saves metrics + graph data for progress reporting.

Usage:
    python evaluate.py [--max_images 500]

Outputs (project root / evaluation/):
    evaluation/metrics.json       — headline metrics + per-epoch history (if available)
    evaluation/graph_data.json    — ROC curve, PR curve, confusion matrix, score histogram
    evaluation/roc_curve.png      — ROC curve plot
    evaluation/pr_curve.png       — Precision-Recall curve plot
    evaluation/confusion_matrix.png
    evaluation/score_histogram.png
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.config import DATA_DIR, MODEL_DIR, ROOT_DIR, IMAGE_SIZE
from app.core.dataset import RSNADataset
from app.core.preprocessing import get_val_transform
from app.models.classifier import load_model

EVAL_DIR = os.path.join(ROOT_DIR, "evaluation")


def main(max_images: int, batch_size: int = 32):
    os.makedirs(EVAL_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    weights_path = os.path.join(MODEL_DIR, "best_model.pth")
    model = load_model(weights_path, device=device).to(device)

    csv_path = os.path.join(DATA_DIR, "raw", "stage_2_train_labels.csv")
    img_dir = os.path.join(DATA_DIR, "raw", "stage_2_train_images")

    # Subset of the full dataset holding only the validation indices, with a
    # DataLoader for batched inference (decodes DICOMs in parallel workers).
    ds = RSNADataset(csv_path, img_dir, transform=get_val_transform(IMAGE_SIZE))
    # Same 15% validation split logic as train.py (seed 42 for reproducibility)
    n_val = int(0.15 * len(ds))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(42)
    _, val_indices = torch.utils.data.random_split(range(len(ds)), [n_train, n_val], generator=g)
    val_indices = list(val_indices)
    if max_images and max_images < len(val_indices):
        rng = np.random.default_rng(42)
        val_indices = sorted(rng.choice(val_indices, size=max_images, replace=False).tolist())
    print(f"Validation images: {len(val_indices)}")

    val_ds = torch.utils.data.Subset(ds, val_indices)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    probs, labels, ids = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls, pids in loader:
            logits = model(imgs.to(device))            # (B, 1)
            probs.extend(logits.squeeze(1).cpu().tolist())
            labels.extend(l for l in lbls)
            ids.extend(pids)
    probs = np.array(probs)
    labels = np.array(labels, dtype=int)

    # ---- Metrics ----
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    metrics = {
        "n_evaluated": int(len(labels)),
        "positive_cases": int(labels.sum()),
        "negative_cases": int((labels == 0).sum()),
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "accuracy@0.5": round(float(accuracy_score(labels, preds)), 4),
        "precision@0.5": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall@0.5": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1@0.5": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "threshold": 0.5,
    }
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = {
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }

    # ---- Graph data ----
    fpr, tpr, _ = roc_curve(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    graph = {
        "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist(), "auc": round(float(auc), 4)},
        "pr_curve": {"precision": prec.round(4).tolist(), "recall": rec.round(4).tolist(), "average_precision": round(float(ap), 4)},
        "confusion_matrix": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
        "score_histogram": {
            "edges": np.linspace(0, 1, 21).round(2).tolist(),
            "normal_counts": np.histogram(probs[labels == 0], bins=20, range=(0, 1))[0].tolist(),
            "pneumonia_counts": np.histogram(probs[labels == 1], bins=20, range=(0, 1))[0].tolist(),
        },
    }

    with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(EVAL_DIR, "graph_data.json"), "w") as f:
        json.dump(graph, f, indent=2)

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — DenseNet-121 Pneumonia Classifier"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(EVAL_DIR, "roc_curve.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"PR (AP = {ap:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(EVAL_DIR, "pr_curve.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Pred Normal", "Pred Pneumonia"])
    ax.set_yticks([0, 1], ["Actual Normal", "Actual Pneumonia"])
    ax.set_title("Confusion Matrix @ 0.5")
    fig.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[labels == 0], bins=20, alpha=0.6, label="Normal", density=True)
    ax.hist(probs[labels == 1], bins=20, alpha=0.6, label="Pneumonia", density=True)
    ax.axvline(0.5, color="k", linestyle="--", label="Threshold 0.5")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Density")
    ax.set_title("Prediction Score Distribution"); ax.legend()
    fig.savefig(os.path.join(EVAL_DIR, "score_histogram.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\nAll outputs saved to: {EVAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=500, help="Cap validation images for speed (0 = all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    args = parser.parse_args()
    main(args.max_images, args.batch_size)
