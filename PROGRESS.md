# PROGRESS — AI-Assisted Medical Imaging & Diagnosis System

> Reference file kept up to date across sessions. Last updated: 2026-11-03 (session 6)
> GitHub: https://github.com/HP/AI-Assisted-Medical-Imaging-and-Diagnosis-System

---

## 1. Current Status — TRAINING COMPLETE ✅

| Item     | Value                                                                          |
| -------- | ------------------------------------------------------------------------------ |
| Model    | DenseNet-121 (torchvision, ImageNet-pretrained), 1-output head                 |
| Task     | Binary pneumonia classification (RSNA Pneumonia Detection Challenge)           |
| Weights  | `models/weights/best_model.pth` (28 MB, saved 2026-11-03 13:23)                |
| Training | 10 epochs on GTX 1650 (`torch 2.3.0+cu121`), `BCEWithLogitsLoss`, Adam lr=1e-4 |
| Data     | 26,971 RSNA train images; 85/15 train/val split (seed 42)                      |
| Backend  | FastAPI on port 8080 — `/predict` with Grad-CAM overlay working                |
| Frontend | React (create-react-app) on port 3000 — upload + review working                |
| Audit    | `audit_log.jsonl` — 4 events (2 predictions, 2 reviews)                        |

---

## 2. Evaluation Results — GRAPH DATA

### 2.0 Full-Split Evaluation (FINAL NUMBERS, 2026-11-03)

Ran `backend/evaluate.py --max_images 0` — all **4,002** validation images (15% split, seed 42).

| Metric | Value |
|---|---|
| **ROC AUC** | **0.9079** |
| Average Precision (PR AUC) | 0.7499 |
| Accuracy @ 0.5 | 0.8613 |
| Precision @ 0.5 | 0.7081 |
| Recall @ 0.5 | 0.6284 |
| F1 @ 0.5 | 0.6659 |
| Threshold | 0.5 |
| Cases evaluated | 4,002 (880 pneumonia / 3,122 normal) |

Confusion matrix @ 0.5:

|  | Pred Normal | Pred Pneumonia |
|---|---|---|
| **Actual Normal** | TN = 2,894 | FP = 228 |
| **Actual Pneumonia** | FN = 327 | TP = 553 |

Derived: specificity = 2894/3122 = 0.9270, sensitivity = 553/880 = 0.6284, PPV = 553/781 = 0.7081, NPV = 2894/3221 = 0.8985.

Score histogram (counts per class, bins 0→1 step 0.05):

| Bin | Normal | Pneumonia |
|---|---|---|
| 0.00–0.05 | 1700 | 27 |
| 0.05–0.10 | 356 | 20 |
| 0.10–0.15 | 200 | 27 |
| 0.15–0.20 | 165 | 45 |
| 0.20–0.25 | 107 | 31 |
| 0.25–0.30 | 98 | 16 |
| 0.30–0.35 | 85 | 48 |
| 0.35–0.40 | 68 | 30 |
| 0.40–0.45 | 51 | 40 |
| 0.45–0.50 | 64 | 43 |
| 0.50–0.55 | 38 | 34 |
| 0.55–0.60 | 43 | 35 |
| 0.60–0.65 | 24 | 43 |
| 0.65–0.70 | 36 | 49 |
| 0.70–0.75 | 23 | 57 |
| 0.75–0.80 | 27 | 45 |
| 0.80–0.85 | 14 | 62 |
| 0.85–0.90 | 9 | 61 |
| 0.90–0.95 | 10 | 102 |
| 0.95–1.00 | 4 | 65 |

### 2.1 Preview Evaluation (500 images, superseded by 2.0 but kept for reference)

Evaluation run on 500 validation images (`backend/evaluate.py --max_images 500`).

### 2.1 Headline Metrics (`evaluation/metrics.json`)

| Metric                     | Value                            |
| -------------------------- | -------------------------------- |
| **ROC AUC**                | **0.9048**                       |
| Average Precision (PR AUC) | 0.7341                           |
| Accuracy @ 0.5             | 0.8620                           |
| Precision @ 0.5            | 0.6882                           |
| Recall @ 0.5               | 0.6154                           |
| F1 @ 0.5                   | 0.6497                           |
| Threshold                  | 0.5                              |
| Cases evaluated            | 500 (104 pneumonia / 396 normal) |

### 2.2 Confusion Matrix @ 0.5

|                      | Pred Normal | Pred Pneumonia |
| -------------------- | ----------- | -------------- |
| **Actual Normal**    | TN = 367    | FP = 29        |
| **Actual Pneumonia** | FN = 40     | TP = 64        |

Derived: specificity = 367/396 = 0.9268, sensitivity = 64/104 = 0.6154, PPV = 64/93 = 0.6882, NPV = 367/407 = 0.9017.

### 2.3 ROC Curve Data (`evaluation/graph_data.json` → `roc_curve`)

Full arrays in `evaluation/graph_data.json`. Summary — AUC = 0.9048, chance line plotted for reference. Curve reaches TPR ≈ 0.62 at FPR ≈ 0.20; TPR ≈ 0.90 at FPR ≈ 0.45.

### 2.4 Precision-Recall Curve Data

AP = 0.7341. Precision stays ≥ 0.69 up to recall ≈ 0.62; drops to ≈ 0.60 by recall ≈ 0.75.

### 2.5 Prediction Score Histogram (density per class)

Bins edges 0 → 1 step 0.05.

| Bin       | Normal | Pneumonia |
| --------- | ------ | --------- |
| 0.00–0.05 | 223    | 3         |
| 0.05–0.10 | 36     | 4         |
| 0.10–0.15 | 21     | 3         |
| 0.15–0.20 | 24     | 4         |
| 0.20–0.25 | 10     | 4         |
| 0.25–0.30 | 15     | 3         |
| 0.30–0.35 | 16     | 7         |
| 0.35–0.40 | 13     | 4         |
| 0.40–0.45 | 4      | 5         |
| 0.45–0.50 | 5      | 3         |
| 0.50–0.55 | 6      | 1         |
| 0.55–0.60 | 8      | 6         |
| 0.60–0.65 | 4      | 8         |
| 0.65–0.70 | 4      | 8         |
| 0.70–0.75 | 3      | 5         |
| 0.75–0.80 | 0      | 6         |
| 0.80–0.85 | 0      | 6         |
| 0.85–0.90 | 2      | 8         |
| 0.90–0.95 | 1      | 8         |
| 0.95–1.00 | 1      | 8         |

Reads well — normal cases pile at low scores, pneumonia at high scores; threshold 0.5 splits cleanly.

### 2.6 Plots saved (kept for reference)

- `evaluation/roc_curve.png`
- `evaluation/pr_curve.png`
- `evaluation/confusion_matrix.png`
- `evaluation/score_histogram.png`

(Regenerated by the full-split run — these are the final plots.)

---

## 3. Session History

| #   | Date       | What happened                                                                                               |
| --- | ---------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | 2026-11-02 | Project scaffolded; training script written; Docker/DVC/DICOM setup                                         |
| 2   | 2026-11-02 | Full training run on GTX 1650; FastAPI backend + React frontend working                                     |
| 3   | 2026-11-02 | Pushed to GitHub; README written; TODO updated                                                              |
| 4   | 2026-11-03 | Grad-CAM integrated; `/predict` with overlay working; audit log event 1                                     |
| 5   | 2026-11-03 | **Training complete — `best_model.pth` saved.** Evaluation run → metrics + graph data. PROGRESS.md created. |
| 6   | 2026-11-03 | Patched `train.py` to persist `train_history.json`; restricted CORS (config-driven, env-overridable); **full-split evaluation on 4,002 images — ROC AUC 0.9079**; docs updated; committed to GitHub. |

---

## 4. Known Issues / Notes

1. ~~`train.py` prints per-epoch loss/AUC to the console only~~ **Fixed (session 6)** — now writes `models/weights/train_history.json` with per-epoch loss/AUC + config. Note: the *already-completed* training run predates this patch, so no historical curve exists for it; re-training would produce one.
2. Grad-CAM target layer `"features.denseblock4"` resolved correctly via `named_modules()` — verified working.
3. `evaluate.py` defaults to 500 validation images for speed; pass `--max_images 0` to evaluate on the full 15% split (4,002 images) — **done in session 6**.
4. Audit `review` events have no structured `probability` field — they reference a `prediction_id` instead.
5. Full-split evaluation takes ~30 min on GTX 1650 (all 4,002 images processed one at a time). Batched inference would speed this up if it needs re-running often.

---

## 5. Next Steps (priority order)

1. Lower Grad-CAM threshold to a tuned operating point (e.g. 0.35) if recall matters more — current recall 0.6284 vs precision 0.7081.
2. Push `models/weights/` to a DVC remote (currently only local).
3. Record 10-minute demo video for final presentation.
4. Batched inference in `evaluate.py` (currently image-by-image; batching would cut eval time ~10x).
5. Consider a second training run with the new `train_history.json` patch to capture the training curve for the report.
