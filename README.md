# AI-Assisted Medical Imaging and Diagnosis System

> ⚠️ **FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE**

A full-stack platform for chest X-ray pneumonia detection and volumetric CT/MRI segmentation using deep learning, Grad-CAM explainability, Orthanc DICOM server integration, and a React radiologist review interface.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Compose                        │
│                                                             │
│  ┌──────────┐    DICOM    ┌─────────────────────────────┐  │
│  │  Orthanc │◄───────────►│   Orthanc Worker (Python)   │  │
│  │  :8042   │             │   Polls → Infer → Tag        │  │
│  └──────────┘             └─────────────────────────────┘  │
│       │                              │                       │
│       │                    ┌─────────▼──────────┐           │
│       │                    │  FastAPI Backend    │           │
│       │                    │  :8000              │           │
│       │                    │  /api/predict       │           │
│       │                    │  /api/review        │           │
│       │                    │  /api/audit         │           │
│       │                    └─────────┬──────────┘           │
│       │                              │                       │
│       │                    ┌─────────▼──────────┐           │
│       └────────────────────│  React Frontend     │           │
│                            │  :3000              │           │
│                            │  Viewer + AuditLog  │           │
│                            └────────────────────┘           │
└─────────────────────────────────────────────────────────────┘

Models:
  Classification : DenseNet-121 (transfer learning) + Grad-CAM
  Segmentation   : 3D U-Net (MONAI) for CT/MRI volumes
Data versioning  : DVC → S3
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Classification model | PyTorch + DenseNet-121 |
| Segmentation model | MONAI + 3D U-Net |
| Explainability | Grad-CAM |
| DICOM server | Orthanc |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 |
| Data versioning | DVC |
| Containerisation | Docker + Docker Compose |

---

## Quick Start

### 1. Clone & set up data

```bash
git clone <repo-url>
cd "AI-Assisted Medical Imaging and Diagnosis System"

# Pull versioned data (requires DVC + AWS credentials)
dvc pull
```

Download RSNA dataset from Kaggle into `data/raw/`:
- `stage_2_train_labels.csv`
- `stage_2_train_images/` (DICOM files)

### 2. Run with Docker (recommended)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Orthanc | http://localhost:8042 |

### 3. Run locally (development)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm start
```

---

## Training

### Classification (DenseNet-121)

```bash
cd backend
python train.py --epochs 10 --batch_size 32 --lr 1e-4
```

### Segmentation (3D U-Net, MONAI)

```bash
cd backend
python train_segmentation.py --task Task09_Spleen --epochs 50
```

### DVC pipeline (runs both)

```bash
dvc repro
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Upload DICOM/PNG → prediction + Grad-CAM overlay |
| POST | `/api/review` | Radiologist approve/reject decision |
| GET | `/api/audit` | Retrieve audit log entries |

Full interactive docs at `http://localhost:8000/docs`

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints
│   │   ├── core/
│   │   │   ├── config.py          # Paths & constants
│   │   │   ├── dataset.py         # RSNA PyTorch Dataset
│   │   │   └── preprocessing.py   # DICOM → PIL, transforms
│   │   ├── models/
│   │   │   ├── classifier.py      # DenseNet-121
│   │   │   ├── gradcam.py         # Grad-CAM
│   │   │   └── segmentation.py    # MONAI 3D U-Net
│   │   └── services/
│   │       └── orthanc_worker.py  # Orthanc polling service
│   ├── train.py                   # Classification training
│   ├── train_segmentation.py      # Segmentation training
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── components/Viewer.jsx  # Upload + Grad-CAM viewer
│   │   └── pages/AuditLog.js     # Audit log table
│   └── package.json
├── data/
│   ├── raw/                       # RSNA DICOM + CSV (DVC tracked)
│   └── processed/                 # Preprocessed outputs (DVC tracked)
├── models/weights/                # Trained model weights (DVC tracked)
├── docker-compose.yml
├── dvc.yaml
├── Main.py                        # Root launcher
└── SESSION_HISTORY.md
```

---

## Model Performance (Reference Targets)

| Model | Dataset | Metric | Target |
|-------|---------|--------|--------|
| DenseNet-121 | RSNA Pneumonia | AUC-ROC | ≥ 0.85 |
| 3D U-Net | MSD Task09 Spleen | Dice | ≥ 0.90 |

---

## Limitations & Bias

- Trained on RSNA Pneumonia Detection Challenge data (US hospital demographics)
- Not validated on paediatric, non-PA view, or non-chest X-ray images
- Grad-CAM is an approximation — not a clinical-grade explanation
- **Not approved for clinical diagnosis under any regulatory framework**

---

## Disclaimer

> This system is built for **educational and research purposes only**.
> It has **not** been validated for clinical use and must **not** be used to make medical decisions.
