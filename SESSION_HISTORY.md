# AI-Assisted Medical Imaging and Diagnosis System — Session History

---

## Session 1 — Project Audit & Issue Report

### Project Structure (Confirmed at Session 1)

```
AI-Assisted Medical Imaging and Diagnosis System/
├── backend/
│   ├── app/
│   │   ├── api/         routes.py, __init__.py
│   │   ├── core/        config.py, dataset.py, preprocessing.py, __init__.py
│   │   ├── models/      classifier.py, gradcam.py, __init__.py
│   │   └── __init__.py
│   ├── main.py
│   ├── requirements.txt
│   └── train.py
├── data/processed/, data/raw/
├── frontend/src/components/Viewer.jsx
├── notebooks/
├── Main.py          ← empty
└── rsna-pneumonia-detection-eda.ipynb
```

### Issues Found in Session 1

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 Critical | `Main.py` (root) | Empty file |
| 2 | 🔴 Critical | `backend/app/core/config.py` | Wrong `MODEL_DIR` path |
| 3 | 🔴 Critical | `backend/app/models/classifier.py` | Deprecated `pretrained=True` API |
| 4 | 🟡 Medium | `backend/train.py` | Transform data leakage via shared dataset |
| 5 | 🟡 Medium | `backend/app/models/gradcam.py` | `Tensor \| None` requires Python 3.10+ |
| 6 | 🟡 Medium | `backend/app/api/routes.py` | Double forward pass (redundant no_grad call) |
| 7 | 🟢 Low | `backend/main.py` | Wildcard CORS `allow_origins=["*"]` |
| 8 | 🟢 Low | `frontend/` | Incomplete React scaffold (no package.json, index.html, App.js) |
| 9 | 🟢 Info | `data/raw/` | Empty — RSNA dataset needs Kaggle download |

---

## Session 2 — Full Build-Out (All Deliverables)

### Bugs Fixed

| # | File | Fix Applied |
|---|------|-------------|
| 1 | `backend/app/core/config.py` | Fixed `MODEL_DIR` and `DATA_DIR` to resolve from project root via `ROOT_DIR` |
| 2 | `backend/app/models/classifier.py` | Replaced `pretrained=True` with `weights=DenseNet121_Weights.DEFAULT` |
| 3 | `backend/app/models/gradcam.py` | Replaced `Tensor \| None` with `Optional[torch.Tensor]` (Python 3.9 compatible) |
| 4 | `backend/app/models/gradcam.py` | Added `last_prob` attribute so routes.py can read probability from single pass |
| 5 | `backend/app/api/routes.py` | Removed redundant `no_grad` forward pass — single forward+backward via `gradcam.generate()` |
| 6 | `backend/train.py` | Fixed transform leakage — independent `RSNADataset` instances for train/val |

### New Files Created

#### Backend
| File | Purpose |
|------|---------|
| `backend/app/models/segmentation.py` | MONAI 3D U-Net — `build_segmentation_model()`, `load_segmentation_model()` |
| `backend/app/services/__init__.py` | Package init |
| `backend/app/services/orthanc_worker.py` | Polls Orthanc every 30s, runs inference, tags instances with metadata/9999 |
| `backend/train_segmentation.py` | Full MONAI training loop with DiceCELoss, sliding window inference, Dice metric |
| `backend/requirements.txt` | Added `monai[all]==1.3.0`, `requests==2.32.3`, `nibabel==5.2.1` |
| `backend/Dockerfile` | Python 3.11-slim, installs requirements, runs uvicorn |

#### Frontend
| File | Purpose |
|------|---------|
| `frontend/package.json` | React 18 + axios, proxy to :8000 |
| `frontend/public/index.html` | HTML shell |
| `frontend/src/index.js` | ReactDOM root render |
| `frontend/src/App.js` | Nav bar + page routing (Viewer / Audit Log) |
| `frontend/src/pages/AuditLog.js` | Fetches `/api/audit`, renders table |
| `frontend/Dockerfile` | Multi-stage: node build → nginx serve |

#### Infrastructure
| File | Purpose |
|------|---------|
| `docker-compose.yml` | Orchestrates orthanc, backend, orthanc_worker, frontend |
| `dvc.yaml` | DVC pipeline: preprocess → train_classifier → train_segmentation |
| `.dvc/config` | DVC remote config (S3) |
| `Main.py` | Root launcher — `cd backend && uvicorn main:app --reload` |
| `README.md` | Full setup, architecture diagram, API reference, model targets, disclaimer |

### Deliverables Status

| Deliverable | Status |
|-------------|--------|
| Working demo (Docker) | ✅ `docker-compose up --build` |
| GitHub-ready repo with README | ✅ `README.md` created |
| Architecture diagram | ✅ In `README.md` |
| API documentation | ✅ FastAPI `/docs` + README table |
| Classification model (DenseNet-121 + Grad-CAM) | ✅ |
| Segmentation model (MONAI 3D U-Net) | ✅ |
| Orthanc DICOM server integration | ✅ `orthanc_worker.py` |
| React viewer (scan + overlay + approve/reject) | ✅ `Viewer.jsx` |
| Audit log | ✅ `routes.py` + `AuditLog.js` |
| DVC data versioning | ✅ `dvc.yaml` + `.dvc/config` |
| Docker containerisation | ✅ `Dockerfile` × 2 + `docker-compose.yml` |
| Education-only disclaimer | ✅ In API, frontend, README |

### Pending / Manual Steps

- [ ] Download RSNA dataset from Kaggle → `data/raw/`
- [ ] Download Medical Segmentation Decathlon task → `data/raw/Task09_Spleen/`
- [ ] Configure real S3 bucket in `.dvc/config`
- [ ] Run `dvc init && dvc repro` after data is in place
- [ ] Run `git init` and make initial commit
- [ ] Restrict `allow_origins` in `backend/main.py` before production deployment
- [ ] Record 10-minute demo video for final presentation

---

## Final Project Structure (After Session 2)

```
AI-Assisted Medical Imaging and Diagnosis System/
├── .dvc/
│   └── config
├── backend/
│   ├── app/
│   │   ├── api/         routes.py (fixed)
│   │   ├── core/        config.py (fixed), dataset.py, preprocessing.py
│   │   ├── models/      classifier.py (fixed), gradcam.py (fixed), segmentation.py (new)
│   │   └── services/    orthanc_worker.py (new)
│   ├── main.py
│   ├── requirements.txt (updated)
│   ├── train.py (fixed)
│   ├── train_segmentation.py (new)
│   └── Dockerfile (new)
├── data/raw/, data/processed/
├── frontend/
│   ├── public/index.html (new)
│   ├── src/
│   │   ├── App.js (new)
│   │   ├── index.js (new)
│   │   ├── components/Viewer.jsx
│   │   └── pages/AuditLog.js (new)
│   ├── package.json (new)
│   └── Dockerfile (new)
├── models/weights/          ← DVC tracked
├── docker-compose.yml (new)
├── dvc.yaml (new)
├── Main.py (fixed)
├── README.md (new)
└── SESSION_HISTORY.md
```

---

## Session 3 — Environment Setup & First Run

### Completed
- Fixed `monai[all]` → `monai` (cucim not available on Windows)
- Identified Python 3.10 vs 3.12 conflict — installed all packages to Python 3.10
- Installed Node.js v26.13.0 (x64 MSI)
- Fixed PowerShell execution policy issue — used CMD instead
- Added Node.js to system environment variables PATH
- Completed `npm install --legacy-peer-deps` successfully (1295 packages)
- Fixed `DATA_DIR`/`MODEL_DIR` path — `config.py` had wrong number of `dirname()` calls
- Backend running on port 8080 (8000 blocked by Windows)
- Frontend running on port 3000
- Both services verified working in browser

### Errors Fixed
| Error | Fix |
|-------|-----|
| `cucim` not found on Windows | Changed `monai[all]` to `monai` in requirements.txt |
| `ModuleNotFoundError: monai` | Installed to correct Python 3.10 interpreter via `python -m pip` |
| npm PowerShell execution policy | Used CMD instead of PowerShell |
| `node not recognized` in npm scripts | Added `C:\Program Files\nodejs` to system PATH env variable |
| Port 8000 access denied | Switched to port 8080 |
| `FileNotFoundError: stage_2_train_labels.csv` | Fixed `BASE_DIR` in `config.py` — needed 3 `dirname()` calls not 2 |

### Pending for Next Session
- Run `python train.py --epochs 10 --batch_size 32 --lr 1e-4`
- Test full predict → review → audit flow in browser

---

## Session 4 — CUDA Setup & Training Start

### Completed
- Verified GTX 1650 GPU available via `nvidia-smi` (driver 610.74)
- CUDA Toolkit installer failed — not needed, PyTorch bundles CUDA libraries
- Installed `torch==2.3.0+cu121` and `torchvision==0.18.0+cu121` via PyTorch CUDA wheel
- Verified CUDA working: `torch.cuda.is_available() = True`, GPU = NVIDIA GeForce GTX 1650
- Started DenseNet-121 training on CUDA (`python train.py --epochs 10 --batch_size 32 --lr 1e-4`)
- Training running in CMD — expected time ~20-30 mins on GTX 1650
- Fixed VS Code terminal PATH by updating `.vscode/settings.json`
  - Added Python 3.10 and Node.js to VS Code integrated terminal PATH
  - `python` and `node` now work directly in VS Code terminal after restart

### Key Notes
- Project uses React (not Streamlit) — frontend runs via `npm start` on port 3000
- No Streamlit needed — React handles all UI
- Backend runs on port 8080 (8000 blocked by Windows)
- CUDA install command: `pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121`

### Pending
- [ ] Wait for training to complete → verify `models/weights/best_model.pth` created
- [ ] Check AUC score after training (target ≥ 0.85)
- [ ] Test `/api/predict` with a sample DICOM file
- [ ] Test full browser flow: upload → Grad-CAM → approve/reject → audit log
