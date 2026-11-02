# TODO — AI-Assisted Medical Imaging and Diagnosis System

## 🔴 Next Session (Priority)

- [ ] Run training: `python train.py --epochs 10 --batch_size 32 --lr 1e-4`
- [ ] Wait for training to complete (~1-2 hours on CPU, faster on GPU)
- [ ] Verify `models/weights/best_model.pth` is created after training
- [ ] Test `/api/predict` endpoint with a sample DICOM file from `data/raw/stage_2_test_images/`

## 🟡 Frontend

- [ ] Test full flow on browser: upload image → see Grad-CAM → approve/reject
- [ ] Verify Audit Log page shows entries after predictions
- [ ] Fix npm audit vulnerabilities: `npm audit fix`
- [ ] Start frontend properly with backend URL: `set REACT_APP_API_URL=http://localhost:8080/api && npm start`

## 🟡 Backend

- [ ] Start backend on port 8080: `py -m uvicorn main:app --reload --port 8080`
- [ ] Verify API docs at http://localhost:8080/docs
- [ ] Test all 3 endpoints: `/api/predict`, `/api/review`, `/api/audit`

## 🟢 Optional / Later

- [ ] Train segmentation model: `python train_segmentation.py --task Task09_Spleen --epochs 50`
  - Requires Medical Segmentation Decathlon dataset download
- [ ] Set up DVC remote (S3 bucket) in `.dvc/config`
- [ ] Run `dvc init && dvc repro` for full pipeline
- [ ] Set up Orthanc DICOM server (Docker required)
- [ ] Push trained model weights to DVC remote
- [ ] Commit trained weights reference to GitHub
- [ ] Restrict CORS `allow_origins` in `backend/main.py` before any deployment
- [ ] Record 10-minute demo video for final presentation
- [ ] Write individual intern reflections

## ✅ Done

- [x] Project structure created
- [x] All 6 bugs fixed (config paths, pretrained API, type hints, double forward pass, transform leakage, GradCAM last_prob)
- [x] MONAI 3D U-Net segmentation model added
- [x] Orthanc worker service added
- [x] React frontend scaffolded (App.js, AuditLog.js, index.js, package.json)
- [x] Docker setup (Dockerfile x2, docker-compose.yml)
- [x] DVC pipeline configured (dvc.yaml, .dvc/config)
- [x] .gitignore created (RSNA data excluded)
- [x] RSNA dataset downloaded (26,684 DICOM files)
- [x] All dependencies installed (Python 3.10 + Node.js v26)
- [x] Frontend npm install completed
- [x] Backend + Frontend verified running (ports 8080 + 3000)
- [x] Pushed to GitHub: https://github.com/Sammy1123/AI-Assisted-Medical-Imaging-and-Diagnosis-System
- [x] Fixed DATA_DIR/MODEL_DIR path resolution in config.py

## 📝 Commands Cheatsheet

### Start Backend
```cmd
cd /d "d:\AI-Assisted Medical Imaging and Diagnosis System\backend"
py -m uvicorn main:app --reload --port 8080
```

### Start Frontend
```cmd
cd /d "d:\AI-Assisted Medical Imaging and Diagnosis System\frontend"
set REACT_APP_API_URL=http://localhost:8080/api
npm start
```

### Run Training
```cmd
cd /d "d:\AI-Assisted Medical Imaging and Diagnosis System\backend"
python train.py --epochs 10 --batch_size 32 --lr 1e-4
```
