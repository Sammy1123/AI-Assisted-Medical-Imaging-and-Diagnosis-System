"""
FastAPI routes:
  POST /predict   — upload DICOM/PNG, get prediction + Grad-CAM overlay
  POST /review    — radiologist Approve / Reject with audit log
  GET  /audit     — retrieve audit log entries
"""
import io
import json
import os
import base64
from datetime import datetime, timezone

import torch
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from ..core.config import MODEL_DIR, AUDIT_LOG_PATH, POSITIVE_THRESHOLD, DISCLAIMER, IMAGE_SIZE
from ..core.preprocessing import get_val_transform
from ..models.classifier import load_model
from ..models.gradcam import GradCAM, overlay_heatmap

router = APIRouter()

# ── Lazy-load model once ──────────────────────────────────────────────────────
_model = None
_gradcam = None
_transform = get_val_transform(IMAGE_SIZE)


def _get_model():
    global _model, _gradcam
    if _model is None:
        weights = os.path.join(MODEL_DIR, "best_model.pth")
        if not os.path.exists(weights):
            raise HTTPException(status_code=503, detail="Model weights not found. Train first.")
        _model = load_model(weights, device="cpu")
        _gradcam = GradCAM(_model)
    return _model, _gradcam


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _append_audit(entry: dict):
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept a PNG/JPEG/DICOM image and return prediction + Grad-CAM overlay."""
    model, gradcam = _get_model()

    raw = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        # Try DICOM via pydicom
        try:
            import pydicom, tempfile
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            from ..core.preprocessing import dicom_to_pil
            img_pil = dicom_to_pil(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    tensor = _transform(img_pil).unsqueeze(0)  # (1, 3, H, W)
    tensor.requires_grad_(True)

    model.eval()
    # Single forward+backward pass — gradcam.generate() returns prob + heatmap
    cam = gradcam.generate(tensor)
    prob = gradcam.last_prob
    overlay = overlay_heatmap(img_pil, cam)

    prediction_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    result = {
        "prediction_id": prediction_id,
        "probability": round(prob, 4),
        "label": "Pneumonia" if prob >= POSITIVE_THRESHOLD else "Normal",
        "confidence": round(max(prob, 1 - prob) * 100, 1),
        "overlay_b64": _pil_to_b64(overlay),
        "disclaimer": DISCLAIMER,
    }
    _append_audit({"event": "predict", "timestamp": datetime.now(timezone.utc).isoformat(), **result})
    return JSONResponse(result)


@router.post("/review")
async def review(
    prediction_id: str = Form(...),
    decision: str = Form(...),          # "approve" | "reject"
    radiologist_id: str = Form(...),
    notes: str = Form(""),
):
    """Record radiologist Approve/Reject decision."""
    if decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")
    entry = {
        "event": "review",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_id": prediction_id,
        "decision": decision,
        "radiologist_id": radiologist_id,
        "notes": notes,
    }
    _append_audit(entry)
    return {"status": "recorded", **entry}


@router.get("/audit")
def get_audit(limit: int = 100):
    """Return the last `limit` audit log entries."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH) as f:
        lines = f.readlines()
    entries = [json.loads(l) for l in lines[-limit:]]
    return entries
