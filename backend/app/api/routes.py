"""
FastAPI routes:
  POST /predict   — upload DICOM/PNG, get prediction + Grad-CAM overlay
  POST /review    — radiologist Approve / Reject with audit log
  GET  /audit     — retrieve audit log entries
"""
import io
import json
import os
import tempfile
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


# ── Segmentation ────────────────────────────────────────────────────────────
_seg_model = None

def _get_segmentation_model():
    global _seg_model
    if _seg_model is None:
        weights = os.path.join(MODEL_DIR, "best_seg_model.pth")
        if not os.path.exists(weights):
            raise HTTPException(
                status_code=503,
                detail="Segmentation weights not found. Run train_segmentation.py first.",
            )
        from ..models.segmentation import load_segmentation_model
        _seg_model = load_segmentation_model(weights, device="cpu")
    return _seg_model


@router.post("/segment")
async def segment(file: UploadFile = File(...)):
    """Run 3D U-Net segmentation on an uploaded NIfTI (.nii/.nii.gz) volume.

    Returns per-slice PNG overlays (axial view) of the predicted mask.
    FOR EDUCATION / RESEARCH ONLY — the model is undertrained (see PROGRESS.md).
    """
    import nibabel as nib
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Spacing,
        ScaleIntensityRange, CropForeground, AsDiscrete,
    )

    model = _get_segmentation_model()
    raw = await file.read()
    suffix = ".nii.gz" if file.filename and file.filename.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        # Same preprocessing chain as training (spacing/intensity/foreground)
        pre = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForeground(select_fn=lambda x: x > 0),
        ])
        vol = pre(tmp_path)                       # (1, H, W, D)
        with torch.no_grad():
            logits = sliding_window_inference(
                vol.unsqueeze(0), (96, 96, 96), 4, model
            )                                     # (1, 2, H, W, D)
        mask = AsDiscrete(argmax=True)(logits.squeeze(0))  # (H, W, D) in {0,1}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read NIfTI volume: {e}")
    finally:
        os.unlink(tmp_path)

    # Render the middle slices with the mask overlaid (max 8, evenly spaced)
    vol_np = vol.numpy()                          # (1, H, W, D)
    mask_np = mask.numpy().astype(np.uint8)       # (H, W, D) after argmax of (H,W,D) logits
    if vol_np.ndim == 4:
        vol_np = vol_np[0]                        # (H, W, D)
    if mask_np.ndim == 4:
        mask_np = mask_np[0]
    if vol_np.shape != mask_np.shape:
        raise HTTPException(
            status_code=500,
            detail=f"Shape mismatch image {vol_np.shape} vs mask {mask_np.shape} — check transforms",
        )

    img_np = vol_np                               # (H, W, D)
    seg_idx = np.unique(np.argwhere(mask_np > 0)[:, 2])   # axial slice indices
    if len(seg_idx) == 0:
        return JSONResponse({
            "prediction_id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
            "slices_b64": [],
            "n_slices_with_mask": 0,
            "slices_returned": 0,
            "voxels_segmented": 0,
            "disclaimer": DISCLAIMER,
        })
    chosen = seg_idx[np.linspace(0, len(seg_idx) - 1, min(8, len(seg_idx))).astype(int)]

    slices_b64 = []
    cmap_color = np.array([255, 60, 60], dtype=np.uint8)   # mask -> red
    for d in chosen:
        img2d = img_np[:, :, d]
        m2d = mask_np[:, :, d]
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-8) * 255
        rgb = np.stack([img2d] * 3, axis=-1).astype(np.uint8)
        rgb[m2d > 0] = cmap_color
        slices_b64.append(_pil_to_b64(Image.fromarray(rgb)))

    prediction_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    result = {
        "prediction_id": prediction_id,
        "n_slices_with_mask": int(len(seg_idx)),
        "slices_returned": len(slices_b64),
        "voxels_segmented": int(mask_np.sum()),
        "slice_indices": [int(d) for d in chosen],
        "slices_b64": slices_b64,
        "disclaimer": DISCLAIMER,
    }
    _append_audit({
        "event": "segment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_id": prediction_id,
        "voxels_segmented": result["voxels_segmented"],
        "filename": file.filename,
    })
    return JSONResponse(result)
