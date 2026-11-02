"""
Orthanc DICOM server integration service.
Polls Orthanc for new studies, runs inference, stores results back.

Usage:
    python -m app.services.orthanc_worker
"""
import os
import time
import logging
import tempfile
import requests
import torch

from ..core.config import POSITIVE_THRESHOLD, MODEL_DIR, IMAGE_SIZE
from ..core.preprocessing import dicom_to_pil, get_val_transform
from ..models.classifier import load_model
from ..models.gradcam import GradCAM

logger = logging.getLogger(__name__)

ORTHANC_URL   = os.getenv("ORTHANC_URL",  "http://localhost:8042")
ORTHANC_USER  = os.getenv("ORTHANC_USER", "orthanc")
ORTHANC_PASS  = os.getenv("ORTHANC_PASS", "orthanc")
POLL_INTERVAL = int(os.getenv("ORTHANC_POLL_INTERVAL", "30"))

_session = requests.Session()
_session.auth = (ORTHANC_USER, ORTHANC_PASS)


def _get(path: str):
    return _session.get(f"{ORTHANC_URL}{path}").json()


def _get_bytes(path: str) -> bytes:
    return _session.get(f"{ORTHANC_URL}{path}").content


def _store_result(instance_id: str, label: str, probability: float):
    """Attach AI result as metadata on the Orthanc instance (tag 9999)."""
    import json
    _session.put(
        f"{ORTHANC_URL}/instances/{instance_id}/metadata/9999",
        data=json.dumps({"label": label, "probability": round(probability, 4)}),
    )
    logger.info("Stored result for %s: %s (%.3f)", instance_id, label, probability)


def run_worker():
    weights = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found at {weights}. Run train.py first.")

    model     = load_model(weights, device="cpu")
    gradcam   = GradCAM(model)
    transform = get_val_transform(IMAGE_SIZE)
    processed: set = set()

    logger.info("Orthanc worker started. Polling every %ds", POLL_INTERVAL)

    while True:
        try:
            instances = _get("/instances")
            for instance_id in instances:
                if instance_id in processed:
                    continue
                # Skip if already tagged
                meta = _session.get(f"{ORTHANC_URL}/instances/{instance_id}/metadata/9999")
                if meta.status_code == 200:
                    processed.add(instance_id)
                    continue

                raw = _get_bytes(f"/instances/{instance_id}/file")
                with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                try:
                    img_pil = dicom_to_pil(tmp_path)
                finally:
                    os.unlink(tmp_path)

                tensor = transform(img_pil).unsqueeze(0)
                tensor.requires_grad_(True)
                model.eval()
                gradcam.generate(tensor)
                prob  = gradcam.last_prob
                label = "Pneumonia" if prob >= POSITIVE_THRESHOLD else "Normal"

                _store_result(instance_id, label, prob)
                processed.add(instance_id)

        except Exception as e:
            logger.error("Worker error: %s", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_worker()
