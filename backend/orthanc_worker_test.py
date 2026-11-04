"""One-shot Orthanc worker test: process every unprocessed instance, then exit."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ORTHANC_POLL_INTERVAL", "0")

from app.services.orthanc_worker import (
    ORTHANC_URL, _session, _get, _get_bytes, _store_result,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("worker-test")

model = None  # lazy import so we can report load errors clearly
from app.core.config import MODEL_DIR, IMAGE_SIZE
from app.core.preprocessing import dicom_to_pil, get_val_transform
from app.models.classifier import load_model
from app.models.gradcam import GradCAM

weights = os.path.join(MODEL_DIR, "best_model.pth")
model = load_model(weights, device="cpu")
gradcam = GradCAM(model)
transform = get_val_transform(IMAGE_SIZE)

import tempfile
import torch

instances = _get("/instances")
log.info("Orthanc has %d instance(s)", len(instances))

for iid in instances:
    meta = _session.get(f"{ORTHANC_URL}/instances/{iid}/metadata/9999")
    if meta.status_code == 200:
        log.info("%s already tagged: %s", iid[:8], meta.json())
        continue

    raw = _get_bytes(f"/instances/{iid}/file")
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        img = dicom_to_pil(path)
    finally:
        os.unlink(path)

    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_(True)
    model.eval()
    gradcam.generate(tensor)
    prob = gradcam.last_prob
    label = "Pneumonia" if prob >= 0.4 else "Normal"
    _store_result(iid, label, prob)
    log.info("RESULT %s -> %s (%.4f)", iid[:8], label, prob)

print("done")
