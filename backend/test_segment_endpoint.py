"""Test /api/segment with a real Task09 volume via FastAPI TestClient."""
import sys, os
sys.path.insert(0, r"D:\AI-Assisted Medical Imaging and Diagnosis System\backend")
os.chdir(r"D:\AI-Assisted Medical Imaging and Diagnosis System\backend")

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

vol_path = r"D:\AI-Assisted Medical Imaging and Diagnosis System\data\raw\Task09_Spleen\imagesTr\spleen_2.nii.gz"
with open(vol_path, "rb") as f:
    r = client.post("/api/segment", files={"file": ("spleen_2.nii.gz", f, "application/octet-stream")})

print("status:", r.status_code)
j = r.json()
if r.status_code == 200:
    print("prediction_id:", j["prediction_id"])
    print("slices_with_mask:", j["n_slices_with_mask"], "| returned:", j["slices_returned"])
    print("voxels_segmented:", j["voxels_segmented"])
    print("overlay size (b64 chars):", [len(s) for s in j["slices_b64"][:3]])
else:
    print(j)
