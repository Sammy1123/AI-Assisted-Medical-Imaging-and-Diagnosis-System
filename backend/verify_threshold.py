import json
import sys

sys.path.insert(0, r"D:\AI-Assisted Medical Imaging and Diagnosis System\backend")
from app.core.config import POSITIVE_THRESHOLD

print("config on disk: POSITIVE_THRESHOLD =", POSITIVE_THRESHOLD)
assert POSITIVE_THRESHOLD == 0.4, "threshold not updated!"

sw = json.load(open(r"D:\AI-Assisted Medical Imaging and Diagnosis System\evaluation\threshold_sweep.json"))
r4 = [r for r in sw["rows"] if r["threshold"] == 0.4][0]
r5 = [r for r in sw["rows"] if r["threshold"] == 0.5][0]
extra_tp = r4["TP"] - r5["TP"]
extra_fp = r4["FP"] - r5["FP"]
print(f"sweep at 0.4: TP {r4['TP']} / FP {r4['FP']}  |  at 0.5: TP {r5['TP']} / FP {r5['FP']}")
print(f"0.4 catches {extra_tp} extra true positives and {extra_fp} extra false positives vs 0.5")
print("OK: threshold verified at 0.4")
