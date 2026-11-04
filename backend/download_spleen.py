"""Smoke-test: download MSD Task09_Spleen via MONAI's DecathlonDataset."""
import os

os.environ["MONAI_DATA_DIRECTORY"] = r"D:\AI-Assisted Medical Imaging and Diagnosis System\data\raw"

from monai.apps import DecathlonDataset

ds = DecathlonDataset(
    root_dir=os.environ["MONAI_DATA_DIRECTORY"],
    task="Task09_Spleen",
    section="training",
    transform=None,
    download=True,
    cache_num=0,          # don't cache transformed volumes yet
)
print("items:", len(ds))
print("first item keys:", list(ds[0].keys()) if len(ds) else "none")
