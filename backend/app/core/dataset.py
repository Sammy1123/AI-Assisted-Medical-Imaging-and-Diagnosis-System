"""
PyTorch Dataset for the RSNA Pneumonia Detection Challenge.
Expects:
  data/raw/stage_2_train_images/  — DICOM files
  data/raw/stage_2_train_labels.csv
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from .preprocessing import dicom_to_pil


class RSNADataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        df = pd.read_csv(csv_path)
        # One row per patient (drop duplicate patient rows keeping first)
        self.df = df.drop_duplicates(subset="patientId").reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm_path = os.path.join(self.img_dir, f"{row['patientId']}.dcm")
        img = dicom_to_pil(dcm_path)
        if self.transform:
            img = self.transform(img)
        label = float(row["Target"])
        return img, label, row["patientId"]
