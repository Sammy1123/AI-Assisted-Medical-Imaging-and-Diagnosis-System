"""
Preprocessing utilities for RSNA Pneumonia Detection dataset.
Handles DICOM reading, resizing, normalisation, and augmentation.
"""
import os
import numpy as np
import pydicom
from PIL import Image
import torchvision.transforms as T

# ImageNet stats used for DenseNet pre-training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_dicom(path: str) -> np.ndarray:
    """Read a DICOM file and return a uint8 numpy array (H, W)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    # Normalise to [0, 255]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    return arr.astype(np.uint8)


def dicom_to_pil(path: str) -> Image.Image:
    """Load DICOM and convert to 3-channel PIL Image."""
    arr = load_dicom(path)
    img = Image.fromarray(arr).convert("RGB")
    return img


def get_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
