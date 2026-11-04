"""
Training script for 3D U-Net segmentation (Medical Segmentation Decathlon).
Usage:
    python train_segmentation.py --task Task09_Spleen --epochs 50
"""
import argparse
import json
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ToTensord, AsDiscrete,
)
from monai.utils import set_determinism

from app.core.config import DATA_DIR, MODEL_DIR
from app.models.segmentation import build_segmentation_model

set_determinism(seed=42)


def get_transforms(train: bool):
    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image"),
    ]
    if train:
        base += [
            RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotate90d(keys=keys, prob=0.1, max_k=3),
        ]
    base.append(ToTensord(keys=keys))
    return Compose(base)


def build_datalist(task_dir: str):
    images = sorted([
        os.path.join(task_dir, "imagesTr", f)
        for f in os.listdir(os.path.join(task_dir, "imagesTr")) if f.endswith(".nii.gz")
    ])
    labels = sorted([
        os.path.join(task_dir, "labelsTr", f)
        for f in os.listdir(os.path.join(task_dir, "labelsTr")) if f.endswith(".nii.gz")
    ])
    return [{"image": i, "label": l} for i, l in zip(images, labels)]


def train(task: str, epochs: int, batch_size: int, lr: float):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    task_dir  = os.path.join(DATA_DIR, "raw", task)
    datalist  = build_datalist(task_dir)
    split     = int(0.8 * len(datalist))
    train_ds  = Dataset(data=datalist[:split],  transform=get_transforms(train=True))
    val_ds    = Dataset(data=datalist[split:],  transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=2)

    model     = build_segmentation_model(device=device)
    loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    metric    = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label= AsDiscrete(to_onehot=2)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_dice = 0.0
    history = []   # per-epoch metrics, persisted to train_history_segmentation.json

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss_fn(model(imgs), lbls).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                imgs, lbls = batch["image"].to(device), batch["label"].to(device)
                preds = sliding_window_inference(imgs, (96, 96, 96), 4, model)
                preds = [post_pred(p) for p in decollate_batch(preds)]
                lbls  = [post_label(l) for l in decollate_batch(lbls)]
                metric(y_pred=preds, y=lbls)

        dice = metric.aggregate().item()
        metric.reset()
        history.append({"epoch": epoch, "val_dice": round(float(dice), 4)})
        print(f"Epoch {epoch}/{epochs}  val_Dice={dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_seg_model.pth"))
            print(f"  → Saved best segmentation model (Dice={best_dice:.4f})")

        scheduler.step()

    with open(os.path.join(MODEL_DIR, "train_history_segmentation.json"), "w") as f:
        json.dump({
            "config": {"task": task, "epochs": epochs, "batch_size": batch_size, "lr": lr},
            "best_dice": round(float(best_dice), 4),
            "history": history,
        }, f, indent=2)
    print(f"History saved to {os.path.join(MODEL_DIR, 'train_history_segmentation.json')}")

    print(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       type=str,   default="Task09_Spleen")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    train(args.task, args.epochs, args.batch_size, args.lr)
