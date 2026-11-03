"""
Training script for the pneumonia classifier.
Usage:
    python train.py --epochs 10 --batch_size 32 --lr 1e-4
"""
import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
import numpy as np

from app.core.config import DATA_DIR, MODEL_DIR, IMAGE_SIZE
from app.core.dataset import RSNADataset
from app.core.preprocessing import get_train_transform, get_val_transform
from app.models.classifier import build_model


def train(epochs: int, batch_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    csv_path = os.path.join(DATA_DIR, "raw", "stage_2_train_labels.csv")
    img_dir  = os.path.join(DATA_DIR, "raw", "stage_2_train_images")

    full_ds = RSNADataset(csv_path, img_dir, transform=None)
    n_val   = int(0.15 * len(full_ds))
    n_train = len(full_ds) - n_val
    train_indices, val_indices = random_split(
        range(len(full_ds)), [n_train, n_val]
    )

    from torch.utils.data import Subset
    # Create independent subsets with their own transforms to avoid leakage
    train_ds = Subset(RSNADataset(csv_path, img_dir, transform=get_train_transform(IMAGE_SIZE)), list(train_indices))
    val_ds   = Subset(RSNADataset(csv_path, img_dir, transform=get_val_transform(IMAGE_SIZE)),   list(val_indices))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    model     = build_model(pretrained=True).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_auc = 0.0
    history = []   # per-epoch metrics, persisted to train_history.json

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / n_train

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        auc = roc_auc_score(all_labels, all_preds)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_auc": round(float(auc), 4),
        })
        print(f"Epoch {epoch}/{epochs}  loss={train_loss:.4f}  val_AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"  → Saved best model (AUC={best_auc:.4f})")

        scheduler.step()

    with open(os.path.join(MODEL_DIR, "train_history.json"), "w") as f:
        json.dump({
            "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr},
            "best_auc": round(float(best_auc), 4),
            "history": history,
        }, f, indent=2)
    print(f"History saved to {os.path.join(MODEL_DIR, 'train_history.json')}")

    print(f"Training complete. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)
