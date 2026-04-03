"""Data loading and evaluation for sleep stage classification.

Loads preprocessed .npz files from data/{train,val}/, creates DataLoaders,
and provides the fixed evaluation function.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

NUM_CLASSES = 5


class SleepDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.x = torch.from_numpy(data["x"].astype(np.float32))
        self.y = torch.from_numpy(data["y"].astype(np.int64))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataloaders(batch_size):
    """Returns (train_loader, val_loader)."""
    train_ds = SleepDataset("../../data/train/data.npz")
    val_ds = SleepDataset("../../data/val/data.npz")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader


# === DO NOT MODIFY THIS FUNCTION ===
def evaluate(model, loader, device):
    """Run inference and compute metrics. Returns dict with all metric keys.

    This function is the ground truth evaluation harness.
    Do not modify it.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch_y.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
    per_class_p = precision_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)

    for c in range(NUM_CLASSES):
        metrics[f"class_{c}_f1"] = per_class_f1[c]
        metrics[f"class_{c}_precision"] = per_class_p[c]
        metrics[f"class_{c}_recall"] = per_class_r[c]

    return metrics
