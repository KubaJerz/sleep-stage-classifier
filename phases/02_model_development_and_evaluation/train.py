"""Train and evaluate a sleep stage classifier."""

import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchlightning import F1Score

from prepare import get_dataloaders, evaluate

#== DONT EDIT BELOW THIS LINE ===  
TIME_BUDGET_SEC = 10 * 60  # 10 minutes wall clock
NUM_CLASSES = 5
# === DONT EDIT ABOVE THIS LINE ===


# === Hyperparameters ===
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4

# === Model ===

class Stem(nn.Module):
    """Per-epoch feature extraction."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, 1, 3000)
        return self.conv(x)  # (batch, 64, T)


class Backbone(nn.Module):
    """Temporal context aggregation."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, 64, T)
        return self.pool(x).squeeze(-1)  # (batch, 64)


class Head(nn.Module):
    """Classification head."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class SleepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem()
        self.backbone = Backbone()
        self.head = Head(64, NUM_CLASSES)

    def forward(self, x):
        # x: (batch, 1, 3000)
        features = self.stem(x)
        context = self.backbone(features)
        return self.head(context)


# === Training ===

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_depth(model):
    """Count sequential depth (conv + linear layers)."""
    return sum(1 for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear)))


def train():
    total_start = time.time()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Data
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    # Model
    model = SleepModel().to(device)
    num_params = count_params(model)
    depth = count_depth(model)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    losses = []
    f1_scores = []
    f1_metric = F1Score(num_classes=NUM_CLASSES, average="macro").to(device)
    num_steps = 0
    total_samples = 0
    train_start = time.time()

    epoch = 0
    while True:
        # Check time budget
        elapsed = time.time() - train_start
        if elapsed >= TIME_BUDGET_SEC:
            break

        model.train()
        total_loss = 0.0
        f1_metric.reset()
        for batch_x, batch_y in train_loader:
            # Check time budget
            elapsed = time.time() - train_start
            if elapsed >= TIME_BUDGET_SEC:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            f1_metric.update(logits.softmax(dim=-1), batch_y)
            num_steps += 1
            total_samples += batch_x.size(0)

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        f1_scores.append(f1_metric.compute().item())

        epoch += 1

    training_seconds = time.time() - train_start

    # Evaluate
    metrics = evaluate(model, val_loader, device)

    total_seconds = time.time() - total_start

    # VRAM
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Print metrics (exact format from program.md)
    print("---")
    print(f"accuracy:          {metrics['accuracy']:.4f}")
    print(f"kappa:             {metrics['kappa']:.4f}")
    print(f"macro_f1:          {metrics['macro_f1']:.4f}")
    for c in range(NUM_CLASSES):
        f1 = metrics[f"class_{c}_f1"]
        p = metrics[f"class_{c}_precision"]
        r = metrics[f"class_{c}_recall"]
        print(f"class_{c}_f1:       {f1:.4f}  (p={p:.3f} r={r:.3f})")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"total_samples_M:   {total_samples / 1e6:.1f}")
    print(f"num_steps:         {num_steps}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"depth:             {depth}")

    # Save loss curve
    if losses:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(losses, linewidth=0.5, alpha=0.7)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")

        ax2.plot(f1_scores, linewidth=0.5, alpha=0.7)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("Training F1 Score")

        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=100)
        plt.close()


if __name__ == "__main__":
    train()
