import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split

from dataset import SleepWindowDataset
from model import SleepMultiBranchModel
from model_trainer import Trainer
from visualization import plot_confussion_matrix, plot_loss_curves, report_scores, plot_accuracy_curves

CLASS_LABELS = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4
}
    
def run_loop(window_size=5):
    data_path = "/home/kasra/courses/2026-edge-computing/project/dataset/sleepEDF/processed/SC"
    all_files = np.array(sorted(glob.glob(os.path.join(data_path, "*.npz"))))
    
    # Define 10-fold cross-validation on files (subjects)
    train_files, val_files = train_test_split(all_files, test_size=30, random_state=42, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device)
    
    score_reports = {'train_loss': [], 'val_loss': [], 'accuracy': [], 'f1': [], 'kappa': []}
    best_preds = None
    best_labels = None

    print(f"\n{'='*20} Training {'='*20}")
    
    # 1. Initialize Datasets (Subject-wise split)
    train_ds = SleepWindowDataset(train_files, window_size=window_size, future_context=2)
    val_ds = SleepWindowDataset(val_files, window_size=window_size, future_context=2)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 2. Setup Model, Loss (Weighted), and Optimizer
    model = SleepMultiBranchModel(num_classes=5).to(device)
    
    # Compute class weights on train data to handle imbalance
    counts = np.bincount(train_ds.Y.numpy())
    weights = 1.0 / torch.tensor(counts, dtype=torch.float32).to(device)
    # Assign weights to classes based on the proportion of the data for that particular class
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    num_epochs = 100
    val_decrease_deadline = 5
    no_convergence = 0
    last_v_loss = np.inf
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # 3. Training/Validation Loop
    trainer = Trainer(model, device) # Uses the Trainer class from previous step
    for epoch in range(num_epochs):  # Adjust epochs as needed
        print("[processing] Epoch: ", epoch + 1)
        t_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        v_loss, v_scores, preds, labels = trainer.validate(val_loader, criterion)
        _, t_scores, t_preds, t_labels = trainer.validate(train_loader, criterion)
        train_losses.append(t_loss)
        train_accs.append(t_scores["accuracy"])
        val_losses.append(v_loss)
        val_accs.append(v_scores["accuracy"])
        scheduler.step()
        
        score_reports['train_loss'].append(t_loss)
        score_reports['val_loss'].append(v_loss)
        print(f"Epoch {epoch+1:02d} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | V-Acc: {v_scores["accuracy"]:.2f} | V-kappa: {v_scores["kappa"]:.2f}\n")
        
        plot_loss_curves(train_losses, val_losses, save_path="results/loss_curves.png")
        plot_accuracy_curves(train_accs, val_accs, save_path="results/accuracy_curves.png")
        
        # Checkpoint: Save if best val_loss
        if v_loss < trainer.best_val_loss:
            trainer.best_val_loss = v_loss
            torch.save(model.state_dict(), f"checkpoints/best_model.pt")
            best_preds = preds
            best_labels = labels
            score_reports['accuracy'].append(v_scores["accuracy"])
            score_reports['f1'].append(v_scores["f1"])
            score_reports['kappa'].append(v_scores["kappa"])
        
        # Early stopping policy: If validation loss doesn't improve in 
        # a certain times (val_decrease_deadline), then terminate the process.
        if v_loss >= last_v_loss:
            no_convergence += 1
            if no_convergence >= val_decrease_deadline:
                break
        else:
            no_convergence = 0
        last_v_loss = v_loss

    # 4. Final Analysis and Plotting5
    plot_confussion_matrix(best_labels, best_preds, CLASS_LABELS, save_path="results/confussion_matrix.png")
    report_scores(best_labels, best_preds, CLASS_LABELS, score_reports, path="results/scores.txt")

if __name__ == "__main__":
    data_path = "/home/kasra/courses/2026-edge-computing/project/dataset/sleepEDF/processed/SC"
    all_files = np.array(sorted(glob.glob(os.path.join(data_path, "*.npz"))))

    tr, te = train_test_split(all_files, test_size=30, random_state=42, shuffle=True)
    ttrain_ds = SleepWindowDataset(tr, window_size=5)
    tval_ds = SleepWindowDataset(te, window_size=5)
    train_loader = DataLoader(ttrain_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(tval_ds, batch_size=128, shuffle=False)
    
    # run trainer
    run_loop(window_size=11)