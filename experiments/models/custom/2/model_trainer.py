import torch
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, recall_score, precision_score

class Trainer:
    def __init__(self, model, device, fold_idx=None):
        self.model = model.to(device)
        self.device = device
        self.fold_idx = fold_idx if fold_idx else 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        avg_loss = total_loss / len(loader)
        scores = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average='macro'),
            "precision": precision_score(all_labels, all_preds, average='macro'),
            "recall": recall_score(all_labels, all_preds, average='macro'),
            "kappa": cohen_kappa_score(all_labels, all_preds)
        }
        return avg_loss, scores, all_preds, all_labels

    def save_checkpoint(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), f"checkpoints/best_model_fold_{self.fold_idx}.pt")