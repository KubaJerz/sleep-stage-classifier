import numpy as np
import torch
from torch.utils.data import Dataset

class SleepWindowDataset(Dataset):
    def __init__(self, file_paths, window_size=5, future_context=2, trim_margin_min=30 ):
        """
        window_size: must be odd (e.g., 5, 9, 21)
        """
        assert window_size % 2 != 0, "Window size must be an odd number."
        self.window_size = window_size
        self.future_context = future_context
        self.current_idx = window_size - future_context - 1
        self.past_context = self.current_idx
        
        self.features = []
        self.labels = []
        
        margin = (trim_margin_min * 60) // 30
        
        for path in file_paths:
            with np.load(path) as data:
                x, y = data['x'], data['y']
                
                # 1. Trimming based on sleep activity
                sleep_idx = np.where(y != 0)[0]
                if len(sleep_idx) == 0: continue
                
                start = max(0, sleep_idx[0] - margin)
                end = min(len(y), sleep_idx[-1] + margin + 1)
                x = x[start:end]
                y = y[start:end]
                
                # 2. Subject-level Z-score Normalization
                x = (x - np.mean(x)) / (np.std(x) + 1e-8)
                
                # 3. Sliding Centered Window
                # We start from half_win and end at len - half_win
                for i in range(self.past_context, len(x) - self.future_context):
                    window = x[i - self.past_context : i + self.future_context + 1]
                    self.features.append(window)
                    self.labels.append(y[i])
                    
                self.features.append(window)
                self.labels.append(y[i]) # Label of the middle epoch

        # Stack into tensors
        self.X = torch.from_numpy(np.array(self.features)).float()
        self.Y = torch.from_numpy(np.array(self.labels)).long()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]