import torch
from torch.utils.data import Dataset
import numpy as np

class SleepDataset(Dataset):
    def __init__(self, file_paths: list, window_size=1, trimmed: bool=True):
        self.file_paths = file_paths
        self.window_size = window_size
        self.trimmed = trimmed
        self.data_indeces = self._build_trimmed_index() if trimmed else self._build_index()
        
    def _build_index(self):
        # Maps a global index to (file_path, local_epoch_index)
        index = []
        for path in self.file_paths:
            with np.load(path) as data:
                num_epochs = len(data['y'])
                for i in range(num_epochs - self.window_size + 1):
                    index.append((path, i))
        return index
    
    def _build_trimmed_index(self):
        index = []
        for path in self.file_paths:
            with np.load(path, mmap_mode='r') as data:
                labels = data['y']
                
                # Find the true start (first sleep) and true end (last sleep)
                sleep_mask = np.where(labels != 0)[0]
                if len(sleep_mask) == 0: continue
                
                first_sleep = sleep_mask[0]
                last_sleep = sleep_mask[-1]
                
                # Apply the 30-minute buffer (60 epochs)
                start_search = max(0, first_sleep - 60)
                end_search = min(len(labels), last_sleep + 60)
                
                # Add ALL indices in this range, including the 0s in the middle
                for i in range(start_search, end_search - self.window_size + 1):
                    index.append((path, i))
        return index
    
    def __len__(self):
        return len(self.data_indeces)
    
    def __getitem__(self, idx):
        file_path, start_pos = self.data_indeces[idx]
        with np.load(file_path, mmap_mode='r') as data:
            x = data['x'][start_pos : start_pos + self.window_size]
            mid_idx = start_pos + (self.window_size // 2)
            y = data['y'][mid_idx]
            
            print(f"x: ", x)
            print(f"y: ", y)
            
            # Z-score Normalization (per-subject/per-file)
            x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        return torch.from_numpy(x).copy_().float(), torch.tensor(y).long()


NUM_CLASSES = 5


class SleepDataset2(Dataset):
    def __init__(self, npz_path, context=1):
        data = np.load(npz_path)
        self.x = torch.from_numpy(data["x"].astype(np.float32))
        self.y = torch.from_numpy(data["y"].astype(np.int64))
        self.context = context

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.context == 1:
            return self.x[idx], self.y[idx]
        # causal context: use idx-context+1 .. idx, label = y[idx]
        K = self.context
        start = idx - K + 1
        if start < 0:
            # left-pad by repeating the first available epoch
            pad = self.x[0:1].expand(-1 * start, *self.x.shape[1:]) if False else self.x[0].unsqueeze(0).repeat(-start, 1, 1)
            window = torch.cat([pad, self.x[0:idx + 1]], dim=0)
        else:
            window = self.x[start:idx + 1]
        return window, self.y[idx]