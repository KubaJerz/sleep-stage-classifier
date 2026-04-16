import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a matrix of [max_len, d_model] filled with 0s
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term (the denominator in the formula)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill even indices with sin, odd with cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (it's not a parameter, but stays with the model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Simply add the precomputed encoding to the input
        x = x + self.pe[:, :x.size(1), :]
        return x

class SleepMultiBranchModel(nn.Module):
    def __init__(self, num_classes=5, window_size=5, embed_dim=256, num_heads=8, dropout=0.3):
        super(SleepMultiBranchModel, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_classes = num_classes
        
        # 1. Define the feature extractors
        self.fe400 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=400),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(8, 8),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.AdaptiveAvgPool1d(1)
        )
        self.fe50 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=50, padding=25),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=25),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # (K=100): The model is "zoomed out." It can see the entire cycle of a Delta wave (0.5–4 Hz),
        # which is the hallmark of N3 (Deep Sleep).
        self.fe100 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=100, stride=10), # Maintains length
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=50, padding=25), # Smaller kernel here
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # (K=25): The model is "zoomed in." It can easily see the sharp peaks and fast rhythms of Beta waves 
        # (13–30 Hz) and Alpha waves (8–13 Hz). These are crucial for detecting Wakefulness and REM sleep.
        # At layer 2 (K=16), it sees now represents 4 points from the original signal. So, $16 \times 4 = 64$ points, or 0.64s. 
        # This is the perfect "sweet spot" for Sleep Spindles (which usually last 0.5s to 1.5s).
        self.fe25 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=25, padding=13),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16, padding=8),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, padding=4),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding=2),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fe5 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), # Prevent overfitting early
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fe50_2 = nn.Sequential(
            # Layer 1: Capture broad waveforms (0.5s window)
            nn.Conv1d(1, 64, kernel_size=50, stride=1, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), # Prevent overfitting early
            # Layer 2: Extract patterns (1.0s effective window)
            nn.Conv1d(64, 128, kernel_size=16, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), # Prevent overfitting early
            
            # Layer 3: Deep features (4.0s effective window)
            nn.Conv1d(128, 256, kernel_size=8, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            # Layer 4: Global context
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # The Stability Fix:
        self.feature_projection = nn.Sequential(
            nn.Linear(self.embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout) 
        )
                
        # 2. Multi-Head Attention Layer
        self.pos_embedding = SinusoidalPositionalEncoding(self.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layernorm = nn.LayerNorm(embed_dim)
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, win_size, channels, time)
        B, W, C, T = x.shape
        
        # Extract features for each epoch
        x = x.view(B * W, C, T)
        
        fe400_out = self.fe400(x)
        fe50_out = self.fe50(x)
        fe100_out = self.fe100(x)
        fe25_out = self.fe25(x)
        fe5_out = self.fe5(x)
        
        x = torch.concat([fe400_out, fe50_out, fe100_out, fe25_out, fe5_out], dim=1)
        
        # Reshape to (Batch, Window, Embedding)
        # x = x.view(B, W, self.embed_dim)
        
        # Shape is currently (B*W, 256, 1). Let's fix that:
        x = x.squeeze(-1) # (B*W, 256)
        x = x.view(B, W, -1) # (B, W, 256)
        
        # Project and Normalize before adding Positional Encoding
        x = self.feature_projection(x)
        
        # Postitional Embedding
        x = self.pos_embedding(x)
        
        # Self-Attention
        # attn_output: (Batch, Window, Embedding)
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm(x + attn_output)  # Residual connection
        
        # Focus on the middle epoch (index 2 for window size 5)
        future_context = 2
        target_idx = W - (future_context + 1)
        mid_epoch_features = x[:, target_idx, :]
        
        return self.classifier(mid_epoch_features)