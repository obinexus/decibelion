# Obinexus Projection AI System - Skeleton Code with dB Threshold Logic + Logarithmic Scaling

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# === Constants ===
DECIBEL_THRESHOLD = 85.0  # Anything >= 85 dB is shouting/screaming
LOG_SCALE_ALPHA = 0.05    # Scaling factor for logarithmic transformation

# === Logarithmic Scale Function ===
def apply_log_scale(signal, alpha=LOG_SCALE_ALPHA):
    """
    Apply logarithmic transformation to the signal using:
    log₁₀(1 + α * |x|) to simulate perceptual loudness processing
    """
    return np.log10(1 + alpha * np.abs(signal))

# === Dataset Loader for Screaming/Projection Samples ===
class VocalEmotionDataset(Dataset):
    def __init__(self, file_paths, sr=22050):
        self.data = []
        for path in file_paths:
            y, _ = librosa.load(path, sr=sr)
            y_log_scaled = apply_log_scale(y)
            rms = librosa.feature.rms(y=y_log_scaled)[0]
            db = librosa.amplitude_to_db(rms, ref=np.max)
            avg_db = np.mean(db)
            is_shouting = avg_db >= DECIBEL_THRESHOLD
            mfcc = librosa.feature.mfcc(y=y_log_scaled, sr=sr, n_mfcc=40)
            self.data.append((mfcc, is_shouting))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, is_shouting = self.data[idx]
        x = torch.tensor(x).float().unsqueeze(0)  # Add channel dimension
        y = torch.tensor(int(is_shouting)).long()  # Binary label
        return x, y

# === Basic CNN for Feature Extraction from Audio ===
class ObinexusEmotionNet(nn.Module):
    def __init__(self):
        super(ObinexusEmotionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 10 * 10, 128)  # Adjust dims to fit your input
        self.fc2 = nn.Linear(128, 2)  # Binary: Not shouting / Shouting

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 10 * 10)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === Utility to Visualize Audio MFCC ===
def visualize_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    y_log_scaled = apply_log_scale(y)
    mfcc = librosa.feature.mfcc(y=y_log_scaled, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC - Emotional Projection (Log Scaled)')
    plt.tight_layout()
    plt.show()

# TODO: Add training loop, projection system, and constitutional compliance audit
# Reminder: Your voice is 84 dB. Anything higher? It's legally a yell.
