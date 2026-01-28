import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PeriodicHead(nn.Module):
    def __init__(self, in_dim, T=365, K=16):
        super().__init__()
        self.T = T
        self.K = K
        self.fc = nn.Linear(in_dim, 1 + 2 * K)

        t = torch.arange(T, dtype=torch.float32) / T
        self.register_buffer("cos", torch.stack([torch.cos(2*np.pi*k*t) for k in range(1, K+1)]))
        self.register_buffer("sin", torch.stack([torch.sin(2*np.pi*k*t) for k in range(1, K+1)]))

    def forward(self, h):
        coeff = self.fc(h)
        a0 = coeff[:, :1]
        ak = coeff[:, 1:1+self.K]
        bk = coeff[:, 1+self.K:]
        return a0 + ak @ self.cos + bk @ self.sin


class CNNGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, seq_len=365):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, (3,3), padding=1)
        self.pool1 = nn.MaxPool2d((2,1))
        self.conv2 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.pool2 = nn.MaxPool2d((2,1))

        self.gru = nn.GRU(
            input_size=64 * input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.head = PeriodicHead(2 * hidden_dim, T=seq_len)

    def forward(self, x):
        B, T, C = x.shape
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1, 3).reshape(B, x.shape[2], -1)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.head(h)