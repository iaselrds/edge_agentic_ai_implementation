from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .config import TrainingConfig


class TrafficSinrLSTM(nn.Module):
    """Two-stage LSTM forecaster (128 -> 64) with dense head (32 -> 16).

    Architecture mirrors the paper's description.
    Output is a multi-step horizon forecast for (traffic, sinr).
    """

    def __init__(self, input_dim: int, cfg: TrainingConfig, target_dim: int = 2):
        super().__init__()
        self.cfg = cfg
        self.horizon = cfg.horizon
        self.target_dim = target_dim

        self.lstm1 = nn.LSTM(input_dim, cfg.lstm1_hidden, batch_first=True)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.lstm2 = nn.LSTM(cfg.lstm1_hidden, cfg.lstm2_hidden, batch_first=True)
        self.drop2 = nn.Dropout(cfg.dropout)

        self.fc1 = nn.Linear(cfg.lstm2_hidden, cfg.fc1)
        self.fc2 = nn.Linear(cfg.fc1, cfg.fc2)
        self.out = nn.Linear(cfg.fc2, cfg.horizon * target_dim)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lookback, F) -> yhat: (B, horizon, Y)."""
        o1, _ = self.lstm1(x)
        o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1)
        o2 = self.drop2(o2)
        h = o2[:, -1, :]
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        y = self.out(h)
        return y.view(-1, self.horizon, self.target_dim)


@dataclass
class ModelBundle:
    model: TrafficSinrLSTM
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.target_std + self.target_mean


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    dev = device()
    return tuple(t.to(dev) for t in tensors)
