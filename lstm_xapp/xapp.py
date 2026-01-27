from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore

from .config import LSTMxAppConfig
from .model import TrafficSinrLSTM, device
from .preprocess import build_feature_frame


@dataclass
class Prediction:
    """Forecast output.

    Arrays are shaped:
      mean: (horizon, target_dim)
      std:  (horizon, target_dim)
      lo/hi: (horizon, target_dim)
    """

    mean: np.ndarray
    std: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    target_cols: List[str]


class LSTMxApp:
    """Streaming LSTM xApp.

    Usage:
      xapp = LSTMxApp.load(Path("artifacts/lstm_xapp"))
      xapp.update(kpis={"minute": t, "traffic": ..., "interference_dbm": ..., "tx_power_dbm": ..., "sinr_db": ...})
      pred = xapp.predict()  # None until enough lookback samples
    """

    def __init__(
        self,
        cfg: LSTMxAppConfig,
        model: "TrafficSinrLSTM",
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        feature_cols: List[str],
        target_cols: List[str],
    ):
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for LSTMxApp. Install torch, or run this component on a machine with torch available."
            )

        self.cfg = cfg
        self.model = model
        self.model.eval()

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)

        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)

        self.lookback = int(cfg.training.lookback)
        self.horizon = int(cfg.training.horizon)
        self._buffer: List[Dict[str, float]] = []

    @staticmethod
    def load(artifact_dir: Path, cfg: Optional[LSTMxAppConfig] = None) -> "LSTMxApp":
        if torch is None:
            raise RuntimeError("PyTorch is required to load the LSTMxApp artifacts.")

        cfg = cfg or LSTMxAppConfig()
        artifact_dir = Path(artifact_dir)

        scalers = np.load(artifact_dir / "scalers.npz", allow_pickle=True)
        x_mean = scalers["x_mean"]
        x_std = scalers["x_std"]
        y_mean = scalers["y_mean"]
        y_std = scalers["y_std"]
        feature_cols = list(scalers["feature_cols"].tolist())
        target_cols = list(scalers["target_cols"].tolist())

        model = TrafficSinrLSTM(input_dim=len(feature_cols), cfg=cfg.training, target_dim=len(target_cols))
        model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location=device()))
        model.to(device())
        model.eval()
        return LSTMxApp(cfg, model, x_mean, x_std, y_mean, y_std, feature_cols, target_cols)

    def reset(self) -> None:
        self._buffer = []

    def update(self, kpis: Dict[str, float]) -> None:
        """Pushes one timestep of KPI data into the xApp buffer.

        Required keys (prototype): minute, traffic, interference_dbm, tx_power_dbm, sinr_db.
        You can pass additional keys; they are ignored.
        """

        # Keep only what the feature builder needs.
        self._buffer.append(
            {
                "minute": float(kpis["minute"]),
                "traffic": float(kpis.get("traffic", 0.0)),
                "interference_dbm": float(kpis.get("interference_dbm", kpis.get("interference", 0.0))),
                "tx_power_dbm": float(kpis.get("tx_power_dbm", kpis.get("tx_power", 0.0))),
                "sinr_db": float(kpis.get("sinr_db", kpis.get("sinr", 0.0))),
            }
        )
        if len(self._buffer) > self.lookback:
            self._buffer = self._buffer[-self.lookback :]

    def _build_x_window(self) -> np.ndarray:
        df = pd.DataFrame(self._buffer)
        feat_df = build_feature_frame(df, self.cfg.feature_spec)

        # Ensure same feature order as training
        feat_df = feat_df[self.feature_cols]
        x = feat_df.to_numpy(dtype=np.float32)
        x = (x - self.x_mean) / self.x_std
        return x

    def predict(self) -> Optional[Prediction]:
        """Returns a forecast if enough history exists, otherwise None."""

        if len(self._buffer) < self.lookback:
            return None

        x = self._build_x_window()  # (lookback, F)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device())

        # Monte-Carlo dropout for uncertainty
        samples = int(self.cfg.mc_dropout_samples)
        if samples <= 1:
            with torch.no_grad():
                yhat = self.model(x_t)
            yhat = yhat.squeeze(0).cpu().numpy()
            mean = yhat
            std = np.zeros_like(mean)
        else:
            preds = []
            self.model.train(True)  # enable dropout
            with torch.no_grad():
                for _ in range(samples):
                    preds.append(self.model(x_t).squeeze(0).cpu().numpy())
            self.model.train(False)
            stack = np.stack(preds, axis=0)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)

        # denormalize
        mean = mean * self.y_std + self.y_mean
        std = std * self.y_std

        z = float(self.cfg.confidence_z)
        lo = mean - z * std
        hi = mean + z * std

        return Prediction(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            lo=lo.astype(np.float32),
            hi=hi.astype(np.float32),
            target_cols=list(self.target_cols),
        )
