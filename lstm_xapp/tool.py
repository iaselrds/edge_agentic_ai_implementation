from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .xapp import LSTMxApp, Prediction


@dataclass
class LSTMToolResult:
    """Compact object to be embedded in an agent prompt/memory."""

    horizon: int
    traffic_p50: float
    traffic_p95: float
    sinr_p50: float
    sinr_p05: float


class LSTMForecastTool:
    """Thin wrapper that makes the LSTM xApp behave like an 'agent tool'."""

    def __init__(self, artifact_dir: Path):
        self.xapp = LSTMxApp.load(artifact_dir)

    def update_and_predict(self, kpis: Dict[str, float]) -> Optional[LSTMToolResult]:
        self.xapp.update(kpis)
        pred: Optional[Prediction] = self.xapp.predict()
        if pred is None:
            return None

        # Targets: [traffic, sinr_db] by default.
        # Use the last point in the horizon as a simple 'risk' summary.
        last = -1
        idx_traffic = pred.target_cols.index("traffic") if "traffic" in pred.target_cols else 0
        idx_sinr = pred.target_cols.index("sinr_db") if "sinr_db" in pred.target_cols else 1

        traffic_p50 = float(pred.mean[last, idx_traffic])
        traffic_p95 = float(pred.hi[last, idx_traffic])
        sinr_p50 = float(pred.mean[last, idx_sinr])
        sinr_p05 = float(pred.lo[last, idx_sinr])

        return LSTMToolResult(
            horizon=pred.mean.shape[0],
            traffic_p50=traffic_p50,
            traffic_p95=traffic_p95,
            sinr_p50=sinr_p50,
            sinr_p05=sinr_p05,
        )
