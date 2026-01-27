from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from simulator.net_sim import SimulatorConfig, Urban5GSimulator

from .config import LSTMxAppConfig
from .model import TrafficSinrLSTM, device
from .preprocess import (
    StandardScaler,
    build_feature_frame,
    build_targets,
    make_supervised_sequences,
    train_val_split,
)


def generate_training_df(
    num_days: int = 60,
    base_cfg: Optional[SimulatorConfig] = None,
) -> pd.DataFrame:
    """Generates training data by running the repo's simulator.

    The paper mentions training on long horizons; the repo default is 8 days.
    For an LSTM to learn daily patterns reliably, training on ~30-60 days is
    a decent starting point for this prototype.
    """

    cfg = base_cfg or SimulatorConfig()
    cfg.num_days = int(num_days)
    sim = Urban5GSimulator(cfg)
    sim.run(agent_fn=None)
    return sim.to_dataframe()


def _epoch_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train: bool,
) -> float:
    model.train(train)
    total = 0.0
    n = 0
    dev = device()
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        yhat = model(xb)
        loss = loss_fn(yhat, yb)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu().item())
        n += 1
    return total / max(1, n)


def train_lstm_xapp(
    df: pd.DataFrame,
    cfg: LSTMxAppConfig,
    out_dir: Path,
    seed: int = 7,
) -> Path:
    """Trains the LSTM xApp and writes a loadable artifact directory.

    Output directory layout:
      out_dir/
        model.pt
        scalers.npz
        config.json
    Returns: out_dir
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    feat_df = build_feature_frame(df, cfg.feature_spec)
    tgt_df = build_targets(df)

    X_raw = feat_df.to_numpy(dtype=np.float32)
    y_raw = tgt_df.to_numpy(dtype=np.float32)

    x_scaler = StandardScaler.fit(X_raw)
    y_scaler = StandardScaler.fit(y_raw)

    Xn = x_scaler.transform(X_raw)
    yn = y_scaler.transform(y_raw)

    X, y = make_supervised_sequences(Xn, yn, cfg.training)
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_ratio=0.15)

    dev = device()
    # Keep tensors on CPU; move per-batch (more memory-friendly).
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)
    yva_t = torch.from_numpy(yva)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(Xva_t, yva_t),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=(dev.type == "cuda"),
    )

    model = TrafficSinrLSTM(input_dim=X.shape[-1], cfg=cfg.training, target_dim=y.shape[-1]).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(int(cfg.training.epochs)):
        tr_loss = _epoch_loop(model, train_loader, optimizer, loss_fn, train=True)
        va_loss = _epoch_loop(model, val_loader, optimizer, loss_fn, train=False)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(cfg.training.early_stopping_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez(
        out_dir / "scalers.npz",
        x_mean=x_scaler.mean_,
        x_std=x_scaler.std_,
        y_mean=y_scaler.mean_,
        y_std=y_scaler.std_,
        feature_cols=np.array(list(feat_df.columns), dtype=object),
        target_cols=np.array(list(tgt_df.columns), dtype=object),
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return out_dir


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train the LSTM xApp on simulator data")
    ap.add_argument("--days", type=int, default=60, help="number of simulated days for training")
    ap.add_argument("--out", type=str, default="artifacts/lstm_xapp", help="output artifact dir")
    args = ap.parse_args()

    df = generate_training_df(num_days=args.days)
    out = train_lstm_xapp(df=df, cfg=LSTMxAppConfig(), out_dir=Path(args.out))
    print(f"Saved LSTM xApp artifacts to: {out}")


if __name__ == "__main__":
    main()
