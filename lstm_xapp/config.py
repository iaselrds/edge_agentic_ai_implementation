from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    """Defines which features the LSTM consumes.

    The default set is compatible with the repo simulator and the paper's
    "contextual features" idea (time-of-day and event indicators).
    """

    # Raw KPI-like features (from simulator / E2)
    use_traffic: bool = True
    use_interference: bool = True
    use_sinr: bool = True
    use_tx_power: bool = True

    # Context features (cheap proxies for external context in this prototype)
    use_time_sincos: bool = True
    use_is_event: bool = True
    use_is_business_hour: bool = True


@dataclass
class TrainingConfig:
    # Sequence setup
    lookback: int = 60          # minutes in the past
    horizon: int = 30           # minutes to predict ahead

    # Model (matches the paper: 2 LSTM layers 128/64, then dense 32/16)
    lstm1_hidden: int = 128
    lstm2_hidden: int = 64
    fc1: int = 32
    fc2: int = 16
    dropout: float = 0.2

    # Training hyper-params (paper uses 500 epochs, batch 32, lr 1e-3)
    epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-3
    early_stopping_patience: int = 10
    weight_decay: float = 0.0


@dataclass
class LSTMxAppConfig:
    """Runtime config for the streaming xApp."""

    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Uncertainty estimation
    mc_dropout_samples: int = 30
    confidence_z: float = 1.96  # ~95% interval
