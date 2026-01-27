"""LSTM xApp (traffic/SINR forecasting) for the Edge Agentic AI prototype.

This package is intentionally self-contained and simulator-friendly:

* Train on the repo's ``Urban5GSimulator`` generated data (CSV or DataFrame)
* Run streaming inference with a sliding-window buffer
* Optionally return uncertainty via Monte-Carlo Dropout
"""

from .xapp import LSTMxApp, LSTMxAppConfig, Prediction

__all__ = ["LSTMxApp", "LSTMxAppConfig", "Prediction"]
