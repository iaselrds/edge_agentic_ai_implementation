"""Demo: LSTM xApp Integration with the repo simulator.

This is NOT a full O-RAN xApp. It is a prototype that mirrors the paper's
pipeline:

Stage 1: contextual feature aggregation (time-of-day / event proxy)
Stage 2: LSTM forecasting (traffic + SINR horizon)
Stage 3: agent uses forecasts to choose proactive actions

Run:
  python -m lstm_xapp.train --days 60 --out artifacts/lstm_xapp
  python examples/lstm_integration_demo.py
"""

from pathlib import Path

from simulator.net_sim import SimulatorConfig, Urban5GSimulator

from lstm_xapp.tool import LSTMForecastTool


def main() -> None:
    artifacts = Path("artifacts/lstm_xapp")
    tool = LSTMForecastTool(artifacts)

    sim = Urban5GSimulator(SimulatorConfig(num_days=2))

    def agent_fn(state: dict) -> int:
        # Get current KPI view (state may be empty on first tick)
        t = int(state.get("minute", 0))
        kpis = {
            "minute": t,
            "traffic": float(state.get("traffic", 0.0)),
            "interference_dbm": float(state.get("interference_dbm", state.get("interference", -110.0))),
            "tx_power_dbm": float(state.get("tx_power_dbm", state.get("tx_power", 43.0))),
            "sinr_db": float(state.get("sinr_db", state.get("sinr", 25.0))),
        }

        forecast = tool.update_and_predict(kpis)
        if forecast is None:
            return 0

        # Simple proactive heuristic:
        # - if forecast suggests SINR might drop below 15 dB, pre-boost power
        # - if forecast suggests SINR is safely above 20 dB, reduce power for efficiency
        if forecast.sinr_p05 < 15.0:
            return +2
        if forecast.sinr_p50 > 20.0 and forecast.traffic_p50 < forecast.traffic_p95:
            return -1
        return 0

    sim.run(agent_fn=agent_fn)
    df = sim.to_dataframe()
    print(df[["minute", "traffic", "sinr_db", "tx_power_dbm", "outage", "action_db"]].tail(10))


if __name__ == "__main__":
    main()
