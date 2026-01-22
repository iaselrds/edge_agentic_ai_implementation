import numpy as np
import pandas as pd
from dataclasses import dataclass
import random

@dataclass
class SimulatorConfig:
    duration_minutes: int = 1440  # one full day
    base_traffic: float = 100.0   # Mbps
    peak_multiplier: float = 1.5
    event_multiplier: float = 5.0
    noise_floor_dbm: float = -104.0
    base_interference_dbm: float = -110.0
    tx_power_dbm: float = 43.0
    sinr_threshold_db: float = 15.0
    pathloss_db: float = 120.0  # static pathloss for now
    rain_loss_db: float = 0.0

class Urban5GSimulator:
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.tx_power = config.tx_power_dbm
        self.results = []

    def gaussian_peak(self, t_min, center_min, std_dev):
        return np.exp(-0.5 * ((t_min - center_min) / std_dev)**2)

    def is_event_time(self, t_min):
        # Example: event from 1080 to 1260 minutes (18:00 to 21:00)
        return 1080 <= t_min <= 1260

    def is_business_hour(self, t_min):
        hour = (t_min // 60) % 24
        return 9 <= hour <= 17

    def traffic_model(self, t_min):
        morning_peak = self.gaussian_peak(t_min, 480, 45)  # 8:00 AM
        evening_peak = self.gaussian_peak(t_min, 1110, 60) # 6:30 PM
        traffic = self.config.base_traffic * (1 + self.config.peak_multiplier * (morning_peak + evening_peak))
        if self.is_event_time(t_min):
            traffic *= self.config.event_multiplier
        return traffic

    def interference_model(self, traffic, t_min):
        cochannel = 10 * (traffic / self.config.base_traffic)
        industrial = 3.0 if self.is_business_hour(t_min) else 0.0
        crowd = random.uniform(2, 8) if self.is_event_time(t_min) else 0.0
        noise = np.random.normal(0, 2)
        return self.config.base_interference_dbm + cochannel + industrial + crowd + noise

    def channel_loss(self, t_min):
        shadowing = np.random.normal(0, 4)  # log-normal shadowing
        rain_fade = self.config.rain_loss_db if self.is_event_time(t_min) else 0.0
        return self.config.pathloss_db + shadowing + rain_fade

    def compute_sinr(self, interference_dbm, t_min):
        pathloss = self.channel_loss(t_min)
        prx = self.tx_power - pathloss
        interference_power = 10 ** (interference_dbm / 10)
        noise_power = 10 ** (self.config.noise_floor_dbm / 10)
        sinr_linear = 10 ** ((prx) / 10) / (interference_power + noise_power)
        return 10 * np.log10(sinr_linear)

    def step(self, t_min, agent_action_db=0):
        self.tx_power = min(46, max(30, self.tx_power + agent_action_db))
        traffic = self.traffic_model(t_min)
        interference = self.interference_model(traffic, t_min)
        sinr = self.compute_sinr(interference, t_min)
        outage = sinr < self.config.sinr_threshold_db

        self.results.append({
            "minute": t_min,
            "traffic": traffic,
            "interference_dbm": interference,
            "tx_power_dbm": self.tx_power,
            "sinr_db": sinr,
            "outage": outage,
            "action_db": agent_action_db
        })

    def run(self, agent_fn=None):
        for t in range(self.config.duration_minutes):
            current_state = self.results[-1] if self.results else {}
            action = agent_fn(current_state) if agent_fn else 0
            self.step(t, action)

    def to_dataframe(self):
        return pd.DataFrame(self.results)

if __name__ == "__main__":
    sim = Urban5GSimulator(SimulatorConfig())
    sim.run(agent_fn=lambda state: 0)  # no agent, static power
    df = sim.to_dataframe()
    print(df.head())
