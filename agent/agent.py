import numpy as np
from lstm_xapp.tool import LSTMForecastTool
from simulator.net_sim import Urban5GSimulator

class ReActAgent:
    def __init__(self, simulator: Urban5GSimulator, lstm_tool: LSTMForecastTool):
        """
        Inicializa o agente ReAct.
        - simulator: Instância do simulador de rede
        - lstm_tool: Instância do modelo LSTM para previsão de tráfego e SINR
        """
        self.simulator = simulator
        self.lstm_tool = lstm_tool
        self.tx_power_dbm = 30.0  # Inicialização da potência de transmissão

    def observe(self):
        """
        Observa o estado atual da rede, coletando métricas do simulador.
        """
        # Coleta das métricas diretamente do simulador
        kpis = {
            'traffic': self.simulator.config.base_traffic,  # Exemplo: Tráfego base configurado no simulador
            'interference_dbm': self.simulator.config.base_interference_dbm,  # Interferência base
            'sinr_db': self.simulator.config.sinr_threshold_db,  # SINR baseado no threshold configurado
            'tx_power_dbm': self.tx_power_dbm  # Potência de transmissão
        }
        return kpis

    def think(self, kpis):
        """
        Processa as observações e gera uma previsão com base no LSTM.
        - kpis: As métricas coletadas da rede.
        """
        forecast = self.lstm_tool.update_and_predict(kpis)
        return forecast

    def act(self, forecast):
        """
        Toma decisões de ação com base na previsão do LSTM.
        Ajusta a potência de transmissão ou outros parâmetros com base nas previsões.
        - forecast: Previsões geradas pelo LSTM para o futuro da rede.
        """
        if forecast:
            final_action = self._adjust_tx_power_based_on_forecast(forecast)
            self.tx_power_dbm = final_action

    def _adjust_tx_power_based_on_forecast(self, forecast):
        """
        Ajusta a potência de transmissão com base nas previsões do LSTM.
        - forecast: Previsão do LSTM (tráfego, SINR, etc.).
        """
        predicted_traffic = forecast.traffic_p50
        predicted_sinr = forecast.sinr_p50
        
        # Lógica simples para ajustar a potência com base nas previsões
        if predicted_traffic > 50:
            return min(46.0, self.tx_power_dbm + 2)  # Aumenta a potência se o tráfego previsto for alto
        elif predicted_sinr < 10:
            return max(30.0, self.tx_power_dbm - 2)  # Diminui a potência se o SINR previsto for baixo
        else:
            return self.tx_power_dbm  # Caso contrário, mantém a potência atual

    def run(self):
        """
        Executa o loop principal de observação, pensamento e ação.
        """
        for t in range(self.simulator.config.duration_minutes):
            # 1. Observação: Coleta as métricas da rede
            kpis = self.observe()

            # 2. Pensamento: Faz a previsão usando o LSTM
            forecast = self.think(kpis)

            # 3. Ação: Ajusta a potência de transmissão ou outros parâmetros com base na previsão
            self.act(forecast)

            # Passa a ação para o simulador
            self.simulator.step(tx_power_dbm=self.tx_power_dbm) # type: ignore

