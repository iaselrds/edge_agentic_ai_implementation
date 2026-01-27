from pathlib import Path
from simulator.net_sim import SimulatorConfig, Urban5GSimulator
from lstm_xapp.tool import LSTMForecastTool
from agent.agent import  ReActAgent

# Inicializar o simulador e o modelo LSTM
simulator = Urban5GSimulator(SimulatorConfig())
lstm_tool = LSTMForecastTool(Path("artifacts/lstm_xapp"))

# Inicializar o agente ReAct
agent = ReActAgent(simulator, lstm_tool)

# Executar o loop de simulação por um número de minutos
agent.run()
