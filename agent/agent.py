import pandas as pd
import openai

class AgenticReActxApp:
    def __init__(self, config):
        self.config = config
        self.tx_power = 43  
        self.results = []   
    
    # Setp 1. Data collection 
    def collect_data(self, t_min):
        traffic = self.traffic_model(t_min)  
        interference = self.interference_model(traffic, t_min)  
        sinr = self.compute_sinr(interference, t_min)  
        return {
            'traffic': traffic,
            'interference': interference,
            'sinr': sinr
        }
    
    # Step 2 - Intelligent Analysis
    def intelligent_analysis(self, data):
        prompt = f"""
        You are an Outage Mitigator agent for a 5G network. Your goal is to optimize the SINR while minimizing power usage.
        Current data: Traffic: {data['traffic']} Mbps, Interference: {data['interference']} dBm, SINR: {data['sinr']} dB.
        What action should you take to optimize SINR and avoid outage?
        """
        
        # LLM reasoning
        response = openai.Completion.create(
            engine="text-davinci-003",  # Exemplo
            prompt=prompt,
            max_tokens=100
        )
        
        action = response.choices[0].text.strip() 
        return action
    
    # Step 3. Tiered Decision Making
    def tiered_decision(self, sinr):
        if sinr < 15:  # γ1 = 15 dB
            return "Tier 1: Increase power by 3 dB"
        elif 15 <= sinr < 18:  # γ2 = 18 dB
            return "Tier 2: Increase power by 2 dB"
        elif 18 <= sinr < 20:  # γ3 = 20 dB
            return "Tier 3: No change"
        else:
            return "Tier 4: Decrease power by 1 dB"
    
    # Step 4. Action implementation
    def apply_action(self, action):
        if "Increase" in action:
            power_change = int(action.split("by")[1].split("dB")[0].strip())
            self.tx_power = min(46, max(30, self.tx_power + power_change)) 
        elif "Decrease" in action:
            power_change = int(action.split("by")[1].split("dB")[0].strip())
            self.tx_power = max(30, self.tx_power - power_change) 
        return self.tx_power
    
    # Step 5. Performance feedback
    def performance_feedback(self, action, t_min):
        sinr = self.compute_sinr(self.interference_model(self.traffic_model(t_min), t_min), t_min)
        outage = sinr < self.config.sinr_threshold_db
        reward = 10 * (sinr - self.config.sinr_threshold_db) - abs(int(action.split("by")[1].split("dB")[0].strip())) * 2
        
        return sinr, outage, reward
    
    # Run step
    def run_step(self, t_min):
        data = self.collect_data(t_min)  
        action = self.intelligent_analysis(data) 
        tier_action = self.tiered_decision(data['sinr'])
        final_action = tier_action if action == "" else action
        tx_power = self.apply_action(final_action) 
        sinr, outage, reward = self.performance_feedback(final_action, t_min) 
        return sinr, outage, reward, tx_power
    
    # Main loop
    def run(self):
        total_minutes = self.config.num_days * self.config.duration_minutes
        for t in range(total_minutes):
            current_state = self.results[-1] if self.results else {}
            sinr, outage, reward, tx_power = self.run_step(t)  # Execute one timestep
            self.results.append({
                'minute': t % self.config.duration_minutes,
                'day': t // self.config.duration_minutes,
                'traffic': current_state.get('traffic', 0),
                'sinr': sinr,
                'tx_power': tx_power,
                'outage': outage,
                'action': final_action,
                'reward': reward
            })
        
    # Export results to a df
    def to_dataframe(self):
        return pd.DataFrame(self.results)
