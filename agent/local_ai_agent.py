import ollama
import json
from typing import Dict, Any

class LocalTradingAgent:
    def __init__(self, model="qwen2.5-coder:14b"):
        self.model = model
    
    def analyze_signal(self, quant_signal: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
Kamu adalah AI Trading Assistant. Analisis sinyal dari quant engine ini:

{json.dumps(quant_signal, indent=2)}

Berikan output JSON format:
{{
    "decision": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reason": "alasan singkat"
}}

JANGAN ada teks lain selain JSON.
"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']
            content = content.replace('```json', '').replace('```', '').strip()
            return json.loads(content)
        except Exception as e:
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "reason": f"Error: {str(e)}"
            }

if __name__ == "__main__":
    print("Testing Local AI Agent...")
    agent = LocalTradingAgent()
    test_signal = {
        "signal": "BUY",
        "symbol": "BTC/USDT",
        "price": 65000,
        "quant_score": 0.75
    }
    result = agent.analyze_signal(test_signal)
    print("AI Decision:", result)
8