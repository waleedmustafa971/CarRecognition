import logging
from datetime import datetime
from collections import defaultdict
import json

class ModelMetrics:
    def __init__(self):
        self.predictions = defaultdict(list)
        self.model_usage = defaultdict(int)
    
    def log_prediction(self, brand: str, confidence: float, models_used: List[str]):
        self.predictions[brand].append({
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "models": models_used
        })
        
        for model in models_used:
            self.model_usage[model] += 1
    
    def get_stats(self):
        return {
            "total_predictions": sum(len(v) for v in self.predictions.values()),
            "brands_detected": list(self.predictions.keys()),
            "model_usage": dict(self.model_usage)
        }
    
    def export_to_file(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump({
                "predictions": dict(self.predictions),
                "model_usage": dict(self.model_usage),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)