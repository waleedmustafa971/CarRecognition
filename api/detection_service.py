from core.model_manager import ModelManager
from core.ensemble_voter import EnsembleVoter
from core.model_orchestrator import ModelOrchestrator
import json

class DetectionService:
    def __init__(self):
        with open('config/model_config.json', 'r') as f:
            config = json.load(f)
        
        self.model_manager = ModelManager()
        self.ensemble_voter = EnsembleVoter(
            strategy=config.get('voting_strategy', 'weighted_consensus'),
            consensus_threshold=config.get('consensus_threshold', 0.6)
        )
        self.orchestrator = ModelOrchestrator(self.model_manager, self.ensemble_voter)
    
    def detect_brand(self, vehicle_crop):
        return self.orchestrator.detect_brand(vehicle_crop)

detection_service = DetectionService()

def detect_brand_roboflow(vehicle_crop):
    return detection_service.detect_brand(vehicle_crop)