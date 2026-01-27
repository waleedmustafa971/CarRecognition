import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ModelOrchestrator:
    def __init__(self, model_manager, ensemble_voter):
        self.model_manager = model_manager
        self.ensemble_voter = ensemble_voter
    
    def detect_brand(self, vehicle_crop) -> List[Dict]:
        all_detections = []
        
        specialist_models = self.model_manager.get_specialist_models()
        for name, model in specialist_models.items():
            config = self.model_manager.get_model_config(name)
            detections = self._run_model(model, vehicle_crop, name, config)
            all_detections.extend(detections)
        
        general_models = self.model_manager.get_general_models()
        for name, model in general_models.items():
            config = self.model_manager.get_model_config(name)
            detections = self._run_model(model, vehicle_crop, name, config)
            all_detections.extend(detections)
        
        all_detections = [d for d in all_detections if d['score'] >= d.get('min_confidence', 0.3)]
        
        if not all_detections:
            logger.warning("No valid detections found")
            return [{"make": "Unknown", "model": "", "score": 0.0}]
        
        result = self.ensemble_voter.vote(all_detections)
        
        return [result] if result else [{"make": "Unknown", "model": "", "score": 0.0}]
    
    def _run_model(self, model, vehicle_crop, model_name: str, config) -> List[Dict]:
        detections = []
        
        try:
            results = model(vehicle_crop, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        
                        if confidence < config.min_confidence:
                            continue
                        
                        if config.tier == 'specialist':
                            make = config.brands[0] if config.brands else "Unknown"
                            model_detail = class_name if class_name not in ['0', '1', 'car', 'vehicle', 'object'] else ""
                        else:
                            make = class_name.split()[0] if class_name else "Unknown"
                            model_detail = " ".join(class_name.split()[1:]) if len(class_name.split()) > 1 else ""
                        
                        if config.only_for_brands and make.lower() not in [b.lower() for b in config.only_for_brands]:
                            continue
                        
                        logger.info(f"[{config.tier.upper()}] {model_name}: {make} (conf: {confidence:.3f})")
                        
                        detections.append({
                            "make": make.capitalize(),
                            "model": model_detail,
                            "score": confidence,
                            "source": model_name,
                            "tier": config.tier,
                            "min_confidence": config.min_confidence
                        })
        
        except Exception as e:
            logger.error(f"Error running {model_name}: {e}")
        
        return detections