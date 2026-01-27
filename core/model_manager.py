import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from ultralytics import YOLO
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    path: str
    brands: List[str]
    min_confidence: float
    tier: str
    only_for_brands: Optional[List[str]] = None
    is_primary: bool = False
    is_backup: bool = False

class ModelManager:
    def __init__(self, config_path: str = "config/model_config.json"):
        self.config_path = config_path
        self.models: Dict[str, YOLO] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.load_config()
        self.initialize_models()
    
    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        for tier_config in config['model_hierarchy']:
            tier = tier_config['tier']
            for model_data in tier_config['models']:
                model_config = ModelConfig(
                    name=model_data['name'],
                    path=model_data['path'],
                    brands=model_data['brands'],
                    min_confidence=model_data['min_confidence'],
                    tier=tier,
                    only_for_brands=model_data.get('only_for_brands'),
                    is_primary=model_data.get('is_primary', False),
                    is_backup=model_data.get('is_backup', False)
                )
                self.model_configs[model_data['name']] = model_config
        
        logger.info(f"Loaded {len(self.model_configs)} model configurations")
    
    def initialize_models(self):
        for name, config in self.model_configs.items():
            if Path(config.path).exists():
                try:
                    self.models[name] = YOLO(config.path)
                    logger.info(f"✓ Loaded {name} model from {config.path}")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
            else:
                logger.warning(f"Model not found: {config.path}")
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def get_models_by_tier(self, tier: str) -> Dict[str, YOLO]:
        return {
            name: model 
            for name, model in self.models.items() 
            if self.model_configs[name].tier == tier
        }
    
    def get_specialist_models(self) -> Dict[str, YOLO]:
        return self.get_models_by_tier("specialist")
    
    def get_general_models(self) -> Dict[str, YOLO]:
        return self.get_models_by_tier("general")
    
    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        return self.model_configs.get(name)