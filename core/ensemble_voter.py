import logging
from typing import List, Dict, Optional
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class EnsembleVoter:
    def __init__(self, strategy: str = "weighted_consensus", consensus_threshold: float = 0.6):
        self.strategy = strategy
        self.consensus_threshold = consensus_threshold
    
    def vote(self, detections: List[Dict]) -> Optional[Dict]:
        if not detections:
            return None
        
        if self.strategy == "weighted_consensus":
            return self._weighted_consensus(detections)
        elif self.strategy == "highest_confidence":
            return self._highest_confidence(detections)
        elif self.strategy == "majority_vote":
            return self._majority_vote(detections)
        else:
            return self._weighted_consensus(detections)
    
    def _weighted_consensus(self, detections: List[Dict]) -> Dict:
        brand_votes = defaultdict(lambda: {"total_confidence": 0.0, "count": 0, "detections": []})
        
        for detection in detections:
            brand = detection['make'].lower()
            confidence = detection['score']
            tier = detection.get('tier', 'general')
            
            weight = 1.5 if tier == 'specialist' else 1.0
            
            brand_votes[brand]['total_confidence'] += confidence * weight
            brand_votes[brand]['count'] += 1
            brand_votes[brand]['detections'].append(detection)
        
        best_brand = None
        best_score = 0
        
        for brand, data in brand_votes.items():
            avg_confidence = data['total_confidence'] / data['count']
            
            if data['count'] >= 2:
                avg_confidence *= 1.2
            
            logger.info(f"Brand: {brand.capitalize()}, Votes: {data['count']}, Weighted Score: {avg_confidence:.3f}")
            
            if avg_confidence > best_score:
                best_score = avg_confidence
                best_brand = brand
        
        if best_brand:
            winner = brand_votes[best_brand]['detections'][0]
            winner['score'] = min(0.95, best_score)
            winner['consensus'] = True
            winner['vote_count'] = brand_votes[best_brand]['count']
            logger.info(f"✓ CONSENSUS WINNER: {winner['make']} (score: {winner['score']:.3f}, votes: {winner['vote_count']})")
            return winner
        
        return detections[0]
    
    def _highest_confidence(self, detections: List[Dict]) -> Dict:
        return max(detections, key=lambda x: x['score'])
    
    def _majority_vote(self, detections: List[Dict]) -> Dict:
        brands = [d['make'].lower() for d in detections]
        most_common = Counter(brands).most_common(1)[0]
        
        for detection in detections:
            if detection['make'].lower() == most_common[0]:
                return detection
        
        return detections[0]