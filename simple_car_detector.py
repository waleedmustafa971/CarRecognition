# simple_car_detector.py
import cv2
import numpy as np
from pathlib import Path

class SimpleCarDetector:
    def __init__(self):
        """Simple car detection with pattern matching and fallback logic"""
        self.common_brands = [
            'Audi', 'BMW', 'Mercedes-Benz', 'Volkswagen', 'Toyota', 'Honda', 
            'Ford', 'Chevrolet', 'Nissan', 'Hyundai', 'Kia', 'Mazda',
            'Subaru', 'Mitsubishi', 'Lexus', 'Infiniti', 'Acura', 'Volvo',
            'Jaguar', 'Land Rover', 'Porsche', 'Ferrari', 'Lamborghini',
            'Bentley', 'Rolls-Royce', 'Maserati', 'Alfa Romeo', 'FIAT',
            'Peugeot', 'Citroen', 'Renault', 'Skoda', 'SEAT'
        ]
        
        # Simple heuristics based on visual features
        self.brand_features = {
            'Audi': {'grille_pattern': 'hexagonal', 'rings': 4, 'style': 'modern'},
            'BMW': {'grille_pattern': 'kidney', 'roundels': True, 'style': 'sporty'},
            'Mercedes-Benz': {'grille_pattern': 'star', 'logo': 'three-pointed-star', 'style': 'luxury'},
            'Volkswagen': {'grille_pattern': 'simple', 'logo': 'VW', 'style': 'clean'},
            'Toyota': {'grille_pattern': 'wide', 'logo': 'oval', 'style': 'reliable'},
            'Honda': {'grille_pattern': 'horizontal', 'logo': 'H', 'style': 'practical'},
            'Lamborghini': {'style': 'aggressive', 'lines': 'angular', 'stance': 'low'},
            'Ferrari': {'style': 'curved', 'stance': 'low', 'color_preference': 'red'},
            'Porsche': {'headlights': 'round', 'stance': 'sporty', 'style': 'classic'}
        }
    
    def analyze_vehicle_features(self, vehicle_crop):
        """Analyze visual features of the vehicle"""
        try:
            height, width = vehicle_crop.shape[:2]
            aspect_ratio = width / height
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            
            features = {
                'aspect_ratio': aspect_ratio,
                'is_sports_car': aspect_ratio > 2.0,  # Wide, low profile
                'is_luxury': False,  # To be determined by other features
                'dominant_color': self.get_dominant_color(hsv),
                'has_aggressive_lines': self.detect_aggressive_styling(vehicle_crop),
                'front_end_style': self.analyze_front_end(vehicle_crop)
            }
            
            return features
            
        except Exception as e:
            print(f"Feature analysis error: {e}")
            return {}
    
    def get_dominant_color(self, hsv_image):
        """Get the dominant color of the vehicle"""
        # Simple color detection
        colors = {
            'red': [(0, 100, 100), (10, 255, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'white': [(0, 0, 200), (180, 25, 255)],
            'silver': [(0, 0, 120), (180, 30, 200)],
            'blue': [(100, 100, 100), (130, 255, 255)]
        }
        
        max_pixels = 0
        dominant_color = 'unknown'
        
        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            pixel_count = cv2.countNonZero(mask)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color_name
        
        return dominant_color
    
    def detect_aggressive_styling(self, vehicle_crop):
        """Detect aggressive/sporty styling"""
        # Look for sharp angles and angular lines
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Count strong edges - more edges might indicate aggressive styling
        edge_count = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_ratio = edge_count / total_pixels
        
        return edge_ratio > 0.1  # Threshold for "aggressive" styling
    
    def analyze_front_end(self, vehicle_crop):
        """Analyze front-end characteristics"""
        # This is a simplified analysis
        # In reality, you'd use more sophisticated image processing
        height, width = vehicle_crop.shape[:2]
        
        # Look at the bottom third of the image (likely front bumper area)
        front_section = vehicle_crop[int(height*0.7):, :]
        
        # Simple edge detection in front section
        gray_front = cv2.cvtColor(front_section, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_front, 50, 150)
        
        # Analyze edge patterns (simplified)
        horizontal_edges = np.sum(edges, axis=1)
        vertical_edges = np.sum(edges, axis=0)
        
        return {
            'complexity': np.std(horizontal_edges),
            'symmetry': self.calculate_symmetry(vertical_edges)
        }
    
    def calculate_symmetry(self, vertical_edges):
        """Calculate how symmetric the front end is"""
        mid_point = len(vertical_edges) // 2
        left_half = vertical_edges[:mid_point]
        right_half = vertical_edges[mid_point:][::-1]  # Reverse right half
        
        min_len = min(len(left_half), len(right_half))
        if min_len == 0:
            return 0
        
        left_half = left_half[:min_len]
        right_half = right_half[:min_len]
        
        # Calculate correlation between left and right halves
        correlation = np.corrcoef(left_half, right_half)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def predict_brand_from_features(self, features):
        """Predict brand based on analyzed features"""
        predictions = []
        
        # Rule-based predictions
        if features.get('is_sports_car') and features.get('has_aggressive_lines'):
            if features.get('dominant_color') == 'red':
                predictions.append({"make": "Ferrari", "model": "Unknown", "score": 0.6})
            predictions.append({"make": "Lamborghini", "model": "Unknown", "score": 0.5})
            predictions.append({"make": "Porsche", "model": "Unknown", "score": 0.4})
        
        elif features.get('aspect_ratio', 0) > 1.8:  # Wide vehicle
            predictions.extend([
                {"make": "Audi", "model": "Unknown", "score": 0.4},
                {"make": "BMW", "model": "Unknown", "score": 0.4},
                {"make": "Mercedes-Benz", "model": "Unknown", "score": 0.3}
            ])
        
        else:  # Regular proportions
            predictions.extend([
                {"make": "Toyota", "model": "Unknown", "score": 0.3},
                {"make": "Honda", "model": "Unknown", "score": 0.3},
                {"make": "Volkswagen", "model": "Unknown", "score": 0.3}
            ])
        
        # If no specific predictions, return common brands
        if not predictions:
            predictions = [
                {"make": "Unknown", "model": "Unknown", "score": 0.1}
            ]
        
        return sorted(predictions, key=lambda x: x['score'], reverse=True)[:3]
    
    def detect_brand_simple(self, vehicle_crop):
        """Main detection method using simple feature analysis"""
        try:
            print("Using simple feature-based car detection...")
            
            # Analyze vehicle features
            features = self.analyze_vehicle_features(vehicle_crop)
            print(f"Detected features: {features}")
            
            # Predict brand based on features
            predictions = self.predict_brand_from_features(features)
            
            return predictions
            
        except Exception as e:
            print(f"Simple detection error: {e}")
            return [{"make": "Unknown", "model": "Unknown", "score": 0.0}]

if __name__ == "__main__":
    # Test the simple detector
    detector = SimpleCarDetector()
    
    # Test with a sample image
    test_image = cv2.imread("data/test_car.jpg")
    if test_image is not None:
        result = detector.detect_brand_simple(test_image)
        print("Simple detection result:")
        for i, brand in enumerate(result):
            print(f"  {i+1}. {brand['make']} {brand['model']} (confidence: {brand['score']:.3f})")
    else:
        print("Test image not found")