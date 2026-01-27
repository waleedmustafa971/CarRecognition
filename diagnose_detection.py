import sys
import os
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def test_vehicle_detection(image_path):
    print(f"\n{'='*80}")
    print("DIAGNOSTIC TEST - VEHICLE DETECTION")
    print(f"{'='*80}")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    print(f"\n1. Testing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Failed to load image")
        return
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    print(f"\n2. Loading YOLO vehicle detection model...")
    yolo_model = YOLO('yolov8n.pt')
    print("   ✓ YOLO loaded")
    
    print(f"\n3. Testing vehicle detection with different thresholds:")
    
    for conf_threshold in [0.5, 0.35, 0.25, 0.15, 0.1, 0.05]:
        print(f"\n   Threshold: {conf_threshold}")
        results = yolo_model(image, verbose=False, classes=[2, 5, 7], conf=conf_threshold)
        
        vehicle_count = 0
        for detection in results:
            if detection.boxes is not None:
                for box in detection.boxes:
                    confidence = float(box.conf[0])
                    if confidence >= conf_threshold:
                        vehicle_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1
                        print(f"      Vehicle {vehicle_count}: {w}x{h}px (conf: {confidence:.3f})")
        
        if vehicle_count == 0:
            print(f"      ⚠️  NO VEHICLES DETECTED")
        else:
            print(f"      ✓ Found {vehicle_count} vehicles")
    
    print(f"\n4. Testing brand classification models:")
    
    results = yolo_model(image, verbose=False, classes=[2, 5, 7], conf=0.15)
    
    vehicle_found = False
    for detection in results:
        if detection.boxes is not None and len(detection.boxes) > 0:
            box = detection.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_crop = image[y1:y2, x1:x2].copy()
            vehicle_found = True
            
            print(f"\n   Testing with vehicle crop: {vehicle_crop.shape[1]}x{vehicle_crop.shape[0]}")
            
            model_paths = {
                'best.pt': 'runs/classify/unified_car_brand/weights/best.pt',
                'epoch10.pt': 'runs/classify/unified_car_brand/weights/epoch10.pt',
            }
            
            for model_name, model_path in model_paths.items():
                if not os.path.exists(model_path):
                    print(f"\n   ❌ {model_name}: NOT FOUND at {model_path}")
                    continue
                
                print(f"\n   Testing {model_name}:")
                try:
                    brand_model = YOLO(model_path)
                    results_brand = brand_model(vehicle_crop, verbose=False)
                    
                    for result in results_brand:
                        if result.probs is not None:
                            top5_idx = result.probs.top5
                            top5_conf = result.probs.top5conf.cpu().numpy()
                            
                            print(f"      Top 5 predictions:")
                            for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf), 1):
                                brand = brand_model.names[idx]
                                print(f"         {i}. {brand}: {conf:.3f} ({conf*100:.1f}%)")
                            
                            if top5_conf[0] < 0.3:
                                print(f"\n      ⚠️  WARNING: Top confidence is LOW ({top5_conf[0]:.3f})")
                                print(f"         This explains why results show 'Unknown'")
                except Exception as e:
                    print(f"   ❌ Error loading {model_name}: {e}")
            
            break
    
    if not vehicle_found:
        print(f"\n   ❌ NO VEHICLES DETECTED - Brand classification cannot run!")
        print(f"   Try:")
        print(f"      - Use images with clear, centered vehicles")
        print(f"      - Ensure good lighting and image quality")
        print(f"      - Test with different images")
    
    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        test_image = "test_images/test1.jpg"
        if os.path.exists(test_image):
            print(f"No image specified, using: {test_image}")
            test_vehicle_detection(test_image)
        else:
            print("\nUsage: python diagnose_detection.py <image_path>")
            sys.exit(1)
    else:
        test_vehicle_detection(sys.argv[1])