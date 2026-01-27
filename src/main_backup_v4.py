import ssl
import urllib3
import certifi
import os
import sys
import cv2
import json
import numpy as np
import tempfile
import requests
from ultralytics import YOLO
import easyocr
from pathlib import Path
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gc
from contextlib import contextmanager
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'
)
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR.endswith('src'):
    BASE_DIR = os.path.dirname(BASE_DIR)

sys.path.append(BASE_DIR)

yolo_model = None
ocr_reader = None
brand_model_epoch10 = None
brand_model_best = None
brand_model_logo = None
brand_model_kia = None

MODEL_PATHS = {
    'brand_epoch10': os.path.join(BASE_DIR, 'runs', 'classify', 'unified_car_brand', 'weights', 'epoch10.pt'),
    'brand_best': os.path.join(BASE_DIR, 'runs', 'classify', 'unified_car_brand', 'weights', 'best.pt'),
    'logo_model': os.path.join(BASE_DIR, 'models', 'car_logo', 'weights', 'best.pt'),
    'kia_model': os.path.join(BASE_DIR, 'models', 'kia', 'weights', 'best.pt'),
}

CAR_COLOR_RANGES = {
    'White': {
        'lower': np.array([0, 0, 180]),
        'upper': np.array([180, 40, 255]),
        'priority': 1
    },
    'Silver': {
        'lower': np.array([0, 0, 120]),
        'upper': np.array([180, 50, 200]),
        'priority': 2
    },
    'Black': {
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 255, 60]),
        'priority': 10
    },
    'Gray': {
        'lower': np.array([0, 0, 60]),
        'upper': np.array([180, 40, 180]),
        'priority': 3
    },
    'Red': {
        'lower': np.array([0, 100, 80]),
        'upper': np.array([10, 255, 255]),
        'priority': 1
    },
    'Red2': {
        'lower': np.array([170, 100, 80]),
        'upper': np.array([180, 255, 255]),
        'priority': 1,
        'merge_with': 'Red'
    },
    'Blue': {
        'lower': np.array([90, 80, 80]),
        'upper': np.array([130, 255, 255]),
        'priority': 1
    },
    'Green': {
        'lower': np.array([35, 80, 80]),
        'upper': np.array([85, 255, 255]),
        'priority': 1
    },
    'Yellow': {
        'lower': np.array([20, 100, 100]),
        'upper': np.array([35, 255, 255]),
        'priority': 1
    },
    'Orange': {
        'lower': np.array([10, 100, 100]),
        'upper': np.array([20, 255, 255]),
        'priority': 1
    },
    'Brown': {
        'lower': np.array([10, 100, 40]),
        'upper': np.array([20, 255, 150]),
        'priority': 1
    },
    'Beige': {
        'lower': np.array([15, 20, 140]),
        'upper': np.array([30, 80, 230]),
        'priority': 2
    },
    'Gold': {
        'lower': np.array([20, 100, 140]),
        'upper': np.array([35, 200, 255]),
        'priority': 1
    }
}

def find_model_file(model_dir):
    """
    Find the best model file in a directory
    Priority: best.pt > last.pt > epoch*.pt (highest number)
    """
    if not os.path.exists(model_dir):
        return None
    
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.pt'):
            model_files.append(file)
    
    if not model_files:
        return None
    
    if 'best.pt' in model_files:
        return os.path.join(model_dir, 'best.pt')
    
    if 'last.pt' in model_files:
        return os.path.join(model_dir, 'last.pt')
    
    epoch_files = [f for f in model_files if f.startswith('epoch')]
    if epoch_files:
        epoch_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'), reverse=True)
        return os.path.join(model_dir, epoch_files[0])
    
    return os.path.join(model_dir, model_files[0])

def initialize_models():
    global yolo_model, ocr_reader, brand_model_epoch10, brand_model_best, brand_model_logo, brand_model_kia
    
    logger.info("="*60)
    logger.info("CAR RECOGNITION SYSTEM - MULTI-MODEL v5.0")
    logger.info("SUPPORTS: OLD MODELS (2013-2019) + NEW MODELS + KIA SPECIALIST")
    logger.info("="*60)
    logger.info(f"Base directory: {BASE_DIR}")
    
    if yolo_model is None:
        logger.info("Loading YOLO vehicle detection model...")
        yolo_model = YOLO('yolov8n.pt')
        logger.info("✓ YOLO loaded")
    
    if ocr_reader is None:
        logger.info("Loading EasyOCR...")
        ocr_reader = easyocr.Reader(['en'], gpu=True)
        logger.info("✓ EasyOCR loaded")
    
    logger.info("\n" + "="*60)
    logger.info("LOADING BRAND CLASSIFICATION MODELS")
    logger.info("="*60)
    
    models_loaded = 0
    
    if brand_model_epoch10 is None:
        epoch10_path = MODEL_PATHS['brand_epoch10']
        logger.info(f"\n1. Checking EPOCH10 model (old models 2013-2019):")
        logger.info(f"   Path: {epoch10_path}")
        if os.path.exists(epoch10_path):
            try:
                brand_model_epoch10 = YOLO(epoch10_path)
                logger.info(f"   ✓ Loaded ({len(brand_model_epoch10.names)} classes)")
                models_loaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to load: {e}")
        else:
            logger.warning(f"   ⚠️  Not found")
    
    if brand_model_best is None:
        best_path = MODEL_PATHS['brand_best']
        logger.info(f"\n2. Checking BEST model (old models 2013-2019):")
        logger.info(f"   Path: {best_path}")
        if os.path.exists(best_path):
            try:
                brand_model_best = YOLO(best_path)
                logger.info(f"   ✓ Loaded ({len(brand_model_best.names)} classes)")
                models_loaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to load: {e}")
        else:
            logger.warning(f"   ⚠️  Not found")
    
    if brand_model_logo is None:
        logo_dir = os.path.join(BASE_DIR, 'models', 'car_logo', 'weights')
        logger.info(f"\n3. Checking LOGO model (new models):")
        logger.info(f"   Directory: {logo_dir}")
        logo_path = find_model_file(logo_dir)
        if logo_path:
            logger.info(f"   Found: {os.path.basename(logo_path)}")
            try:
                brand_model_logo = YOLO(logo_path)
                logger.info(f"   ✓ Loaded ({len(brand_model_logo.names)} classes)")
                models_loaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to load: {e}")
        else:
            logger.warning(f"   ⚠️  No model files found in directory")
    
    if brand_model_kia is None:
        kia_dir = os.path.join(BASE_DIR, 'models', 'kia', 'weights')
        logger.info(f"\n4. Checking KIA SPECIALIST model:")
        logger.info(f"   Directory: {kia_dir}")
        kia_path = find_model_file(kia_dir)
        if kia_path:
            logger.info(f"   Found: {os.path.basename(kia_path)}")
            try:
                brand_model_kia = YOLO(kia_path)
                logger.info(f"   ✓ Loaded ({len(brand_model_kia.names)} classes)")
                models_loaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to load: {e}")
        else:
            logger.warning(f"   ⚠️  No model files found in directory")
    
    logger.info("\n" + "="*60)
    logger.info(f"MODELS LOADED: {models_loaded}/4")
    
    if models_loaded == 0:
        logger.error("CRITICAL: NO BRAND MODELS LOADED!")
        logger.error("All brand detections will return 'Unknown'")
    elif models_loaded < 4:
        logger.warning(f"Only {models_loaded}/4 models loaded - some predictions may be limited")
    else:
        logger.info("ALL MODELS LOADED SUCCESSFULLY!")
    
    logger.info("="*60)
    logger.info("✓ OpenCV HSV color detection ready")
    logger.info("="*60)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("="*60)

try:
    initialize_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    raise

@contextmanager
def managed_cv2_image(path):
    image = None
    try:
        image = cv2.imread(path)
        yield image
    finally:
        if image is not None:
            del image
        gc.collect()

def clean_plate_text(text):
    if not text:
        return ""
    
    text = text.upper().strip()
    digits = ''.join(c for c in text if c.isdigit())
    
    if not digits:
        return ""
    
    if len(digits) <= 5:
        return digits
    
    if len(digits) == 6:
        if digits[0] == digits[1]:
            return digits[1:]
        if digits[0] == '7' and digits[1] != '7':
            return digits[1:]
        return digits[-5:]
    
    if len(digits) > 6:
        return digits[-5:]
    
    return digits

def normalize_brand_name(brand):
    """
    Normalize brand names for comparison
    """
    brand = brand.upper().strip()
    
    brand_mappings = {
        'KIA': ['KIA', 'KIA MOTORS'],
        'HYUNDAI': ['HYUNDAI', 'HYUNDAI MOTOR'],
        'TOYOTA': ['TOYOTA', 'TOYOTA MOTOR'],
        'HONDA': ['HONDA', 'HONDA MOTOR'],
        'NISSAN': ['NISSAN', 'NISSAN MOTOR'],
        'MERCEDES-BENZ': ['MERCEDES', 'MERCEDES-BENZ', 'MERCEDES BENZ', 'BENZ'],
        'BMW': ['BMW'],
        'AUDI': ['AUDI'],
        'VOLKSWAGEN': ['VOLKSWAGEN', 'VW'],
        'FORD': ['FORD', 'FORD MOTOR'],
        'CHEVROLET': ['CHEVROLET', 'CHEVY'],
        'TESLA': ['TESLA', 'TESLA MOTORS'],
        'MG': ['MG', 'MG MOTOR', 'MORRIS GARAGES'],
        'MAZDA': ['MAZDA', 'MAZDA MOTOR'],
        'VOLVO': ['VOLVO', 'VOLVO CARS'],
    }
    
    for normalized, variants in brand_mappings.items():
        if brand in variants:
            return normalized
    
    return brand

def detect_brand_multi_model_ensemble(vehicle_crop):
    """
    Advanced ensemble that runs ALL available models and combines predictions intelligently
    """
    logger.info(f"   Vehicle crop size: {vehicle_crop.shape[1]}x{vehicle_crop.shape[0]}")
    
    available_models = []
    if brand_model_epoch10 is not None:
        available_models.append(('epoch10', brand_model_epoch10, 1.0))
    if brand_model_best is not None:
        available_models.append(('best', brand_model_best, 1.0))
    if brand_model_logo is not None:
        available_models.append(('logo', brand_model_logo, 1.3))
    if brand_model_kia is not None:
        available_models.append(('kia', brand_model_kia, 1.5))
    
    if len(available_models) == 0:
        logger.error("❌ No brand models loaded!")
        return [{"make": "Unknown", "model": "", "score": 0.0}]
    
    logger.info(f"   Running {len(available_models)} models...")
    
    try:
        all_predictions = {}
        model_results = {}
        
        for model_name, model, weight in available_models:
            try:
                logger.info(f"\n   Running {model_name.upper()} model...")
                results = model(vehicle_crop, verbose=False)
                
                predictions_found = False
                for result in results:
                    if result.probs is not None:
                        top5_idx = result.probs.top5
                        top5_conf = result.probs.top5conf.cpu().numpy()
                        
                        if len(top5_idx) > 0 and top5_conf[0] > 0.001:
                            predictions_found = True
                            logger.info(f"   {model_name.upper()} top 5 predictions:")
                            predictions_list = []
                            
                            for i, (idx, conf) in enumerate(zip(top5_idx[:5], top5_conf[:5]), 1):
                                brand = model.names[idx]
                                normalized_brand = normalize_brand_name(brand)
                                
                                logger.info(f"      {i}. {brand}: {conf:.3f} ({conf*100:.1f}%)")
                                
                                if i <= 3:
                                    predictions_list.append({
                                        'brand': normalized_brand,
                                        'original': brand,
                                        'confidence': float(conf)
                                    })
                                    
                                    if normalized_brand not in all_predictions:
                                        all_predictions[normalized_brand] = []
                                    
                                    weighted_score = float(conf) * weight
                                    all_predictions[normalized_brand].append({
                                        'score': weighted_score,
                                        'raw_score': float(conf),
                                        'source': model_name,
                                        'weight': weight
                                    })
                            
                            model_results[model_name] = predictions_list
                        else:
                            logger.warning(f"   {model_name.upper()}: No confident predictions (top score: {top5_conf[0]:.3f})")
                        break
                
                if not predictions_found:
                    logger.warning(f"   {model_name.upper()}: Model returned no predictions")
                        
            except Exception as e:
                logger.error(f"   {model_name.upper()} model failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        if len(all_predictions) == 0:
            logger.error("❌ All models failed to predict!")
            return [{"make": "Unknown", "model": "", "score": 0.0}]
        
        logger.info(f"\n   Combining predictions from {len(model_results)} models...")
        
        combined_scores = {}
        for brand, preds in all_predictions.items():
            if brand == 'KIA' and 'kia' in [p['source'] for p in preds]:
                kia_scores = [p['score'] for p in preds if p['source'] == 'kia']
                if kia_scores and max(kia_scores) > 0.5:
                    combined_scores[brand] = max(kia_scores) * 1.5
                    logger.info(f"   ⭐ KIA specialist detected KIA with high confidence - boosting score")
                    continue
            
            if 'logo' in [p['source'] for p in preds]:
                logo_scores = [p for p in preds if p['source'] == 'logo']
                if logo_scores and logo_scores[0]['raw_score'] > 0.5:
                    old_model_scores = [p for p in preds if p['source'] in ['epoch10', 'best']]
                    if not old_model_scores or max([p['raw_score'] for p in old_model_scores]) < 0.8:
                        combined_scores[brand] = logo_scores[0]['score'] * 1.4
                        logger.info(f"   ⭐ Logo model has strong prediction for {brand} - boosting score")
                        continue
            
            if len(preds) >= 3:
                avg_score = sum(p['score'] for p in preds) / len(preds)
                combined_scores[brand] = avg_score * 1.3
                logger.info(f"   ✓ {brand}: Strong agreement from {len(preds)} models")
            elif len(preds) == 2:
                avg_score = sum(p['score'] for p in preds) / len(preds)
                combined_scores[brand] = avg_score * 1.2
            else:
                combined_scores[brand] = preds[0]['score']
        
        sorted_brands = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        predictions = []
        for brand, score in sorted_brands[:5]:
            predictions.append({
                "make": brand,
                "model": "",
                "score": min(score, 1.0)
            })
        
        if len(predictions) > 0:
            sources = [p['source'] for brand_preds in all_predictions.get(predictions[0]['make'], []) for p in [brand_preds]]
            logger.info(f"\n   ✓ FINAL Top brand: {predictions[0]['make']} ({predictions[0]['score']:.1%})")
            logger.info(f"      Sources: {', '.join(set(sources))}")
            
            if predictions[0]['score'] < 0.30:
                logger.warning(f"   ⚠️  Low confidence: {predictions[0]['score']:.3f}")
            
            return predictions[:3]
        
        return [{"make": "Unknown", "model": "", "score": 0.0}]
    
    except Exception as e:
        logger.error(f"CRITICAL ERROR in brand detection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [{"make": "Unknown", "model": "", "score": 0.0}]

def detect_color_hsv_simple(vehicle_crop):
    """
    HSV color detection with robust fallbacks
    """
    try:
        if vehicle_crop is None or vehicle_crop.size == 0:
            logger.warning("Empty vehicle crop")
            return [{"color": "Gray", "percentage": 0.0, "rgb": [128, 128, 128]}]
        
        h, w = vehicle_crop.shape[:2]
        
        y_start = max(0, int(h * 0.2))
        y_end = min(h, int(h * 0.7))
        x_start = max(0, int(w * 0.15))
        x_end = min(w, int(w * 0.85))
        
        car_body = vehicle_crop[y_start:y_end, x_start:x_end]
        
        if car_body.size == 0:
            car_body = vehicle_crop
        
        hsv = cv2.cvtColor(car_body, cv2.COLOR_BGR2HSV)
        hsv_blur = cv2.GaussianBlur(hsv, (7, 7), 0)
        
        total_pixels = car_body.shape[0] * car_body.shape[1]
        color_scores = {}
        
        for color_name, color_info in CAR_COLOR_RANGES.items():
            mask = cv2.inRange(hsv_blur, color_info['lower'], color_info['upper'])
            color_pixels = cv2.countNonZero(mask)
            percentage = (color_pixels / total_pixels) * 100
            
            if 'merge_with' in color_info:
                target = color_info['merge_with']
                if target not in color_scores:
                    color_scores[target] = 0
                color_scores[target] += percentage
            else:
                if color_name not in color_scores:
                    color_scores[color_name] = 0
                color_scores[color_name] += percentage
        
        sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for color_name, percentage in sorted_colors:
            if percentage > 2.0:
                results.append({
                    "color": color_name,
                    "percentage": float(percentage),
                    "rgb": [255, 255, 255] if color_name == "White" else [0, 0, 0]
                })
        
        if len(results) == 0:
            gray = cv2.cvtColor(car_body, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness > 180:
                return [{"color": "White", "percentage": 50.0, "rgb": [255, 255, 255]}]
            elif avg_brightness > 120:
                return [{"color": "Silver", "percentage": 50.0, "rgb": [192, 192, 192]}]
            elif avg_brightness > 60:
                return [{"color": "Gray", "percentage": 50.0, "rgb": [128, 128, 128]}]
            else:
                return [{"color": "Black", "percentage": 50.0, "rgb": [0, 0, 0]}]
        
        if len(results) > 1 and results[0]['color'] in ['White', 'Silver', 'Gray']:
            if results[1]['color'] == 'Black' and results[0]['percentage'] > 30:
                results = [r for r in results if r['color'] != 'Black']
        
        logger.info(f"   Color: {results[0]['color']} ({results[0]['percentage']:.1f}%)")
        
        return results[:2]
    
    except Exception as e:
        logger.error(f"Color detection error: {e}")
        return [{"color": "Gray", "percentage": 0.0, "rgb": [128, 128, 128]}]

def detect_license_plate(vehicle_crop):
    """
    Detect license plate using OCR
    """
    try:
        ocr_results = ocr_reader.readtext(vehicle_crop)
        
        if not ocr_results:
            return None
        
        all_texts = []
        for detection in ocr_results:
            text = detection[1]
            confidence = detection[2]
            
            if confidence > 0.3:
                cleaned = clean_plate_text(text)
                if cleaned:
                    all_texts.append({
                        'text': cleaned,
                        'confidence': confidence
                    })
        
        if all_texts:
            all_texts.sort(key=lambda x: x['confidence'], reverse=True)
            return all_texts[0]['text']
        
        return None
    
    except Exception as e:
        logger.error(f"Plate detection error: {e}")
        return None

def detect_car_and_plate(image_path):
    """
    Main detection function with progressive confidence thresholds
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {image_path}")
    logger.info("="*60)
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return {
            "car_detections": [],
            "plate_number": None,
            "car_make_model": [],
            "colors": []
        }
    
    with managed_cv2_image(image_path) as image:
        if image is None:
            logger.error("Failed to load image")
            return {
                "car_detections": [],
                "plate_number": None,
                "car_make_model": [],
                "colors": []
            }
        
        img_height, img_width = image.shape[:2]
        logger.info(f"Image: {img_width}x{img_height}")
        
        logger.info("🚗 Detecting vehicles...")
        
        all_brands = []
        all_colors = []
        all_plates = []
        car_boxes = []
        
        vehicle_count = 0
        
        for conf_threshold in [0.25, 0.15, 0.10, 0.05]:
            if vehicle_count > 0:
                logger.info(f"   Vehicles found at threshold {conf_threshold}, skipping lower thresholds")
                break
            
            logger.info(f"   Trying confidence threshold: {conf_threshold}")
            vehicle_detections = yolo_model(image, verbose=False, classes=[2, 5, 7], conf=conf_threshold)
            
            for detection in vehicle_detections:
                if detection.boxes is None or len(detection.boxes) == 0:
                    continue
                
                for box in detection.boxes:
                    confidence = float(box.conf[0])
                    
                    if confidence < conf_threshold:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    if box_width < 80 or box_height < 80:
                        logger.info(f"   Skipping small detection: {box_width}x{box_height}")
                        continue
                    
                    vehicle_count += 1
                    
                    car_boxes.append({
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "confidence": confidence
                    })
                    
                    vehicle_crop = image[y1:y2, x1:x2].copy()
                    logger.info(f"\n🚘 Vehicle #{vehicle_count} - crop: {box_width}x{box_height} (conf: {confidence:.2%})")
                    
                    logger.info("🏷️  Detecting brand (multi-model ensemble)...")
                    try:
                        brands = detect_brand_multi_model_ensemble(vehicle_crop)
                        all_brands.extend(brands)
                    except Exception as e:
                        logger.error(f"Brand detection failed: {e}")
                        brands = [{"make": "Unknown", "model": "", "score": 0.0}]
                        all_brands.extend(brands)
                    
                    logger.info("🎨 Detecting color...")
                    try:
                        colors = detect_color_hsv_simple(vehicle_crop)
                        all_colors.extend(colors)
                    except Exception as e:
                        logger.error(f"Color detection failed: {e}")
                        colors = [{"color": "Gray", "percentage": 0.0, "rgb": [128, 128, 128]}]
                        all_colors.extend(colors)
                    
                    logger.info("🔢 Detecting plate...")
                    try:
                        plate = detect_license_plate(vehicle_crop)
                        if plate:
                            all_plates.append(plate)
                            logger.info(f"   Plate: {plate}")
                        else:
                            logger.info(f"   No plate detected")
                    except Exception as e:
                        logger.error(f"Plate detection failed: {e}")
                    
                    if vehicle_count >= 3:
                        logger.info("   Reached 3 vehicles, stopping detection")
                        break
                
                if vehicle_count >= 3:
                    break
        
        if vehicle_count == 0:
            logger.error("="*60)
            logger.error("❌ NO VEHICLES DETECTED!")
            logger.error("This image may not contain visible vehicles")
            logger.error("Try images with clear, centered vehicles")
            logger.error("="*60)
        
        result = {
            "car_detections": car_boxes,
            "plate_number": all_plates[0] if all_plates else None,
            "car_make_model": all_brands,
            "colors": all_colors
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ FINAL RESULTS:")
        logger.info(f"   Vehicles detected: {len(car_boxes)}")
        logger.info(f"   Brands: {len(all_brands)}")
        if len(all_brands) > 0:
            logger.info(f"   Top brand: {all_brands[0]['make']} (score: {all_brands[0]['score']:.3f})")
        logger.info(f"   Colors: {len(all_colors)}")
        if len(all_colors) > 0:
            logger.info(f"   Top color: {all_colors[0]['color']}")
        if all_plates:
            logger.info(f"   Plate: {all_plates[0]}")
        logger.info("="*60)
        
        return result

def process_image(image_path):
    return detect_car_and_plate(image_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python main.py <image_path>")
        sys.exit(1)
    
    result = detect_car_and_plate(sys.argv[1])
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))