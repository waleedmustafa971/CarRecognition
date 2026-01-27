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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
brand_model_unified = None
brand_model_logo = None
specialist_models = {}

MODEL_PATHS = {
    'unified': os.path.join(BASE_DIR, 'runs', 'classify', 'unified_car_brand', 'weights', 'epoch10.pt'),
    'logo': os.path.join(BASE_DIR, 'models', 'car_logo', 'weights', 'best.pt'),
}

SPECIALIST_BRANDS = ['kia', 'tesla', 'mclaren']

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

BRAND_NORMALIZATION = {
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
    'CADILLAC': ['CADILLAC'],
    'GMC': ['GMC'],
    'MCLAREN': ['MCLAREN', 'MC LAREN'],
    'LAMBORGHINI': ['LAMBORGHINI', 'LAMBO'],
    'FERRARI': ['FERRARI'],
    'PORSCHE': ['PORSCHE'],
    'BENTLEY': ['BENTLEY'],
    'ROLLS-ROYCE': ['ROLLS-ROYCE', 'ROLLS ROYCE', 'ROLLSROYCE'],
    'LAND ROVER': ['LAND ROVER', 'LAND-ROVER', 'LANDROVER', 'RANGE ROVER'],
    'JAGUAR': ['JAGUAR'],
    'LEXUS': ['LEXUS'],
    'INFINITI': ['INFINITI'],
    'ACURA': ['ACURA'],
    'GENESIS': ['GENESIS'],
    'LINCOLN': ['LINCOLN'],
    'BUICK': ['BUICK'],
    'CHRYSLER': ['CHRYSLER'],
    'DODGE': ['DODGE'],
    'JEEP': ['JEEP'],
    'RAM': ['RAM'],
    'SUBARU': ['SUBARU'],
    'MITSUBISHI': ['MITSUBISHI'],
    'SUZUKI': ['SUZUKI'],
    'PEUGEOT': ['PEUGEOT'],
    'RENAULT': ['RENAULT'],
    'CITROEN': ['CITROEN'],
    'FIAT': ['FIAT'],
    'ALFA ROMEO': ['ALFA ROMEO', 'ALFA'],
    'MASERATI': ['MASERATI'],
    'ASTON MARTIN': ['ASTON MARTIN', 'ASTON'],
    'BYD': ['BYD'],
    'GEELY': ['GEELY'],
    'CHERY': ['CHERY'],
    'GREAT WALL': ['GREAT WALL', 'GREATWALL', 'HAVAL'],
    'NIO': ['NIO'],
    'XPENG': ['XPENG'],
    'LI AUTO': ['LI AUTO', 'LI'],
}

LOGO_MODEL_BRANDS = set()

UAE_COMMON_BRANDS = {
    'TOYOTA', 'NISSAN', 'HONDA', 'HYUNDAI', 'KIA', 'MITSUBISHI', 'MAZDA',
    'LEXUS', 'INFINITI', 'BMW', 'MERCEDES-BENZ', 'AUDI', 'PORSCHE',
    'LAND ROVER', 'RANGE ROVER', 'FORD', 'CHEVROLET', 'GMC', 'JEEP',
    'DODGE', 'CHRYSLER', 'VOLKSWAGEN', 'TESLA', 'BENTLEY', 'ROLLS-ROYCE',
    'FERRARI', 'LAMBORGHINI', 'MASERATI', 'ASTON MARTIN', 'MCLAREN',
    'JAGUAR', 'VOLVO', 'PEUGEOT', 'RENAULT', 'MG', 'GENESIS', 'CADILLAC',
    'LINCOLN', 'SUBARU', 'SUZUKI', 'ISUZU', 'HUMMER', 'MINI', 'FIAT'
}

CHINESE_BRANDS = {
    'JIANGHUAI', 'YIQI', 'DONGFENG', 'CHANGAN', 'GEELY', 'BYD', 'CHERY',
    'GREAT WALL', 'HAVAL', 'BAOJUN', 'WULING', 'FOTON', 'JAC', 'BAIC',
    'ZOTYE', 'LIFAN', 'BRILLIANCE', 'FAW', 'SAIC', 'GAC', 'NIO', 'XPENG',
    'LI AUTO', 'HONGQI', 'ROEWE', 'MG', 'MAXUS', 'LYNK & CO', 'TANK',
    'ORA', 'GEOMETRY', 'ZEEKR', 'VOYAH', 'ARCFOX', 'AVATR', 'DENZA',
    'SHANGQIDATONG', 'DONGFENGFENGSHEN', 'JIULONG', 'KARRY', 'KAWEI'
}


def find_model_file(model_dir, preferred_epoch=None):
    if not os.path.exists(model_dir):
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    if not model_files:
        return None
    
    if preferred_epoch and f'epoch{preferred_epoch}.pt' in model_files:
        return os.path.join(model_dir, f'epoch{preferred_epoch}.pt')
    
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
    global yolo_model, ocr_reader, brand_model_unified, brand_model_logo, specialist_models, LOGO_MODEL_BRANDS
    
    logger.info("=" * 60)
    logger.info("CAR RECOGNITION SYSTEM - v6.0 (IMPROVED ENSEMBLE)")
    logger.info("=" * 60)
    logger.info(f"Base directory: {BASE_DIR}")
    
    if yolo_model is None:
        logger.info("Loading YOLO vehicle detection model...")
        yolo_model = YOLO('yolov8n.pt')
        logger.info("✓ YOLO loaded")
    
    if ocr_reader is None:
        logger.info("Loading EasyOCR...")
        ocr_reader = easyocr.Reader(['en'], gpu=True)
        logger.info("✓ EasyOCR loaded")
    
    logger.info("\n" + "=" * 60)
    logger.info("LOADING BRAND CLASSIFICATION MODELS")
    logger.info("=" * 60)
    
    models_loaded = 0
    
    if brand_model_unified is None:
        unified_dir = os.path.join(BASE_DIR, 'runs', 'classify', 'unified_car_brand', 'weights')
        epoch10_path = os.path.join(unified_dir, 'epoch10.pt')
        best_path = os.path.join(unified_dir, 'best.pt')
        
        logger.info("\n1. Loading UNIFIED model:")
        
        if os.path.exists(epoch10_path):
            logger.info(f"   Using epoch10.pt (better generalization)")
            try:
                brand_model_unified = YOLO(epoch10_path)
                logger.info(f"   ✓ Loaded ({len(brand_model_unified.names)} classes)")
                models_loaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
        elif os.path.exists(best_path):
            logger.info(f"   Using best.pt (fallback)")
            try:
                brand_model_unified = YOLO(best_path)
                logger.info(f"   ✓ Loaded ({len(brand_model_unified.names)} classes)")
                models_loaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
        else:
            logger.warning("   ⚠️ No unified model found")
    
    if brand_model_logo is None:
        logo_path = MODEL_PATHS['logo']
        logger.info(f"\n2. Loading LOGO model (modern cars):")
        if os.path.exists(logo_path):
            try:
                brand_model_logo = YOLO(logo_path)
                LOGO_MODEL_BRANDS = set(brand_model_logo.names.values())
                model_type = brand_model_logo.task
                logger.info(f"   ✓ Loaded ({len(brand_model_logo.names)} classes, type: {model_type})")
                logger.info(f"   Brands: {', '.join(sorted(LOGO_MODEL_BRANDS)[:10])}...")
                if model_type == 'detect':
                    logger.info(f"   ℹ️  Detection model - will find logos in images")
                models_loaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
        else:
            logger.warning(f"   ⚠️ Not found at {logo_path}")
    
    logger.info(f"\n3. Loading SPECIALIST models:")
    for brand in SPECIALIST_BRANDS:
        brand_dir = os.path.join(BASE_DIR, 'models', brand, 'weights')
        if os.path.exists(brand_dir):
            model_path = find_model_file(brand_dir)
            if model_path:
                try:
                    specialist_models[brand.upper()] = YOLO(model_path)
                    logger.info(f"   ✓ {brand.upper()} specialist loaded")
                    models_loaded += 1
                except Exception as e:
                    logger.error(f"   ❌ {brand.upper()} failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TOTAL MODELS LOADED: {models_loaded}")
    logger.info("=" * 60)


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


def cleanup_memory(force=False):
    gc.collect()
    gc.collect()
    
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        if force:
            torch.cuda.synchronize()
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()


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
    brand = brand.upper().strip()
    brand = brand.replace('_', ' ').replace('-', ' ')
    
    for normalized, variants in BRAND_NORMALIZATION.items():
        if brand in variants or brand == normalized:
            return normalized
    
    return brand


def get_model_predictions(model, vehicle_crop, model_name, top_k=5):
    results = None
    try:
        results = model(vehicle_crop, verbose=False)
        predictions = []
        
        for result in results:
            if result.probs is not None:
                top_indices = result.probs.top5[:top_k]
                top_confs = result.probs.top5conf.cpu().numpy()[:top_k]
                
                for idx, conf in zip(top_indices, top_confs):
                    brand = model.names[idx]
                    normalized = normalize_brand_name(brand)
                    predictions.append({
                        'brand': normalized,
                        'original': brand,
                        'confidence': float(conf),
                        'source': model_name
                    })
                break
            
            elif result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(min(len(boxes), top_k)):
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    brand = model.names[cls_id]
                    normalized = normalize_brand_name(brand)
                    predictions.append({
                        'brand': normalized,
                        'original': brand,
                        'confidence': conf,
                        'source': model_name
                    })
                
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                break
        
        return predictions[:top_k]
    except Exception as e:
        logger.error(f"{model_name} prediction failed: {e}")
        return []
    finally:
        if results is not None:
            del results
        cleanup_memory()


def run_specialist_check(vehicle_crop, candidate_brands):
    specialist_results = {}
    
    for brand in candidate_brands:
        if brand in specialist_models:
            results = None
            try:
                model = specialist_models[brand]
                results = model(vehicle_crop, verbose=False)
                
                for result in results:
                    if result.probs is not None:
                        top_conf = float(result.probs.top1conf.cpu().numpy())
                        specialist_results[brand] = top_conf
                        logger.info(f"   Specialist {brand}: {top_conf:.1%}")
                        break
            except Exception as e:
                logger.error(f"   Specialist {brand} failed: {e}")
            finally:
                if results is not None:
                    del results
                cleanup_memory()
    
    return specialist_results


def detect_brand_ensemble(vehicle_crop):
    logger.info(f"   Crop size: {vehicle_crop.shape[1]}x{vehicle_crop.shape[0]}")
    
    all_predictions = {}
    model_outputs = {}
    
    if brand_model_logo is not None:
        logo_preds = get_model_predictions(brand_model_logo, vehicle_crop, 'logo', top_k=5)
        model_outputs['logo'] = logo_preds
        
        if logo_preds and logo_preds[0]['confidence'] > 0.1:
            logger.info(f"   LOGO detections:")
            for i, p in enumerate(logo_preds[:3], 1):
                logger.info(f"      {i}. {p['brand']}: {p['confidence']:.1%}")
                
                brand = p['brand']
                if brand not in all_predictions:
                    all_predictions[brand] = {'logo': None, 'unified': None, 'specialist': None}
                all_predictions[brand]['logo'] = p['confidence']
        else:
            logger.info(f"   LOGO: No logo detected in image")
    
    if brand_model_unified is not None:
        unified_preds = get_model_predictions(brand_model_unified, vehicle_crop, 'unified', top_k=5)
        model_outputs['unified'] = unified_preds
        
        if unified_preds:
            logger.info(f"   UNIFIED predictions:")
            for i, p in enumerate(unified_preds[:3], 1):
                logger.info(f"      {i}. {p['brand']}: {p['confidence']:.1%}")
                
                brand = p['brand']
                if brand not in all_predictions:
                    all_predictions[brand] = {'logo': None, 'unified': None, 'specialist': None}
                all_predictions[brand]['unified'] = p['confidence']
    
    candidate_brands = set()
    for brand, scores in all_predictions.items():
        logo_score = scores.get('logo') or 0
        unified_score = scores.get('unified') or 0
        if logo_score > 0.1 or unified_score > 0.1:
            candidate_brands.add(brand)
    
    if specialist_models and candidate_brands:
        specialist_results = run_specialist_check(vehicle_crop, candidate_brands)
        for brand, conf in specialist_results.items():
            if brand in all_predictions:
                all_predictions[brand]['specialist'] = conf
    
    final_scores = {}
    
    for brand, scores in all_predictions.items():
        logo_conf = scores.get('logo') or 0
        unified_conf = scores.get('unified') or 0
        specialist_conf = scores.get('specialist')
        
        if specialist_conf is not None and specialist_conf > 0.7:
            final_scores[brand] = specialist_conf * 1.3
            logger.info(f"   ⭐ {brand}: Specialist high confidence ({specialist_conf:.1%})")
            continue
        
        if logo_conf > 0.3:
            if unified_conf > 0.2:
                final_scores[brand] = min(logo_conf * 1.4, 1.0)
                logger.info(f"   ⭐ {brand}: Logo detected + unified agrees")
            else:
                final_scores[brand] = min(logo_conf * 1.3, 1.0)
                logger.info(f"   ⭐ {brand}: Logo detected ({logo_conf:.1%})")
            continue
        
        if logo_conf > 0.5 and unified_conf > 0.3:
            agreement_bonus = 1.25
            final_scores[brand] = ((logo_conf * 1.2 + unified_conf) / 2) * agreement_bonus
            logger.info(f"   ✓ {brand}: Both models agree")
            continue
        
        if logo_conf > 0.6:
            final_scores[brand] = logo_conf * 1.15
            continue
        
        if logo_conf > 0 and unified_conf > 0:
            final_scores[brand] = (logo_conf * 1.2 + unified_conf * 0.8) / 2
        elif logo_conf > 0:
            final_scores[brand] = logo_conf * 1.1
        elif unified_conf > 0:
            final_scores[brand] = unified_conf * 0.9
    
    if not final_scores:
        logger.warning("   ❌ No predictions from any model")
        return [{"make": "Unknown", "model": "", "score": 0.0}]
    
    sorted_brands = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_brands) >= 2:
        top_brand, top_score = sorted_brands[0]
        second_brand, second_score = sorted_brands[1]
        score_diff = top_score - second_score
        
        top_is_chinese = top_brand in CHINESE_BRANDS
        second_is_common = second_brand in UAE_COMMON_BRANDS
        
        if top_is_chinese and second_is_common and score_diff < 0.3:
            logger.info(f"   ⚠️ Regional bias correction: {top_brand} vs {second_brand}")
            logger.info(f"      {top_brand} (Chinese): {top_score:.1%}")
            logger.info(f"      {second_brand} (UAE common): {second_score:.1%}")
            
            if score_diff < 0.15:
                final_scores[second_brand] = second_score * 1.3
                logger.info(f"      → Boosting {second_brand} (close match)")
            elif score_diff < 0.25:
                final_scores[second_brand] = second_score * 1.15
                logger.info(f"      → Slight boost to {second_brand}")
            
            sorted_brands = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    predictions = []
    for brand, score in sorted_brands[:5]:
        predictions.append({
            "make": brand,
            "model": "",
            "score": min(float(score), 1.0)
        })
    
    if len(sorted_brands) >= 2:
        top_score = predictions[0]['score']
        second_score = predictions[1]['score'] if len(predictions) > 1 else 0
        if top_score - second_score < 0.2 and top_score < 0.7:
            logger.info(f"   ⚠️ UNCERTAIN: {predictions[0]['make']} vs {predictions[1]['make']}")
    
    logger.info(f"\n   ✓ FINAL: {predictions[0]['make']} ({predictions[0]['score']:.1%})")
    
    del model_outputs
    cleanup_memory()
    
    return predictions


def detect_color_hsv(vehicle_crop):
    try:
        if vehicle_crop is None or vehicle_crop.size == 0:
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
                color_scores[target] = color_scores.get(target, 0) + percentage
            else:
                color_scores[color_name] = color_scores.get(color_name, 0) + percentage
        
        sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for color_name, percentage in sorted_colors:
            if percentage > 2.0:
                results.append({
                    "color": color_name,
                    "percentage": float(percentage),
                    "rgb": [255, 255, 255] if color_name == "White" else [0, 0, 0]
                })
        
        if not results:
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


def calculate_iou(box1, box2):
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_brand_predictions(all_brands):
    if not all_brands:
        return [{"make": "Unknown", "model": "", "score": 0.0}]
    
    brand_scores = {}
    
    for pred in all_brands:
        brand = pred['make']
        score = pred['score']
        
        if brand not in brand_scores:
            brand_scores[brand] = []
        brand_scores[brand].append(score)
    
    merged = []
    for brand, scores in brand_scores.items():
        if len(scores) > 1:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            final_score = (avg_score + max_score) / 2
            logger.info(f"   Merged {brand}: {len(scores)} detections → {final_score:.1%}")
        else:
            final_score = scores[0]
        
        merged.append({
            "make": brand,
            "model": "",
            "score": final_score
        })
    
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged


def detect_car_and_plate(image_path):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing: {image_path}")
    logger.info("=" * 60)
    
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
                break
            
            logger.info(f"   Trying threshold: {conf_threshold}")
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
                    logger.info(f"\n🚘 Vehicle #{vehicle_count} ({box_width}x{box_height})")
                    
                    logger.info("🏷️  Detecting brand...")
                    try:
                        brands = detect_brand_ensemble(vehicle_crop)
                        all_brands.extend(brands)
                    except Exception as e:
                        logger.error(f"Brand detection failed: {e}")
                        all_brands.append({"make": "Unknown", "model": "", "score": 0.0})
                    
                    logger.info("🎨 Detecting color...")
                    try:
                        colors = detect_color_hsv(vehicle_crop)
                        all_colors.extend(colors)
                    except Exception as e:
                        logger.error(f"Color detection failed: {e}")
                        all_colors.append({"color": "Gray", "percentage": 0.0, "rgb": [128, 128, 128]})
                    
                    logger.info("🔢 Detecting plate...")
                    try:
                        plate = detect_license_plate(vehicle_crop)
                        if plate:
                            all_plates.append(plate)
                            logger.info(f"   Plate: {plate}")
                        else:
                            logger.info("   No plate detected")
                    except Exception as e:
                        logger.error(f"Plate detection failed: {e}")
                    
                    del vehicle_crop
                    cleanup_memory()
                    
                    if vehicle_count >= 3:
                        break
                
                if vehicle_count >= 3:
                    break
            
            del vehicle_detections
            cleanup_memory()
        
        if vehicle_count == 0:
            logger.error("❌ NO VEHICLES DETECTED")
        
        if len(all_brands) > 1:
            logger.info("\n🔄 Merging predictions from multiple detections...")
            merged_brands = merge_brand_predictions(all_brands)
        else:
            merged_brands = all_brands if all_brands else [{"make": "Unknown", "model": "", "score": 0.0}]
        
        result = {
            "car_detections": car_boxes,
            "plate_number": all_plates[0] if all_plates else None,
            "car_make_model": merged_brands,
            "colors": all_colors
        }
        
        logger.info(f"\n{'=' * 60}")
        logger.info("✅ RESULTS:")
        logger.info(f"   Vehicles: {len(car_boxes)}")
        if merged_brands:
            logger.info(f"   Brand: {merged_brands[0]['make']} ({merged_brands[0]['score']:.1%})")
            if len(merged_brands) > 1 and merged_brands[1]['score'] > 0.3:
                logger.info(f"   Alt:   {merged_brands[1]['make']} ({merged_brands[1]['score']:.1%})")
        if all_colors:
            logger.info(f"   Color: {all_colors[0]['color']}")
        if all_plates:
            logger.info(f"   Plate: {all_plates[0]}")
        logger.info("=" * 60)
        
        cleanup_memory(force=True)
        
        return result


def process_image(image_path):
    return detect_car_and_plate(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python main.py <image_path>")
        sys.exit(1)
    
    result = detect_car_and_plate(sys.argv[1])
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))