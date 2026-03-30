import os
import sys
import cv2
import json
import numpy as np
from ultralytics import YOLO
import easyocr
import logging
import gc
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
    'White': {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 40, 255])},
    'Silver': {'lower': np.array([0, 0, 120]), 'upper': np.array([180, 50, 200])},
    'Black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 60])},
    'Gray': {'lower': np.array([0, 0, 60]), 'upper': np.array([180, 40, 180])},
    'Red': {'lower': np.array([0, 100, 80]), 'upper': np.array([10, 255, 255])},
    'Red2': {'lower': np.array([170, 100, 80]), 'upper': np.array([180, 255, 255]), 'merge_with': 'Red'},
    'Blue': {'lower': np.array([90, 80, 80]), 'upper': np.array([130, 255, 255])},
    'Green': {'lower': np.array([35, 80, 80]), 'upper': np.array([85, 255, 255])},
    'Yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
    'Orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([20, 255, 255])},
    'Brown': {'lower': np.array([10, 100, 40]), 'upper': np.array([20, 255, 150])},
    'Beige': {'lower': np.array([15, 20, 140]), 'upper': np.array([30, 80, 230])},
    'Gold': {'lower': np.array([20, 100, 140]), 'upper': np.array([35, 200, 255])},
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
    'LAND ROVER', 'FORD', 'CHEVROLET', 'GMC', 'JEEP', 'DODGE', 'CHRYSLER',
    'VOLKSWAGEN', 'TESLA', 'BENTLEY', 'ROLLS-ROYCE', 'FERRARI',
    'LAMBORGHINI', 'MASERATI', 'ASTON MARTIN', 'MCLAREN', 'JAGUAR',
    'VOLVO', 'PEUGEOT', 'RENAULT', 'MG', 'GENESIS', 'CADILLAC', 'LINCOLN',
    'SUBARU', 'SUZUKI', 'ISUZU', 'MINI', 'FIAT',
}

CHINESE_BRANDS = {
    'JIANGHUAI', 'YIQI', 'DONGFENG', 'CHANGAN', 'GEELY', 'BYD', 'CHERY',
    'GREAT WALL', 'HAVAL', 'BAOJUN', 'WULING', 'FOTON', 'JAC', 'BAIC',
    'ZOTYE', 'LIFAN', 'BRILLIANCE', 'FAW', 'SAIC', 'GAC', 'NIO', 'XPENG',
    'LI AUTO', 'HONGQI', 'ROEWE', 'MG', 'MAXUS', 'LYNK & CO', 'TANK',
    'ORA', 'GEOMETRY', 'ZEEKR', 'VOYAH', 'ARCFOX', 'AVATR', 'DENZA',
}

UAE_PLATE_CODES = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'AA', 'AB', 'AC', 'AD', 'AE',
}

CHAR_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'S': '5',
    'B': '8', 'G': '6', 'Z': '2', 'T': '7',
    'Q': '0', 'D': '0',
}


def find_model_file(model_dir, preferred_epoch=None):
    if not os.path.exists(model_dir):
        return None

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        return None

    if preferred_epoch and f'epoch{preferred_epoch}.pt' in model_files:
        return os.path.join(model_dir, f'epoch{preferred_epoch}.pt')

    for name in ['best.pt', 'last.pt']:
        if name in model_files:
            return os.path.join(model_dir, name)

    epoch_files = [f for f in model_files if f.startswith('epoch')]
    if epoch_files:
        epoch_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'), reverse=True)
        return os.path.join(model_dir, epoch_files[0])

    return os.path.join(model_dir, model_files[0])


def initialize_models():
    global yolo_model, ocr_reader, brand_model_unified, brand_model_logo
    global specialist_models, LOGO_MODEL_BRANDS

    logger.info("=" * 50)
    logger.info("CAR RECOGNITION SYSTEM v8.0 (MOBILE)")
    logger.info("=" * 50)

    if yolo_model is None:
        yolo_model = YOLO('yolov8n.pt')
        logger.info("YOLO loaded")

    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'], gpu=HAS_TORCH)
        logger.info("EasyOCR loaded")

    if brand_model_unified is None:
        unified_dir = os.path.join(BASE_DIR, 'runs', 'classify', 'unified_car_brand', 'weights')
        for name in ['epoch10.pt', 'best.pt']:
            path = os.path.join(unified_dir, name)
            if os.path.exists(path):
                brand_model_unified = YOLO(path)
                logger.info(f"Unified model loaded: {name}")
                break

    if brand_model_logo is None:
        logo_path = MODEL_PATHS['logo']
        if os.path.exists(logo_path):
            brand_model_logo = YOLO(logo_path)
            LOGO_MODEL_BRANDS = set(brand_model_logo.names.values())
            logger.info(f"Logo model loaded ({len(brand_model_logo.names)} classes)")

    for brand in SPECIALIST_BRANDS:
        brand_dir = os.path.join(BASE_DIR, 'models', brand, 'weights')
        if os.path.exists(brand_dir):
            model_path = find_model_file(brand_dir)
            if model_path:
                specialist_models[brand.upper()] = YOLO(model_path)
                logger.info(f"Specialist {brand.upper()} loaded")

    logger.info("All models ready")


try:
    initialize_models()
except Exception as e:
    logger.error(f"Model init failed: {e}")
    raise


def cleanup_memory():
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_brand_name(brand):
    brand = brand.upper().strip().replace('_', ' ').replace('-', ' ')
    for normalized, variants in BRAND_NORMALIZATION.items():
        if brand in variants or brand == normalized:
            return normalized
    return brand


def get_model_predictions(model, vehicle_crop, model_name, top_k=3):
    try:
        results = model(vehicle_crop, verbose=False)
        predictions = []

        for result in results:
            if result.probs is not None:
                top_indices = result.probs.top5[:top_k]
                top_confs = result.probs.top5conf.cpu().numpy()[:top_k]
                for idx, conf in zip(top_indices, top_confs):
                    predictions.append({
                        'brand': normalize_brand_name(model.names[idx]),
                        'confidence': float(conf),
                        'source': model_name,
                    })
                break
            elif result.boxes is not None and len(result.boxes) > 0:
                for i in range(min(len(result.boxes), top_k)):
                    conf = float(result.boxes.conf[i].cpu().numpy())
                    cls_id = int(result.boxes.cls[i].cpu().numpy())
                    predictions.append({
                        'brand': normalize_brand_name(model.names[cls_id]),
                        'confidence': conf,
                        'source': model_name,
                    })
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                break

        del results
        return predictions[:top_k]
    except Exception as e:
        logger.error(f"{model_name} prediction failed: {e}")
        return []


def detect_brand(vehicle_crop):
    all_predictions = {}

    if brand_model_logo is not None:
        for p in get_model_predictions(brand_model_logo, vehicle_crop, 'logo'):
            brand = p['brand']
            if brand not in all_predictions:
                all_predictions[brand] = {'logo': 0, 'unified': 0}
            all_predictions[brand]['logo'] = max(all_predictions[brand]['logo'], p['confidence'])

    if brand_model_unified is not None:
        for p in get_model_predictions(brand_model_unified, vehicle_crop, 'unified'):
            brand = p['brand']
            if brand not in all_predictions:
                all_predictions[brand] = {'logo': 0, 'unified': 0}
            all_predictions[brand]['unified'] = max(all_predictions[brand]['unified'], p['confidence'])

    if not all_predictions:
        return "Unknown"

    final_scores = {}
    for brand, scores in all_predictions.items():
        logo = scores['logo']
        unified = scores['unified']

        if brand in specialist_models and (logo > 0.1 or unified > 0.1):
            try:
                spec_results = specialist_models[brand](vehicle_crop, verbose=False)
                for r in spec_results:
                    if r.probs is not None:
                        spec_conf = float(r.probs.top1conf.cpu().numpy())
                        if spec_conf > 0.7:
                            final_scores[brand] = spec_conf * 1.3
                            break
                del spec_results
            except Exception:
                pass
            if brand in final_scores:
                continue

        if logo > 0.3 and unified > 0.2:
            final_scores[brand] = min(logo * 1.4, 1.0)
        elif logo > 0.3:
            final_scores[brand] = min(logo * 1.3, 1.0)
        elif logo > 0 and unified > 0:
            final_scores[brand] = (logo * 1.2 + unified * 0.8) / 2
        elif logo > 0:
            final_scores[brand] = logo * 1.1
        elif unified > 0:
            final_scores[brand] = unified * 0.9

    if not final_scores:
        return "Unknown"

    sorted_brands = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_brands) >= 2:
        top_brand, top_score = sorted_brands[0]
        second_brand, second_score = sorted_brands[1]
        if top_brand in CHINESE_BRANDS and second_brand in UAE_COMMON_BRANDS:
            if top_score - second_score < 0.15:
                final_scores[second_brand] = second_score * 1.3
                sorted_brands = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_brands[0][0]


def detect_color(vehicle_crop):
    try:
        h, w = vehicle_crop.shape[:2]
        body = vehicle_crop[int(h * 0.2):int(h * 0.7), int(w * 0.15):int(w * 0.85)]
        if body.size == 0:
            body = vehicle_crop

        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
        total = body.shape[0] * body.shape[1]

        color_scores = {}
        for name, info in CAR_COLOR_RANGES.items():
            mask = cv2.inRange(hsv, info['lower'], info['upper'])
            pct = (cv2.countNonZero(mask) / total) * 100
            target = info.get('merge_with', name)
            color_scores[target] = color_scores.get(target, 0) + pct

        if not color_scores:
            return "Gray"

        top_color = max(color_scores, key=color_scores.get)
        if color_scores[top_color] < 2.0:
            gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if brightness > 180:
                return "White"
            if brightness > 120:
                return "Silver"
            if brightness > 60:
                return "Gray"
            return "Black"

        return top_color
    except Exception:
        return "Gray"


def clean_uae_plate(text):
    if not text:
        return ""

    text = text.upper().strip()
    text = ''.join(c for c in text if c.isalnum())

    alpha_chars = []
    digit_chars = []
    hit_digit = False

    for c in text:
        if c.isdigit():
            hit_digit = True
            digit_chars.append(c)
        elif c.isalpha() and not hit_digit:
            alpha_chars.append(c)
        elif c.isalpha() and hit_digit:
            if c in CHAR_TO_DIGIT:
                digit_chars.append(CHAR_TO_DIGIT[c])

    alpha_part = ''.join(alpha_chars)
    digit_part = ''.join(digit_chars)

    if not digit_part:
        for c in text:
            if c.isdigit():
                digit_part += c
            elif c in CHAR_TO_DIGIT:
                digit_part += CHAR_TO_DIGIT[c]

    if digit_part and len(digit_part) > 1:
        digit_part = digit_part.lstrip('0') or '0'

    if len(digit_part) > 5:
        digit_part = digit_part[-5:]

    if len(alpha_part) > 2:
        alpha_part = alpha_part[:2]

    if alpha_part and digit_part:
        return f"{alpha_part} {digit_part}"
    if digit_part:
        return digit_part
    return ""


def validate_uae_plate(text):
    if not text:
        return 0.0

    parts = text.split()
    score = 0.3

    if len(parts) == 2:
        alpha, digits = parts[0], parts[1]
        if alpha in UAE_PLATE_CODES:
            score += 0.3
        elif len(alpha) <= 2 and alpha.isalpha():
            score += 0.15
        if digits.isdigit() and 1 <= len(digits) <= 5:
            score += 0.3
        if 2 <= len(digits) <= 5:
            score += 0.1
    elif len(parts) == 1 and parts[0].isdigit():
        digits = parts[0]
        if 1 <= len(digits) <= 5:
            score += 0.5
        if 2 <= len(digits) <= 5:
            score += 0.2

    return min(1.0, score)


def detect_plate(vehicle_crop):
    try:
        h, w = vehicle_crop.shape[:2]

        regions = [
            vehicle_crop[h // 2:, :],
            vehicle_crop,
        ]

        all_candidates = []

        for idx, region in enumerate(regions):
            if region.size == 0:
                continue

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ocr_inputs = [
                region,
                cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
            ]

            for ocr_input in ocr_inputs:
                try:
                    results = ocr_reader.readtext(
                        ocr_input,
                        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        paragraph=False,
                        min_size=10,
                        text_threshold=0.4,
                        low_text=0.3,
                    )
                    for det in results:
                        raw_text = det[1]
                        conf = det[2]
                        if conf < 0.15:
                            continue
                        cleaned = clean_uae_plate(raw_text)
                        if cleaned:
                            validity = validate_uae_plate(cleaned)
                            all_candidates.append({
                                'text': cleaned,
                                'confidence': conf,
                                'validity': validity,
                                'score': conf * 0.5 + validity * 0.5,
                            })
                except Exception:
                    continue

        if not all_candidates:
            return None

        text_counts = Counter(c['text'] for c in all_candidates)
        for c in all_candidates:
            freq = text_counts[c['text']]
            if freq > 1:
                c['score'] += 0.1 * min(freq, 3)

        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        best = all_candidates[0]
        if best['score'] < 0.2:
            return None

        logger.info(f"Plate: {best['text']} (score={best['score']:.2f})")
        return best['text']

    except Exception as e:
        logger.error(f"Plate detection error: {e}")
        return None


def process_image(image_path):
    logger.info(f"Processing: {image_path}")

    if not os.path.exists(image_path):
        return {
            "status": "FAILED",
            "error": "Image not found",
            "plateNo": None,
            "vehicleColor": None,
            "vehicleMake": None,
        }

    image = cv2.imread(image_path)
    if image is None:
        return {
            "status": "FAILED",
            "error": "Failed to load image",
            "plateNo": None,
            "vehicleColor": None,
            "vehicleMake": None,
        }

    img_h, img_w = image.shape[:2]
    logger.info(f"Image: {img_w}x{img_h}")

    detections = yolo_model(image, verbose=False, classes=[2, 5, 7], conf=0.2)

    best_conf = 0
    best_box = None

    for det in detections:
        if det.boxes is None:
            continue
        for box in det.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if (x2 - x1) < 60 or (y2 - y1) < 60:
                continue
            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)

    del detections

    if best_box is None:
        if img_w > 200 and img_h > 200:
            vehicle_crop = image
            logger.info("No vehicle box, using full image")
        else:
            cleanup_memory()
            return {
                "status": "FAILED",
                "error": "No vehicle detected",
                "plateNo": None,
                "vehicleColor": None,
                "vehicleMake": None,
            }
    else:
        x1, y1, x2, y2 = best_box
        vehicle_crop = image[y1:y2, x1:x2]
        logger.info(f"Vehicle: {x2 - x1}x{y2 - y1} conf={best_conf:.2f}")

    color_crop = image[best_box[1]:best_box[3], best_box[0]:best_box[2]] if best_box else image

    make = detect_brand(vehicle_crop)
    color = detect_color(color_crop)
    plate = detect_plate(vehicle_crop)

    del vehicle_crop, color_crop, image
    cleanup_memory()

    logger.info(f"Result: make={make}, color={color}, plate={plate}")

    return {
        "status": "SUCCESS",
        "plateNo": plate,
        "vehicleColor": color,
        "vehicleMake": make,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    result = process_image(sys.argv[1])
    print(json.dumps(result, indent=2))