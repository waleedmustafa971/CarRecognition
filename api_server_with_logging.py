import logging
import os
import time
import traceback
import gc
import psutil
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import requests as http_requests
from src.main import process_image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = logging.FileHandler(os.path.join(log_dir, f'api_{datetime.now().strftime("%Y%m%d")}.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

FORWARD_URL = os.environ.get('FORWARD_URL', 'http://86.96.193.137:8085/evaletFusion/api/v1/camera/vehicle-entry')
FORWARD_TIMEOUT = int(os.environ.get('FORWARD_TIMEOUT', '30'))
FORWARD_ENABLED = os.environ.get('FORWARD_ENABLED', 'true').lower() == 'true'
FORWARD_TOKEN = os.environ.get(
    'FORWARD_TOKEN',
    'eyJhbGciOiJIUzI1NiJ9.eyJsb2NhdGlvbklkIjoxNywicm9sZXMiOlsiTE9DQVRJT05fQURNSU4iXSwidGVuYW50SWQiOjE3LCJpc1JlZnJlc2hUb2tlbiI6ZmFsc2UsInVzZXJJZCI6MjksInN1YiI6IjI5IiwiaWF0IjoxNzcwODcxMDc5LCJleHAiOjE3NzE0NzU4Nzl9.TllLy23C7fXhBBqFXwmFX6yKrAZttxPLXRKPDxD8d24'
)
BAY_NO = os.environ.get('BAY_NO', 'A12')

request_counter = 0
server_start_time = time.time()
process_info = psutil.Process(os.getpid())
request_lock = threading.Lock()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024


def get_memory_mb():
    return process_info.memory_info().rss / 1024 / 1024


def force_cleanup():
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def forward_to_external(result_data, image_path):
    if not FORWARD_ENABLED:
        return None

    file_handle = None
    try:
        form_data = {
            'plateNo': result_data.get('plateNo', ''),
            'vehicleMake': result_data.get('vehicleMake', 'Unknown'),
            'vehicleColor': result_data.get('vehicleColor', 'Unknown'),
            'cameraCaptureTime': result_data.get('cameraCaptureTime', ''),
            'bayNo': BAY_NO,
        }

        headers = {}
        if FORWARD_TOKEN:
            headers['Authorization'] = f'Bearer {FORWARD_TOKEN}'

        files = None
        if image_path and os.path.exists(image_path):
            ext = os.path.splitext(image_path)[1].lower()
            content_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}
            file_handle = open(image_path, 'rb')
            files = {'vehicleImage': (os.path.basename(image_path), file_handle, content_types.get(ext, 'image/jpeg'))}

        logger.info(f"Forwarding to {FORWARD_URL}: {json.dumps(form_data)}")

        response = http_requests.post(
            FORWARD_URL, data=form_data, files=files,
            headers=headers, timeout=FORWARD_TIMEOUT
        )

        logger.info(f"Forward response: {response.status_code}")
        return {
            'status_code': response.status_code,
            'success': 200 <= response.status_code < 300,
        }

    except http_requests.exceptions.Timeout:
        logger.error(f"Forward timeout after {FORWARD_TIMEOUT}s")
        return {'status_code': 0, 'success': False, 'error': 'timeout'}
    except http_requests.exceptions.ConnectionError as e:
        logger.error(f"Forward connection failed: {e}")
        return {'status_code': 0, 'success': False, 'error': 'connection_failed'}
    except Exception as e:
        logger.error(f"Forward error: {e}")
        return {'status_code': 0, 'success': False, 'error': str(e)}
    finally:
        if file_handle:
            file_handle.close()


@app.before_request
def log_request():
    global request_counter
    with request_lock:
        request_counter += 1
    request.start_time = time.time()
    request.request_id = f"REQ-{request_counter:05d}"


@app.after_request
def log_response(response):
    if hasattr(request, 'start_time'):
        elapsed = time.time() - request.start_time
        logger.info(f"{request.request_id} {request.method} {request.path} -> {response.status_code} ({elapsed:.2f}s)")

    if request_counter % 5 == 0:
        force_cleanup()

    return response


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled: {type(e).__name__}: {e}")
    logger.error(traceback.format_exc())
    force_cleanup()
    return jsonify({
        "status": "ERROR",
        "plateNo": "",
        "vehicleMake": "Unknown",
        "vehicleColor": "Unknown",
        "error": str(e),
    }), 500


def save_upload_to_temp(file=None, base64_str=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    if file is not None:
        if not file.filename:
            return None, "No filename"
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return None, f"Invalid type: {ext}"

        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size == 0:
            return None, "Empty file"
        if size > MAX_FILE_SIZE:
            return None, "File too large"

        path = os.path.join(log_dir, f"temp_{timestamp}{ext}")
        file.save(path)
        return path, None

    if base64_str is not None:
        if len(str(base64_str)) < 100:
            return None, "Base64 too short"

        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        try:
            image_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return None, "Invalid image data"

            path = os.path.join(log_dir, f"temp_{timestamp}.jpg")
            cv2.imwrite(path, image)
            del image, nparr, image_data
            return path, None
        except Exception as e:
            return None, f"Base64 decode failed: {e}"

    return None, "No image provided"


@app.route('/detect_car', methods=['POST'])
def detect_car_api():
    temp_path = None
    try:
        capture_time = None
        if request.form:
            capture_time = request.form.get('cameraCaptureTime')
        elif request.is_json and request.json:
            capture_time = request.json.get('cameraCaptureTime')
        if not capture_time:
            capture_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        if 'image' in request.files:
            temp_path, error = save_upload_to_temp(file=request.files['image'])
        elif request.is_json and request.json and 'image_base64' in request.json:
            temp_path, error = save_upload_to_temp(base64_str=request.json['image_base64'])
        else:
            return jsonify({
                "status": "ERROR",
                "plateNo": "",
                "vehicleMake": "Unknown",
                "vehicleColor": "Unknown",
                "error": "No image provided. Send 'image' as file or 'image_base64' in JSON.",
            }), 400

        if temp_path is None:
            return jsonify({
                "status": "ERROR",
                "plateNo": "",
                "vehicleMake": "Unknown",
                "vehicleColor": "Unknown",
                "error": error,
            }), 400

        result = process_image(temp_path)

        result['cameraCaptureTime'] = capture_time
        result['bayNo'] = BAY_NO

        if not result.get('plateNo'):
            result['plateNo'] = ''

        forward_result = forward_to_external(result, temp_path)
        if forward_result:
            result['forward_status'] = 'success' if forward_result.get('success') else 'failed'
            result['forward_status_code'] = forward_result.get('status_code', 0)
            if not forward_result.get('success'):
                result['forward_error'] = forward_result.get('error', 'unknown')

        logger.info(f"Response: {json.dumps(result)}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Fatal: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "ERROR",
            "plateNo": "",
            "vehicleMake": "Unknown",
            "vehicleColor": "Unknown",
            "error": str(e),
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        force_cleanup()


@app.route('/health', methods=['GET'])
def health():
    memory_mb = get_memory_mb()
    status = "critical" if memory_mb > 3000 else "warning" if memory_mb > 2000 else "healthy"
    return jsonify({
        "status": status,
        "service": "Car Detection API v8.0",
        "uptime_seconds": time.time() - server_start_time,
        "requests_processed": request_counter,
        "memory_mb": round(memory_mb, 2),
        "forward_enabled": FORWARD_ENABLED,
        "bay_no": BAY_NO,
        "cuda_available": HAS_TORCH and torch.cuda.is_available() if HAS_TORCH else False,
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    mem = process_info.memory_info()
    return jsonify({
        "uptime_seconds": time.time() - server_start_time,
        "requests_total": request_counter,
        "memory_rss_mb": round(mem.rss / 1024 / 1024, 2),
        "cpu_percent": process_info.cpu_percent(interval=0.1),
        "forward_enabled": FORWARD_ENABLED,
        "bay_no": BAY_NO,
    })


@app.route('/config', methods=['GET', 'POST'])
def manage_config():
    global FORWARD_URL, FORWARD_ENABLED, FORWARD_TIMEOUT, FORWARD_TOKEN, BAY_NO

    if request.method == 'GET':
        return jsonify({
            "forward_url": FORWARD_URL,
            "forward_enabled": FORWARD_ENABLED,
            "forward_timeout": FORWARD_TIMEOUT,
            "bay_no": BAY_NO,
            "token_present": bool(FORWARD_TOKEN),
        })

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    if 'forward_url' in data:
        FORWARD_URL = data['forward_url']
    if 'forward_enabled' in data:
        FORWARD_ENABLED = bool(data['forward_enabled'])
    if 'forward_timeout' in data:
        FORWARD_TIMEOUT = int(data['forward_timeout'])
    if 'forward_token' in data:
        FORWARD_TOKEN = data['forward_token']
    if 'bay_no' in data:
        BAY_NO = str(data['bay_no'])

    logger.info(f"Config updated: url={FORWARD_URL}, enabled={FORWARD_ENABLED}, bay={BAY_NO}")
    return jsonify({"status": "updated", "forward_url": FORWARD_URL, "forward_enabled": FORWARD_ENABLED, "bay_no": BAY_NO})


@app.route('/force_gc', methods=['POST'])
def force_gc():
    before = get_memory_mb()
    force_cleanup()
    after = get_memory_mb()
    return jsonify({"before_mb": round(before, 2), "after_mb": round(after, 2), "freed_mb": round(before - after, 2)})


@app.route('/test_forward', methods=['POST'])
def test_forward():
    test_data = {
        'plateNo': 'TEST-0000',
        'vehicleMake': 'TEST',
        'vehicleColor': 'White',
        'cameraCaptureTime': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'bayNo': BAY_NO,
    }
    headers = {'Authorization': f'Bearer {FORWARD_TOKEN}'} if FORWARD_TOKEN else {}

    try:
        response = http_requests.post(FORWARD_URL, data=test_data, headers=headers, timeout=FORWARD_TIMEOUT)
        return jsonify({
            "status": "success" if 200 <= response.status_code < 300 else "failed",
            "response_code": response.status_code,
            "forward_url": FORWARD_URL,
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e), "forward_url": FORWARD_URL}), 502


if __name__ == '__main__':
    import socket

    selected_port = None
    for port in [8080, 5000, 8000]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
            selected_port = port
            break
        except OSError:
            continue

    if selected_port is None:
        logger.error("No available ports")
        exit(1)

    logger.info("=" * 60)
    logger.info(f"Car Detection API v8.0 | Port {selected_port} | Bay {BAY_NO}")
    logger.info(f"Forward: {FORWARD_URL} (enabled={FORWARD_ENABLED})")
    logger.info("=" * 60)

    from waitress import serve
    serve(app, host='0.0.0.0', port=selected_port, threads=2)