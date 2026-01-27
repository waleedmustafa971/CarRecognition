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
from src.main import detect_car_and_plate

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

request_counter = 0
server_start_time = time.time()
process = psutil.Process(os.getpid())
request_lock = threading.Lock()

MEMORY_WARNING_MB = 2000
MEMORY_CRITICAL_MB = 3000
FORCE_GC_EVERY_N_REQUESTS = 5
AGGRESSIVE_GC_THRESHOLD_MB = 1500


def get_memory_usage():
    return process.memory_info().rss / 1024 / 1024


def force_memory_cleanup():
    gc.collect()
    gc.collect()
    gc.collect()
    
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_memory_usage(context=""):
    memory_mb = get_memory_usage()
    
    if memory_mb > MEMORY_CRITICAL_MB:
        logger.error(f"🚨 CRITICAL MEMORY: {memory_mb:.2f} MB {context}")
        force_memory_cleanup()
        new_memory = get_memory_usage()
        logger.info(f"   After emergency cleanup: {new_memory:.2f} MB")
    elif memory_mb > MEMORY_WARNING_MB:
        logger.warning(f"⚠️ HIGH MEMORY: {memory_mb:.2f} MB {context}")
    else:
        logger.info(f"Memory usage {context}: {memory_mb:.2f} MB")
    
    return memory_mb


@app.before_request
def log_request_info():
    global request_counter
    
    with request_lock:
        request_counter += 1
        current_count = request_counter
    
    request.start_time = time.time()
    request.request_id = f"REQ-{current_count:05d}"
    request.start_memory = get_memory_usage()
    
    logger.info(f"NEW REQUEST")
    logger.info(f"Request ID: {request.request_id}")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Path: {request.path}")
    logger.info(f"Remote Address: {request.remote_addr}")
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    if 'Mobile' in user_agent or 'Android' in user_agent or 'iPhone' in user_agent:
        logger.info("CLIENT TYPE: MOBILE")
    elif 'Postman' in user_agent:
        logger.info("CLIENT TYPE: POSTMAN")
    elif 'curl' in user_agent.lower():
        logger.info("CLIENT TYPE: CURL")
    else:
        logger.info("CLIENT TYPE: DESKTOP/OTHER")
    
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Content-Length: {request.content_length} bytes" if request.content_length else "Content-Length: Not specified")
    
    if request.files:
        logger.info("FILES RECEIVED:")
        for file_key, file_obj in request.files.items():
            file_obj.seek(0, os.SEEK_END)
            file_size = file_obj.tell()
            file_obj.seek(0)
            logger.info(f"Key: {file_key}")
            logger.info(f"Filename: {file_obj.filename}")
            logger.info(f"Content-Type: {file_obj.content_type}")
            logger.info(f"Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    if request.form:
        logger.info("FORM DATA:")
        for key, value in request.form.items():
            if len(str(value)) > 200:
                value = str(value)[:200] + "... (truncated)"
            logger.info(f"{key}: {value}")
    
    if request.is_json:
        logger.info("JSON PAYLOAD:")
        try:
            json_data = request.get_json()
            for key in json_data.keys():
                if key == 'image_base64':
                    value_len = len(str(json_data[key]))
                    logger.info(f"{key}: <base64 data, {value_len} chars>")
                else:
                    logger.info(f"{key}: {json_data[key]}")
        except Exception as e:
            logger.warning(f"Could not parse JSON: {e}")
    
    log_memory_usage("at request start")


@app.after_request
def log_response_info(response):
    global request_counter
    
    if hasattr(request, 'start_time'):
        processing_time = time.time() - request.start_time
        logger.info(f"Processing time: {processing_time:.3f} seconds")
    
    if hasattr(request, 'start_memory'):
        current_memory = get_memory_usage()
        memory_delta = current_memory - request.start_memory
        logger.info(f"Memory delta: {memory_delta:+.2f} MB")
        
        if memory_delta > 100:
            logger.warning(f"⚠️ Large memory increase detected: +{memory_delta:.2f} MB")
    
    logger.info("RESPONSE:")
    logger.info(f"Status: {response.status_code} {response.status}")
    logger.info(f"Content-Type: {response.content_type}")
    logger.info(f"Content-Length: {response.content_length} bytes" if response.content_length else "Content-Length: Not calculated")
    
    if response.data and response.content_type and 'json' in response.content_type:
        try:
            response_json = json.loads(response.data.decode('utf-8'))
            logger.info(f"Response JSON: {json.dumps(response_json, indent=4)}")
        except:
            response_preview = response.data.decode('utf-8', errors='ignore')[:500]
            logger.info(f"Response preview: {response_preview}")
    
    if hasattr(request, 'request_id'):
        logger.info(f"Request {request.request_id} completed")
    
    if request_counter % FORCE_GC_EVERY_N_REQUESTS == 0:
        logger.info(f"🧹 Periodic cleanup (every {FORCE_GC_EVERY_N_REQUESTS} requests)")
        force_memory_cleanup()
    else:
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    current_memory = get_memory_usage()
    if current_memory > AGGRESSIVE_GC_THRESHOLD_MB:
        logger.info(f"🧹 Aggressive cleanup triggered (memory: {current_memory:.2f} MB)")
        force_memory_cleanup()
    
    log_memory_usage("after cleanup")
    
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error("UNHANDLED EXCEPTION")
    logger.error(f"Exception type: {type(e).__name__}")
    logger.error(f"Exception message: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    force_memory_cleanup()
    
    return jsonify({
        "status": "ERROR",
        "vehicleMake": "Unknown",
        "plateNo": "",
        "vehicleColor": "Unknown",
        "error": f"Server error: {str(e)}"
    }), 500


def ensure_json_serializable(obj):
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ensure_json_serializable(item) for item in obj]
        else:
            return obj
    except Exception as e:
        logger.error(f"Error in ensure_json_serializable: {e}")
        return str(obj)


def convert_to_simple_format(detection_result):
    try:
        logger.info("Converting to simple format")
        
        if not detection_result:
            logger.warning("No detection results - returning default response")
            return {
                "status": "SUCCESS",
                "vehicleMake": "Unknown",
                "plateNo": "",
                "vehicleColor": "Unknown"
            }
        
        logger.debug(f"Raw detection data: {json.dumps(detection_result, default=str, indent=2)}")
        
        car_make_model = detection_result.get('car_make_model', [])
        colors = detection_result.get('colors', [])
        plate_number = detection_result.get('plate_number', '')
        
        vehicle_make = "Unknown"
        if car_make_model and len(car_make_model) > 0:
            vehicle_make = car_make_model[0].get('make', 'Unknown')
            logger.info(f"Top vehicle make: {vehicle_make} (score: {car_make_model[0].get('score', 0):.3f})")
            
            if len(car_make_model) > 1:
                logger.info("Other brand predictions:")
                for i, brand in enumerate(car_make_model[1:4], 2):
                    logger.info(f"  #{i}: {brand.get('make')} (score: {brand.get('score', 0):.3f})")
        else:
            logger.warning("No vehicle make detected")
        
        vehicle_color = "Unknown"
        if colors and len(colors) > 0:
            vehicle_color = colors[0].get('color', 'Unknown')
            if vehicle_color != "Unknown":
                logger.info(f"Top vehicle color: {vehicle_color} ({colors[0].get('percentage', 0):.1f}%)")
                
                if len(colors) > 1:
                    logger.info("Other colors detected:")
                    for i, color in enumerate(colors[1:], 2):
                        logger.info(f"  #{i}: {color.get('color')} ({color.get('percentage', 0):.1f}%)")
            else:
                logger.warning("Color detection returned Unknown")
        else:
            logger.warning("No vehicle color detected")
        
        plate_no = ""
        if plate_number:
            plate_no = str(plate_number).upper()
            logger.info(f"Plate number: {plate_no}")
        else:
            logger.warning("No plate detected")
        
        result = {
            "status": "SUCCESS",
            "vehicleMake": str(vehicle_make),
            "plateNo": str(plate_no),
            "vehicleColor": str(vehicle_color.capitalize())
        }
        
        logger.info("FINAL RESULT:")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Make: {result['vehicleMake']}")
        logger.info(f"Plate: {result['plateNo']}")
        logger.info(f"Color: {result['vehicleColor']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error converting to simple format: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "status": "ERROR",
            "vehicleMake": "Unknown",
            "plateNo": "",
            "vehicleColor": "Unknown",
            "error": f"Format conversion error: {str(e)}"
        }


def validate_image_file(file):
    try:
        if not file.filename:
            return False, "No filename provided"
        
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return False, f"Invalid file type: {ext}. Allowed: {allowed_extensions}"
        
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        max_size = 16 * 1024 * 1024
        if file_size == 0:
            return False, "File is empty"
        if file_size > max_size:
            return False, f"File too large: {file_size} bytes (max: {max_size})"
        
        logger.info(f"File validation passed: {file.filename} ({file_size:,} bytes)")
        return True, None
        
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return False, f"Validation error: {str(e)}"


@app.route('/detect_car', methods=['POST'])
def detect_car_api():
    start_time = time.time()
    temp_path = None
    image = None
    nparr = None
    image_data = None
    
    try:
        logger.info("STARTING CAR DETECTION")
        log_memory_usage("before processing")
        
        logger.info("REQUEST DIAGNOSTIC:")
        logger.info(f"request.files keys: {list(request.files.keys())}")
        logger.info(f"request.is_json: {request.is_json}")
        if request.is_json:
            logger.info(f"JSON keys: {list(request.json.keys()) if request.json else 'None'}")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        
        if 'image' not in request.files and not (request.is_json and 'image_base64' in request.json):
            logger.error("NO IMAGE PROVIDED IN REQUEST")
            return jsonify({
                "status": "ERROR",
                "vehicleMake": "Unknown",
                "plateNo": "",
                "vehicleColor": "Unknown",
                "error": "No image provided"
            }), 400
        
        if 'image' in request.files:
            logger.info("Processing file upload")
            file = request.files['image']
            
            is_valid, error_msg = validate_image_file(file)
            if not is_valid:
                logger.error(f"File validation failed: {error_msg}")
                return jsonify({
                    "status": "ERROR",
                    "vehicleMake": "Unknown",
                    "plateNo": "",
                    "vehicleColor": "Unknown",
                    "error": error_msg
                }), 400
            
            unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{file.filename}"
            temp_path = os.path.join(log_dir, f"temp_{unique_filename}")
            
            logger.info(f"Saving to: {temp_path}")
            file.save(temp_path)
            logger.info(f"File saved successfully")
            
            if not os.path.exists(temp_path):
                logger.error(f"File was not saved: {temp_path}")
                return jsonify({
                    "status": "ERROR",
                    "vehicleMake": "Unknown",
                    "plateNo": "",
                    "vehicleColor": "Unknown",
                    "error": "Failed to save uploaded file"
                }), 500
            
            saved_size = os.path.getsize(temp_path)
            logger.info(f"Verified saved file: {saved_size} bytes")
            
            if saved_size == 0:
                logger.error("Saved file is EMPTY (0 bytes)")
                return jsonify({
                    "status": "ERROR",
                    "vehicleMake": "Unknown",
                    "plateNo": "",
                    "vehicleColor": "Unknown",
                    "error": "Uploaded file is empty"
                }), 400
            
        elif request.is_json and 'image_base64' in request.json:
            logger.info("Processing base64 image")
            base64_string = request.json['image_base64']
            
            if not base64_string:
                logger.error("Base64 string is EMPTY")
                return jsonify({
                    "status": "ERROR",
                    "vehicleMake": "Unknown",
                    "plateNo": "",
                    "vehicleColor": "Unknown",
                    "error": "Base64 image data is empty"
                }), 400
            
            if ',' in base64_string:
                logger.debug("Removing data URL prefix")
                base64_string = base64_string.split(',')[1]
            
            logger.info(f"Base64 string length: {len(base64_string)} characters")
            
            if len(base64_string) < 100:
                logger.error(f"Base64 string too short: {len(base64_string)} chars")
                return jsonify({
                    "status": "ERROR",
                    "vehicleMake": "Unknown",
                    "plateNo": "",
                    "vehicleColor": "Unknown",
                    "error": "Base64 image data is too short"
                }), 400
            
            try:
                logger.debug("Decoding base64...")
                image_data = base64.b64decode(base64_string)
                logger.info(f"Decoded {len(image_data)} bytes")
                
                logger.debug("Converting to numpy array...")
                nparr = np.frombuffer(image_data, np.uint8)
                
                logger.debug("Decoding image with OpenCV...")
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    logger.error("OpenCV failed to decode image")
                    return jsonify({
                        "status": "ERROR",
                        "vehicleMake": "Unknown",
                        "plateNo": "",
                        "vehicleColor": "Unknown",
                        "error": "Invalid image data"
                    }), 400
                
                logger.info(f"Image decoded: {image.shape[1]}x{image.shape[0]} pixels")
                
                unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_base64.jpg"
                temp_path = os.path.join(log_dir, f"temp_{unique_filename}")
                
                logger.info(f"Saving decoded image to: {temp_path}")
                cv2.imwrite(temp_path, image)
                logger.info("Base64 image saved")
                
            except base64.binascii.Error as e:
                logger.error(f"Base64 decoding failed: {e}")
                return jsonify({
                    "status": "ERROR",
                    "vehicleMake": "Unknown",
                    "plateNo": "",
                    "vehicleColor": "Unknown",
                    "error": "Invalid base64 encoding"
                }), 400
            except Exception as e:
                logger.error(f"Base64 processing failed: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    "status": "ERROR",
                    "vehicleMake": "Unknown",
                    "plateNo": "",
                    "vehicleColor": "Unknown",
                    "error": f"Failed to process image: {str(e)}"
                }), 400
            finally:
                if image is not None:
                    del image
                    image = None
                if nparr is not None:
                    del nparr
                    nparr = None
                if image_data is not None:
                    del image_data
                    image_data = None
        
        if not os.path.exists(temp_path):
            logger.error(f"Temp file does not exist: {temp_path}")
            return jsonify({
                "status": "ERROR",
                "vehicleMake": "Unknown",
                "plateNo": "",
                "vehicleColor": "Unknown",
                "error": "Failed to save image file"
            }), 500
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"Temp file verified: {file_size:,} bytes")
        
        log_memory_usage("before detection")
        
        try:
            logger.info("STARTING YOLO DETECTION")
            detection_start = time.time()
            
            result = detect_car_and_plate(temp_path)
            
            detection_end = time.time()
            detection_time = detection_end - detection_start
            
            logger.info(f"DETECTION COMPLETED in {detection_time:.3f} seconds")
            
        except Exception as det_error:
            logger.error("DETECTION FAILED")
            logger.error(f"Error: {str(det_error)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "ERROR",
                "vehicleMake": "Unknown",
                "plateNo": "",
                "vehicleColor": "Unknown",
                "error": f"Detection failed: {str(det_error)}"
            }), 500
        
        log_memory_usage("after detection")
        
        logger.info(f"Detection returned {len(result) if result else 0} result(s)")
        
        if not result or len(result) == 0:
            logger.warning("EMPTY DETECTION RESULT")
            logger.warning("No vehicles detected")
        
        if result:
            result = ensure_json_serializable(result)
            logger.debug(f"Cleaned result: {json.dumps(result, indent=2)}")
        else:
            logger.warning("Empty detection result")
            result = []
        
        simple_result = convert_to_simple_format(result)
        
        total_time = time.time() - start_time
        logger.info("PERFORMANCE METRICS")
        logger.info(f"Total time: {total_time:.3f}s")
        
        return jsonify(simple_result), 200
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error("FATAL ERROR IN DETECTION API")
        logger.error(f"Error after {error_time:.3f} seconds")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_response = {
            "status": "ERROR",
            "vehicleMake": "Unknown",
            "plateNo": "",
            "vehicleColor": "Unknown",
            "error": f"{type(e).__name__}: {str(e)}"
        }
        
        return jsonify(error_response), 500
        
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
        
        if image is not None:
            del image
        if nparr is not None:
            del nparr
        if image_data is not None:
            del image_data
        
        force_memory_cleanup()
        log_memory_usage("final cleanup")


@app.route('/health', methods=['GET'])
def health_check():
    uptime = time.time() - server_start_time
    memory_mb = get_memory_usage()
    
    status = "healthy"
    if memory_mb > MEMORY_CRITICAL_MB:
        status = "critical"
    elif memory_mb > MEMORY_WARNING_MB:
        status = "warning"
    
    logger.info("Health check requested")
    return jsonify({
        "status": status,
        "service": "Car Detection API",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime,
        "uptime_hours": round(uptime / 3600, 2),
        "requests_processed": request_counter,
        "memory_mb": round(memory_mb, 2),
        "memory_warning_threshold_mb": MEMORY_WARNING_MB,
        "memory_critical_threshold_mb": MEMORY_CRITICAL_MB,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "torch_available": HAS_TORCH,
        "cuda_available": HAS_TORCH and torch.cuda.is_available() if HAS_TORCH else False
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    memory_info = process.memory_info()
    
    return jsonify({
        "uptime_seconds": time.time() - server_start_time,
        "requests_total": request_counter,
        "memory": {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": process.memory_percent(),
            "warning_threshold_mb": MEMORY_WARNING_MB,
            "critical_threshold_mb": MEMORY_CRITICAL_MB
        },
        "cpu": {
            "percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads()
        },
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "memory_percent": psutil.virtual_memory().percent
        },
        "gc": {
            "force_gc_every_n_requests": FORCE_GC_EVERY_N_REQUESTS,
            "aggressive_gc_threshold_mb": AGGRESSIVE_GC_THRESHOLD_MB
        }
    })


@app.route('/force_gc', methods=['POST'])
def force_gc():
    logger.info("Manual garbage collection requested")
    before_memory = get_memory_usage()
    
    force_memory_cleanup()
    
    after_memory = get_memory_usage()
    freed = before_memory - after_memory
    
    return jsonify({
        "status": "success",
        "before_mb": round(before_memory, 2),
        "after_mb": round(after_memory, 2),
        "freed_mb": round(freed, 2)
    })


@app.route('/test_mobile', methods=['POST'])
def test_mobile_connection():
    logger.info("Mobile connection test requested")
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    client_ip = request.remote_addr
    
    response_data = {
        "status": "success",
        "message": "Mobile connection test successful",
        "client_ip": client_ip,
        "user_agent": user_agent,
        "timestamp": datetime.now().isoformat(),
        "server_time": time.time()
    }
    
    logger.info(f"Test response: {json.dumps(response_data, indent=2)}")
    return jsonify(response_data)


@app.route('/test_image_upload', methods=['POST'])
def test_image_upload():
    logger.info("IMAGE UPLOAD TEST")
    
    diagnostic = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "request_info": {
            "content_type": request.content_type,
            "content_length": request.content_length,
            "is_json": request.is_json,
            "files_keys": list(request.files.keys()),
            "form_keys": list(request.form.keys()) if request.form else [],
        },
        "validation": {}
    }
    
    if 'image' in request.files:
        file = request.files['image']
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        diagnostic["validation"]["file_upload"] = {
            "found": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 2),
            "is_empty": file_size == 0,
            "status": "Valid" if file_size > 0 else "Empty file"
        }
        logger.info(f"File upload detected: {file.filename} ({file_size} bytes)")
    else:
        diagnostic["validation"]["file_upload"] = {
            "found": False,
            "status": "No 'image' file in request"
        }
        logger.warning("No 'image' file found")
    
    if request.is_json and 'image_base64' in request.json:
        base64_string = request.json['image_base64']
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        diagnostic["validation"]["base64_upload"] = {
            "found": True,
            "length": len(base64_string),
            "is_empty": len(base64_string) == 0,
            "status": "Valid" if len(base64_string) > 100 else "Too short or empty"
        }
        logger.info(f"Base64 detected: {len(base64_string)} characters")
    else:
        diagnostic["validation"]["base64_upload"] = {
            "found": False,
            "status": "No 'image_base64' in JSON"
        }
        logger.warning("No 'image_base64' in JSON")
    
    has_valid_image = (
        (diagnostic["validation"].get("file_upload", {}).get("found") and 
         diagnostic["validation"]["file_upload"]["size_bytes"] > 0) or
        (diagnostic["validation"].get("base64_upload", {}).get("found") and 
         diagnostic["validation"]["base64_upload"]["length"] > 100)
    )
    
    diagnostic["validation"]["overall"] = {
        "has_valid_image": has_valid_image,
        "ready_for_detection": has_valid_image,
        "message": "Ready for /detect_car" if has_valid_image else "Image upload failed"
    }
    
    logger.info(f"Test result: {'PASS' if has_valid_image else 'FAIL'}")
    
    return jsonify(diagnostic)


if __name__ == '__main__':
    import socket
    
    def is_port_available(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False
    
    preferred_ports = [8080, 5000, 8000, 3000]
    selected_port = None
    
    for port in preferred_ports:
        if is_port_available(port):
            selected_port = port
            logger.info(f"Port {port} is available")
            break
        else:
            logger.warning(f"Port {port} is already in use")
    
    if selected_port is None:
        logger.error("FATAL ERROR: No available ports found")
        import sys
        sys.exit(1)
    
    logger.info("STARTING CAR DETECTION API SERVER")
    logger.info("=" * 80)
    logger.info("Server Configuration:")
    logger.info(f"  Host: 0.0.0.0 (all interfaces)")
    logger.info(f"  Port: {selected_port}")
    logger.info(f"  Public IP: 18.210.19.225")
    logger.info(f"  Threads: 2 (reduced for memory)")
    logger.info(f"  CORS: Enabled")
    logger.info(f"  Log directory: {os.path.abspath(log_dir)}")
    logger.info(f"  Memory warning: {MEMORY_WARNING_MB} MB")
    logger.info(f"  Memory critical: {MEMORY_CRITICAL_MB} MB")
    logger.info(f"  Force GC every: {FORCE_GC_EVERY_N_REQUESTS} requests")
    logger.info(f"  Torch available: {HAS_TORCH}")
    if HAS_TORCH:
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    logger.info("=" * 80)
    logger.info("Server URLs:")
    logger.info(f"  Local:   http://localhost:{selected_port}")
    logger.info(f"  Network: http://172.31.30.5:{selected_port}")
    logger.info(f"  Public: 18.210.19.225:{selected_port}")
    logger.info("=" * 80)
    logger.info("Endpoints:")
    logger.info(f"  POST /detect_car     - Detect car from image")
    logger.info(f"  GET  /health         - Health check with memory status")
    logger.info(f"  GET  /metrics        - Detailed system metrics")
    logger.info(f"  POST /force_gc       - Force garbage collection")
    logger.info(f"  POST /test_mobile    - Test mobile connection")
    logger.info(f"  POST /test_image_upload - Test image upload")
    logger.info("=" * 80)
    logger.info("Ready to accept requests")
    logger.info("=" * 80)
    
    from waitress import serve
    serve(app, host='0.0.0.0', port=selected_port, threads=2)