# api_server.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tempfile
import os
from multibrand_car_detection import MultibrandCarDetector
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the car detector
detector = MultibrandCarDetector()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect_car', methods=['POST'])
def detect_car():
    """API endpoint for car detection matching mobile app expectations"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'status': 'error'
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp',
                'status': 'error'
            }), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        print(f"Processing uploaded image: {file.filename}")
        
        # Run car detection
        results = detector.detect_car_and_plate(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Format response for mobile app (simplified structure)
        if results and len(results) > 0:
            detection = results[0]  # Get the main detection
            
            # Ensure all values are basic types (string, int, float)
            response = {
                'status': 'success',
                'detections': int(len(results)),
                'vehicle_detected': True,
                'vehicle_type': str(detection['vehicle']['type']),
                'vehicle_confidence': float(detection['vehicle']['score']),
                'make': str(detection['vehicle']['props']['make_model'][0]['make']),
                'model': str(detection['vehicle']['props']['make_model'][0]['model']),
                'brand_confidence': float(detection['vehicle']['props']['make_model'][0]['score']),
                'color': str(detection['vehicle']['props']['color'][0]['value']),
                'color_confidence': float(detection['vehicle']['props']['color'][0]['score']),
                'x_min': int(detection['vehicle']['box']['xmin']),
                'y_min': int(detection['vehicle']['box']['ymin']),
                'x_max': int(detection['vehicle']['box']['xmax']),
                'y_max': int(detection['vehicle']['box']['ymax']),
                'plate_detected': False,
                'plate_text': '',
                'plate_confidence': 0.0,
                'plate_x_min': 0,
                'plate_y_min': 0,
                'plate_x_max': 0,
                'plate_y_max': 0
            }
            
            # Add license plate info if detected
            if detection['plate']:
                response.update({
                    'plate_detected': True,
                    'plate_text': str(detection['plate']['props']['plate'][0]['value']).upper(),
                    'plate_confidence': float(detection['plate']['props']['plate'][0]['score']),
                    'plate_x_min': int(detection['plate']['box']['xmin']),
                    'plate_y_min': int(detection['plate']['box']['ymin']),
                    'plate_x_max': int(detection['plate']['box']['xmax']),
                    'plate_y_max': int(detection['plate']['box']['ymax'])
                })
            
            return jsonify(response), 200
        
        else:
            return jsonify({
                'status': 'success',
                'detections': 0,
                'vehicle_detected': False,
                'message': 'No vehicles detected in image',
                'vehicle_type': '',
                'vehicle_confidence': 0.0,
                'make': '',
                'model': '',
                'brand_confidence': 0.0,
                'color': '',
                'color_confidence': 0.0,
                'x_min': 0,
                'y_min': 0,
                'x_max': 0,
                'y_max': 0,
                'plate_detected': False,
                'plate_text': '',
                'plate_confidence': 0.0,
                'plate_x_min': 0,
                'plate_y_min': 0,
                'plate_x_max': 0,
                'plate_y_max': 0
            }), 200
            
    except Exception as e:
        print(f"API Error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Internal server error during car detection'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Car Detection API',
        'version': '1.0'
    }), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Car Detection API',
        'version': '1.0',
        'endpoints': {
            'POST /detect_car': 'Main car detection endpoint',
            'GET /health': 'Health check',
            'GET /': 'This information'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/detect_car', 
            'content_type': 'multipart/form-data',
            'field': 'image',
            'supported_formats': ['png', 'jpg', 'jpeg', 'gif', 'bmp']
        }
    }), 200

if __name__ == '__main__':
    print("Starting Car Detection API Server...")
    print("Available endpoints:")
    print("  POST /detect_car - Main detection endpoint")
    print("  GET /health - Health check")
    print("  GET / - API information")
    
    # Run server on all interfaces, port 8000
    app.run(host='0.0.0.0', port=8000, debug=True)