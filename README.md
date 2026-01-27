# Car Recognition System v6.0

**AI-Powered Vehicle Detection, Brand Classification, Color Detection & License Plate Recognition**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

**Author:** Waleed Mustafa
**Organization:** Focal Soft
**Manager:** Sulaiman Abideen
**Last Updated:** January 2026
**Version:** 6.0 (Improved Ensemble)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Model Pipeline](#model-pipeline)
8. [How It Works](#how-it-works)
9. [Performance Metrics](#performance-metrics)
10. [Troubleshooting](#troubleshooting)
11. [Scaling Recommendations](#scaling-recommendations)
12. [Future Improvements](#future-improvements)
13. [Project Structure](#project-structure)

---

## Overview

I built this Car Recognition System as an enterprise-grade AI solution for automated vehicle identification in the UAE market. The system combines multiple deep learning models in an intelligent ensemble to accurately detect:

- **Vehicle Brand/Make** - Toyota, Nissan, BMW, Mercedes, KIA, etc.
- **Vehicle Color** - Using HSV color space analysis
- **License Plate Number** - UAE plate format recognition via OCR

I optimized the system for real-world deployment with robust error handling, memory management, and production-ready logging.

---

## Features

### Core Capabilities

| Feature | Description | Technology |
|---------|-------------|------------|
| Vehicle Detection | Locates cars, trucks, buses in images | YOLOv8n |
| Brand Classification | Identifies vehicle manufacturer | Multi-model ensemble |
| Logo Detection | Finds and identifies car logos | YOLOv8 Detection |
| Color Analysis | Determines primary vehicle color | OpenCV HSV |
| Plate Recognition | Extracts license plate text | EasyOCR |

### Technical Features

- **Multi-Model Ensemble** - I combined 5 specialized models for better accuracy
- **Regional Optimization** - Prioritizes UAE-common brands over Chinese brands
- **Memory Management** - Automatic CUDA cleanup prevents memory leaks
- **Production Logging** - Comprehensive request/response logging
- **REST API** - Flask-based API with CORS support
- **Health Monitoring** - Built-in health checks and metrics endpoints

---

## System Architecture

```
+-------------------------------------------------------------------------+
|                         CAR RECOGNITION SYSTEM                          |
+-------------------------------------------------------------------------+
|                                                                         |
|  +-------------+    +---------------------------------------------+    |
|  |   Client    |--->|              Flask API Server               |    |
|  |  (Mobile/   |    |         api_server_with_logging.py          |    |
|  |   Postman)  |<---|                                             |    |
|  +-------------+    +---------------------------------------------+    |
|                                      |                                  |
|                                      v                                  |
|  +------------------------------------------------------------------+  |
|  |                    DETECTION PIPELINE (main.py)                   |  |
|  +------------------------------------------------------------------+  |
|  |                                                                   |  |
|  |  +--------------+   +--------------+   +--------------+          |  |
|  |  |   STAGE 1    |   |   STAGE 2    |   |   STAGE 3    |          |  |
|  |  |   Vehicle    |-->|    Brand     |-->|  Color/Plate |          |  |
|  |  |  Detection   |   |  Ensemble    |   |  Detection   |          |  |
|  |  |  (YOLOv8n)   |   |  (5 Models)  |   | (HSV+EasyOCR)|          |  |
|  |  +--------------+   +--------------+   +--------------+          |  |
|  |                                                                   |  |
|  +------------------------------------------------------------------+  |
|                                                                         |
|  +------------------------------------------------------------------+  |
|  |                      MODEL ENSEMBLE                               |  |
|  +--------------+--------------+--------------+--------------+-------+  |
|  |   UNIFIED    |    LOGO      |     KIA      |    TESLA     | ...   |  |
|  |  (171 cls)   |  (45 cls)    | (Specialist) | (Specialist) |       |  |
|  |  epoch10.pt  |  detect.pt   |   best.pt    |   best.pt    |       |  |
|  +--------------+--------------+--------------+--------------+-------+  |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Installation

### Prerequisites

- Windows Server 2019/2022 or Ubuntu 20.04+
- Python 3.11
- NVIDIA GPU with CUDA support (recommended)
- Minimum 8GB RAM (16GB recommended)
- 10GB disk space

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/waleedmustafa971/CarRecognition.git
cd CarRecognition

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux
# or
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Start the server
python api_server_with_logging.py
```

### Dependencies

The project uses the following key dependencies:

- **ultralytics** - YOLOv8 framework for object detection and classification
- **easyocr** - Optical character recognition for license plates
- **opencv-python** - Computer vision operations
- **flask** - Web API framework
- **flask-cors** - Cross-origin resource sharing support
- **waitress** - Production WSGI server
- **psutil** - System monitoring
- **torch** - PyTorch deep learning framework

See [requirements.txt](requirements.txt) for the complete list with versions.

---

## Configuration

### Environment Variables

```bash
# Server Configuration
export HOST=0.0.0.0
export PORT=8080
export THREADS=2

# Memory Thresholds
export MEMORY_WARNING_MB=2000
export MEMORY_CRITICAL_MB=3000
export FORCE_GC_EVERY_N_REQUESTS=5
```

### Model Configuration

Models are automatically loaded from these paths:

```python
MODEL_PATHS = {
    'unified': 'runs/classify/unified_car_brand/weights/epoch10.pt',
    'logo': 'models/car_logo/weights/best.pt',
    'kia': 'models/kia/weights/best.pt',
    'tesla': 'models/tesla/weights/best.pt',
    'mclaren': 'models/mclaren/weights/best.pt',
}
```

### Why epoch10.pt instead of best.pt?

During my training analysis, I found severe overfitting in the model:
- **Epoch 10:** val_loss = 1.65 (good generalization)
- **Epoch 80 (best.pt):** val_loss = 2.6 (overfitted)

I chose to use `epoch10.pt` for better real-world accuracy.

---

## API Reference

### Base URL

```
http://your-server-ip:8080
```

### Endpoints

#### POST /detect_car

Detect vehicle information from an image.

**Request (Form Data):**
```bash
curl -X POST http://localhost:8080/detect_car \
  -F "image=@car_photo.jpg"
```

**Request (Base64):**
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAA..."
}
```

**Response:**
```json
{
  "status": "SUCCESS",
  "vehicleMake": "TOYOTA",
  "vehicleColor": "Silver",
  "plateNo": "25586"
}
```

#### GET /health

Check server health and memory status.

**Response:**
```json
{
  "status": "healthy",
  "memory_mb": 1686.36,
  "requests_processed": 42,
  "uptime_hours": 2.5,
  "cuda_available": true
}
```

#### GET /metrics

Detailed system metrics.

**Response:**
```json
{
  "memory": {
    "rss_mb": 1686.36,
    "percent": 21.5,
    "warning_threshold_mb": 2000,
    "critical_threshold_mb": 3000
  },
  "cpu": {
    "percent": 15.2,
    "num_threads": 8
  },
  "requests_total": 42
}
```

#### POST /force_gc

Manually trigger garbage collection.

**Response:**
```json
{
  "status": "success",
  "before_mb": 1800.5,
  "after_mb": 1650.2,
  "freed_mb": 150.3
}
```

---

## Model Pipeline

### Stage 1: Vehicle Detection

```
Input Image --> YOLOv8n --> Bounding Boxes
                   |
            Classes: [2, 5, 7]
            (car, bus, truck)
                   |
            Confidence: 0.25 --> 0.05
            (Progressive thresholds)
```

### Stage 2: Brand Ensemble

I designed the system to run 5 models and combine their predictions intelligently:

```
Vehicle Crop
     |
     +---> LOGO Model (Detection, 45 classes)
     |         |
     |    Finds actual logo in image
     |    High priority when detected
     |
     +---> UNIFIED Model (Classification, 171 classes)
     |         |
     |    Classifies whole vehicle
     |    Trained on 2013-2019 dataset
     |
     +---> SPECIALIST Models (KIA, Tesla, McLaren)
               |
          Verification for specific brands
```

### Ensemble Scoring Logic

```python
Priority Order:
1. Specialist >70% confidence    --> Use directly (x1.3 boost)
2. Logo detected >30%            --> High priority (x1.4 boost)
3. Logo + Unified agree          --> Boosted (x1.25 bonus)
4. Logo only >60%                --> Use logo (x1.15)
5. Both models have predictions  --> Weighted average
6. Single model only             --> Lower confidence
```

### Regional Bias Correction

I noticed the unified model has bias toward Chinese brands due to training data. I implemented a correction mechanism:

```python
# If Chinese brand beats UAE brand by small margin
if top_brand in CHINESE_BRANDS and second_brand in UAE_COMMON_BRANDS:
    if score_difference < 0.15:
        boost UAE brand by 30%
    elif score_difference < 0.25:
        boost UAE brand by 15%
```

**UAE Common Brands:**
Toyota, Nissan, Honda, Hyundai, KIA, BMW, Mercedes-Benz, Audi, Lexus, Infiniti, Land Rover, Porsche, etc.

**Chinese Brands (deprioritized):**
Jianghuai, Yiqi, Dongfeng, Changan, Geely, etc.

### Stage 3: Color and Plate Detection

**Color Detection (HSV):**
```
Vehicle Crop --> BGR to HSV --> Color Masks --> Percentage Calculation
                                                       |
                                             Top color by coverage
```

**Plate Detection (OCR):**
```
Vehicle Crop --> EasyOCR --> Text Extraction --> Clean (digits only)
                                                       |
                                             5-digit UAE format
```

---

## How It Works

### Complete Request Flow

```
1. Client sends image to /detect_car
                |
2. Image saved to temp file
                |
3. YOLOv8n detects vehicles (up to 3)
                |
4. For each vehicle:
   a. Crop vehicle from image
   b. Run LOGO model (detection)
   c. Run UNIFIED model (classification)
   d. Run SPECIALIST models if candidate matches
   e. Combine predictions with ensemble logic
   f. Detect color using HSV analysis
   g. Run OCR for license plate
                |
5. Merge predictions from multiple detections
                |
6. Apply regional bias correction
                |
7. Return JSON response
                |
8. Cleanup temp files and GPU memory
```

### Memory Management

I implemented aggressive memory management to prevent leaks:

```
Startup:           ~1,124 MB (models loaded)
First request:     ~1,686 MB (CUDA pool allocated)
Subsequent:        ~1,692 MB (stable, no leak)

Cleanup triggers:
- After each request: gc.collect() + torch.cuda.empty_cache()
- Every 5 requests: Aggressive cleanup (3x gc.collect)
- Memory >1,500 MB: Automatic aggressive cleanup
- Memory >3,000 MB: Emergency cleanup + warning
```

---

## Performance Metrics

### Benchmarks (NVIDIA Tesla T4)

| Metric | Value |
|--------|-------|
| First Request | ~6.0 seconds (includes CUDA warmup) |
| Subsequent Requests | ~4.5 seconds |
| Memory (Baseline) | ~1,124 MB |
| Memory (After Warmup) | ~1,686 MB |
| Memory per Request | +5-10 MB (freed after cleanup) |
| Requests per Minute | ~12-15 |

### Accuracy (Estimated)

| Component | Accuracy | Notes |
|-----------|----------|-------|
| Vehicle Detection | 95%+ | YOLOv8 pretrained |
| Brand (Logo Visible) | 85-95% | Logo model excels |
| Brand (Logo Hidden) | 60-75% | Relies on unified model |
| Color Detection | 80-90% | Depends on lighting |
| Plate OCR | 70-85% | Depends on image quality |

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'ultralytics'"

```bash
# Use the correct Python installation
& "C:\Program Files\Python311\python.exe" -m pip install ultralytics
```

#### 2. Memory keeps growing

```bash
# Check for memory leaks
curl http://localhost:8080/metrics

# Force garbage collection
curl -X POST http://localhost:8080/force_gc
```

#### 3. CUDA out of memory

```bash
# Reduce batch size or image resolution
# Restart the server to clear GPU memory
```

#### 4. Logo model not detecting

The logo model requires visible car emblems. If the logo is:
- Covered by dirt/damage
- Not visible in angle
- Too small in image

The system falls back to the unified model.

#### 5. Wrong brand detected (Chinese brand)

This is due to training data bias. The system includes regional correction, but for best results:
- Ensure logo is visible
- Use higher resolution images
- Consider retraining with UAE-specific dataset

---

## Scaling Recommendations

### Short-Term Improvements

#### 1. Add GPU Batching

```python
# Current: Process one image at a time
# Improved: Batch multiple requests
def batch_predict(images, batch_size=4):
    results = model.predict(images, batch=batch_size)
    return results
```

**Expected improvement:** 2-3x throughput

#### 2. Model Quantization

```python
# Convert to FP16 for faster inference
model.export(format='engine', half=True)  # TensorRT
```

**Expected improvement:** 30-50% faster inference

#### 3. Add Redis Caching

```python
# Cache frequent results
import redis
cache = redis.Redis()

def get_cached_prediction(image_hash):
    return cache.get(f"pred:{image_hash}")
```

**Expected improvement:** Instant response for repeated images

### Medium-Term Improvements

#### 4. Horizontal Scaling with Load Balancer

```
                    +-------------+
                    |   NGINX     |
                    |Load Balancer|
                    +------+------+
           +---------------+---------------+
           |               |               |
           v               v               v
    +----------+    +----------+    +----------+
    | Server 1 |    | Server 2 |    | Server 3 |
    | GPU: T4  |    | GPU: T4  |    | GPU: T4  |
    +----------+    +----------+    +----------+
```

#### 5. Implement Model Serving (Triton)

```yaml
# model_repository/car_brand/config.pbtxt
name: "car_brand"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  { name: "images", data_type: TYPE_FP32, dims: [ 3, 640, 640 ] }
]
output [
  { name: "predictions", data_type: TYPE_FP32, dims: [ 171 ] }
]
```

#### 6. Add Monitoring Stack

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
  grafana:
    image: grafana/grafana
  car-recognition:
    build: .
    ports:
      - "8080:8080"
```

### Long-Term Improvements

#### 7. Hierarchical Classification

I recommend implementing a hierarchical approach:

```
Step 1: Region classifier (5 classes)
        +-- Japanese (Toyota, Honda, Nissan, Mazda, etc.)
        +-- German (BMW, Mercedes, Audi, VW, Porsche)
        +-- Korean (Hyundai, KIA, Genesis)
        +-- American (Ford, Chevrolet, Tesla, etc.)
        +-- Other

Step 2: Brand classifier per region (15-30 classes each)

Step 3: Model classifier per brand (optional)
```

**Benefits:**
- Smaller, more accurate models
- Faster inference (fewer classes)
- Easier to update individual regions

#### 8. Edge Deployment

```
+------------------------------------------+
|            CLOUD (Training)              |
|  - Model training                        |
|  - Model versioning                      |
|  - A/B testing                           |
+--------------------+---------------------+
                     | Model sync
                     v
+------------------------------------------+
|         EDGE (Inference)                 |
|  - NVIDIA Jetson / Intel NCS            |
|  - Low latency (<100ms)                  |
|  - Offline capable                       |
+------------------------------------------+
```

#### 9. Continuous Learning Pipeline

```
User Feedback --> Incorrect Predictions --> Review Queue
                         |
              Human Verification
                         |
              Add to Training Set
                         |
              Nightly Retraining
                         |
              A/B Testing New Model
                         |
              Gradual Rollout
```

---

## Project Structure

```
car_recognition/
+-- src/
|   +-- main.py                 # Core detection logic
+-- models/
|   +-- car_logo/
|   |   +-- weights/
|   |       +-- best.pt         # Logo detection model (45 classes)
|   +-- kia/
|   |   +-- weights/
|   |       +-- best.pt         # KIA specialist
|   +-- tesla/
|   |   +-- weights/
|   |       +-- best.pt         # Tesla specialist
|   +-- mclaren/
|       +-- weights/
|           +-- best.pt         # McLaren specialist
+-- runs/
|   +-- classify/
|       +-- unified_car_brand/
|           +-- weights/
|               +-- epoch10.pt  # Unified model (recommended)
|               +-- best.pt     # Overfitted, not used
+-- logs/
|   +-- api_YYYYMMDD.log        # Daily log files
+-- api_server_with_logging.py  # Flask API server
+-- requirements.txt            # Python dependencies
+-- README.md                   # This file
+-- .gitignore
```

---

## Model Details

### Unified Model (epoch10.pt)

| Property | Value |
|----------|-------|
| Architecture | YOLOv8x-cls |
| Parameters | ~57M |
| Classes | 171 |
| Training Data | CompCars + Custom |
| Input Size | 224x224 |
| Task | Classification |

### Logo Model (best.pt)

| Property | Value |
|----------|-------|
| Architecture | YOLOv8m |
| Parameters | ~25M |
| Classes | 45 |
| Training Data | Car logo dataset |
| Input Size | 640x640 |
| Task | Detection |

### Specialist Models

| Model | Classes | Purpose |
|-------|---------|---------|
| KIA | 1 | Verify KIA predictions |
| Tesla | 1 | Verify Tesla predictions |
| McLaren | 1 | Verify McLaren predictions |

---

## Security Considerations

1. **Input Validation** - All uploaded files are validated for type and size
2. **Temp File Cleanup** - Automatic cleanup prevents disk exhaustion
3. **Rate Limiting** - Consider adding rate limiting for production
4. **CORS** - Configured for cross-origin requests (configure for production)
5. **Logging** - No sensitive data logged (only metadata)

---

## License

Proprietary - Focal Soft

---

## Author

**Waleed Mustafa**  
Software Developer  
Focal Soft

---

## Acknowledgments

- Ultralytics - YOLOv8 framework
- EasyOCR - OCR library
- OpenCV - Computer vision library
- Flask - Web framework

---

*Last updated: November 2025*