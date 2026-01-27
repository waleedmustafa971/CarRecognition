from ultralytics import YOLO
import os

print("="*60)
print("DEBUGGING COMPCARS MODEL")
print("="*60)

# Check if model file exists
model_paths = [
    'models/compcars/best.pt',
    'runs/compcars/universal_v1/weights/best.pt'
]

print("\n1. Checking model files:")
for path in model_paths:
    exists = os.path.exists(path)
    if exists:
        size = os.path.getsize(path) / (1024*1024)  # MB
        print(f"   ✓ {path} - {size:.1f} MB")
    else:
        print(f"   ✗ {path} - NOT FOUND")

# Try loading the model
print("\n2. Loading model:")
try:
    model = YOLO('runs/compcars/universal_v1/weights/best.pt')
    print(f"   ✓ Model loaded successfully")
    print(f"   Total classes: {len(model.names)}")
    print(f"   Sample brands: {list(model.names.values())[:10]}")
    
    # Test on a simple image
    print("\n3. Testing detection:")
    test_img = 'data/test_car.jpg'  # Change to your test image
    
    if os.path.exists(test_img):
        results = model(test_img, verbose=True)
        
        print(f"\n4. Detection results:")
        for result in results:
            if result.boxes is not None:
                print(f"   Found {len(result.boxes)} detections")
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = model.names[cls]
                    print(f"   - {name}: {conf:.3f}")
            else:
                print("   No detections found")
    else:
        print(f"   ⚠️  Test image not found: {test_img}")
        
except Exception as e:
    print(f"   ✗ Error loading model: {e}")

print("\n" + "="*60)