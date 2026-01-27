from ultralytics import YOLO
import torch

def train_compcars():
    """Train universal car model on CompCars"""
    
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load pretrained model
    model = YOLO('yolov8m.pt')
    
    # Train
    results = model.train(
        data='datasets/compcars_yolo/data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,  # Adjust based on GPU memory
        device=0,  # GPU
        
        # Optimization
        patience=20,
        save_period=10,
        
        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        
        # Performance
        workers=8,
        amp=True,
        
        # Output
        project='runs/compcars',
        name='universal_v1',
        exist_ok=True
    )
    
    print("\n✅ Training complete!")
    print(f"Best model: runs/compcars/universal_v1/weights/best.pt")

if __name__ == "__main__":
    train_compcars()