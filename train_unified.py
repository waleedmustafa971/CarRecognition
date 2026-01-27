from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def train_unified_model():
    print("="*60)
    print("TRAINING UNIFIED CAR BRAND MODEL")
    print("="*60)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  WARNING: No GPU detected! Training will be VERY slow.")
    
    # Dataset path (directory, not yaml!)
    dataset_path = 'models/unified_car_brand'
    
    if not Path(dataset_path).exists():
        print(f"\n❌ ERROR: {dataset_path} not found!")
        return
    
    # Check train/val folders
    train_path = Path(dataset_path) / 'train'
    val_path = Path(dataset_path) / 'val'
    
    if not train_path.exists():
        print(f"\n❌ ERROR: Train folder not found: {train_path}")
        return
    
    if not val_path.exists():
        print(f"\n❌ ERROR: Val folder not found: {val_path}")
        return
    
    # Count classes and images
    train_classes = [d for d in train_path.iterdir() if d.is_dir()]
    val_classes = [d for d in val_path.iterdir() if d.is_dir()]
    
    train_images = len(list(train_path.glob("*/*.jpg")))
    val_images = len(list(val_path.glob("*/*.jpg")))
    
    print(f"\n📊 Dataset Info:")
    print(f"   Train classes: {len(train_classes)}")
    print(f"   Val classes: {len(val_classes)}")
    print(f"   Train images: {train_images:,}")
    print(f"   Val images: {val_images:,}")
    
    # Load base model
    print("\n📦 Loading YOLOv8x-cls model (classification)...")
    model = YOLO('yolov8x-cls.pt')
    
    print("\n🚀 Starting training...")
    print("   This will take 8-12 hours on Tesla T4")
    print("   Model will be saved to: runs/classify/unified_car_brand/")
    print("\n" + "="*60)
    
    # Training configuration
    results = model.train(
        data=dataset_path,       # ✅ Directory path, not yaml!
        epochs=100,
        imgsz=224,
        batch=32,
        device=device,
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mixup=0.2,
        
        # Validation
        val=True,
        patience=20,
        
        # Saving
        save=True,
        save_period=10,
        
        # Logging
        project='runs/classify',
        name='unified_car_brand',
        exist_ok=True,
        verbose=True,
        
        # Hardware
        workers=8,
        
        # Resume
        resume=False,
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    print(f"\n📊 Results:")
    print(f"   Best model: runs/classify/unified_car_brand/weights/best.pt")
    print(f"   Last model: runs/classify/unified_car_brand/weights/last.pt")
    print(f"   Results: runs/classify/unified_car_brand/results.png")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review training curves")
    print(f"   2. Test the model")
    print(f"   3. Deploy to production!")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UNIFIED CAR BRAND MODEL TRAINING")
    print("="*60)
    print("\n⏱️  Estimated time: 8-12 hours (Tesla T4)")
    print("💾  Disk space needed: ~5GB for checkpoints")
    print("📊  Dataset: 156k images, 161 brands")
    print("\n" + "="*60)
    
    input("\nPress Enter to start training or Ctrl+C to cancel...")
    
    train_unified_model()