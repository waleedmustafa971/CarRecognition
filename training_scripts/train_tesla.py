#!/usr/bin/env python3
"""
Training script for Tesla Car Detection Model
Usage: python training_scripts/train_tesla.py
"""

from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def train_tesla_model():
    """Train the Tesla car detection model"""
    
    # Get the project root directory (parent of training_scripts)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / 'Tesla.v2i.yolov8' / 'data.yaml'
    
    # Verify dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    print(f"Using dataset: {dataset_path}")

    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano model (fastest)
    # You can also use: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)

    # Train the model
    results = model.train(
        data=str(dataset_path),                # path to dataset config
        epochs=100,                             # number of training epochs
        imgsz=640,                              # image size
        batch=16,                               # batch size (adjust based on GPU memory)
        name='tesla_v1',                        # experiment name
        project='models/tesla',                 # save results to this directory
        patience=20,                            # early stopping patience
        save=True,                              # save checkpoints
        device=0,                               # GPU device (0 for first GPU, 'cpu' for CPU)
        workers=4,                              # number of data loader workers
        pretrained=True,                        # use pretrained weights
        optimizer='Adam',                       # optimizer
        verbose=True,                           # verbose output
        seed=42,                                # random seed for reproducibility
        deterministic=True,                     # deterministic mode
        val=True,                               # validate during training
        plots=True,                             # save plots
    )

    # Evaluate the model
    metrics = model.val()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best weights saved to: models/tesla/tesla_v1/weights/best.pt")
    print(f"Last weights saved to: models/tesla/tesla_v1/weights/last.pt")
    print("="*60)

    # Copy best weights to the expected location
    os.makedirs('models/tesla/weights', exist_ok=True)
    shutil.copy(
        'models/tesla/tesla_v1/weights/best.pt',
        'models/tesla/weights/best.pt'
    )
    print("Best weights copied to: models/tesla/weights/best.pt")

    return results

if __name__ == "__main__":
    print("Starting Tesla Car Model Training...")
    print("Dataset location: Tesla.v2i.yolov8/")
    
    # Count images if possible
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        train_dir = project_root / 'Tesla.v2i.yolov8' / 'train' / 'images'
        valid_dir = project_root / 'Tesla.v2i.yolov8' / 'valid' / 'images'
        
        if train_dir.exists():
            train_count = len(list(train_dir.glob('*')))
            print(f"Training images: {train_count}")
        
        if valid_dir.exists():
            valid_count = len(list(valid_dir.glob('*')))
            print(f"Validation images: {valid_count}")
    except:
        print("Training images: (counting...)")
        print("Validation images: (counting...)")
    
    print("="*60)

    train_tesla_model()