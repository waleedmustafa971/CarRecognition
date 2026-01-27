#!/usr/bin/env python3
"""
Training script for Car Color Detection Model
Usage: python training_scripts/train_car_color.py
"""

from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def train_car_color_model():
    """Train the car color detection model"""
    
    # Get the project root directory (parent of training_scripts)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / 'car-color.v1i.yolov8' / 'data.yaml'
    
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
        name='car_color_v1',                    # experiment name
        project='models/car_color',             # save results to this directory
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
    print(f"Best weights saved to: models/car_color/car_color_v1/weights/best.pt")
    print(f"Last weights saved to: models/car_color/car_color_v1/weights/last.pt")
    print("="*60)

    # Copy best weights to the expected location
    os.makedirs('models/car_color/weights', exist_ok=True)
    shutil.copy(
        'models/car_color/car_color_v1/weights/best.pt',
        'models/car_color/weights/best.pt'
    )
    print("Best weights copied to: models/car_color/weights/best.pt")

    return results

if __name__ == "__main__":
    print("Starting Car Color Model Training...")
    print("Dataset location: car-color.v1i.yolov8/")
    
    # Count images if possible
    try:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        train_dir = project_root / 'car-color.v1i.yolov8' / 'train' / 'images'
        valid_dir = project_root / 'car-color.v1i.yolov8' / 'valid' / 'images'
        
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

    train_car_color_model()