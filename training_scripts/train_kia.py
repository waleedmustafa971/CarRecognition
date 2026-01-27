#!/usr/bin/env python3
"""
Training script for Kia Car Detection Model
Usage: python training_scripts/train_kia.py
"""

from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def train_kia_model():
    """Train the Kia car detection model"""
    
    # Get the project root directory (parent of training_scripts)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / 'Kia.v1i.yolov8' / 'data.yaml'
    
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
        name='kia_v1',                          # experiment name
        project='models/kia',                   # save results to this directory
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
    print(f"Best weights saved to: models/kia/kia_v1/weights/best.pt")
    print(f"Last weights saved to: models/kia/kia_v1/weights/last.pt")
    print("="*60)

    # Copy best weights to the expected location
    os.makedirs('models/kia/weights', exist_ok=True)
    shutil.copy(
        'models/kia/kia_v1/weights/best.pt',
        'models/kia/weights/best.pt'
    )
    print("Best weights copied to: models/kia/weights/best.pt")

    return results

if __name__ == "__main__":
    print("Starting Kia Car Model Training...")
    print("Dataset location: Kia.v1i.yolov8/")
    print(f"Training images: 879")
    print(f"Validation images: 146")
    print("="*60)

    train_kia_model()