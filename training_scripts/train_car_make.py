#!/usr/bin/env python3
"""
Training script for Car Make Detection Model
Usage: python training_scripts/train_car_make.py
"""

from ultralytics import YOLO
import os

def train_car_make_model():
    """Train the car make detection model"""

    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano model (fastest)
    # You can also use: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)

    # Train the model
    results = model.train(
        data='datasets/car_make/data.yaml',  # path to dataset config
        epochs=100,                           # number of training epochs
        imgsz=640,                            # image size
        batch=16,                             # batch size (adjust based on GPU memory)
        name='car_make_v1',                   # experiment name
        project='models/car_make',            # save results to this directory
        patience=20,                          # early stopping patience
        save=True,                            # save checkpoints
        device=0,                             # GPU device (0 for first GPU, 'cpu' for CPU)
        workers=4,                            # number of data loader workers
        pretrained=True,                      # use pretrained weights
        optimizer='Adam',                     # optimizer
        verbose=True,                         # verbose output
        seed=42,                              # random seed for reproducibility
        deterministic=True,                   # deterministic mode
        val=True,                             # validate during training
        plots=True,                           # save plots
    )

    # Evaluate the model
    metrics = model.val()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best weights saved to: models/car_make/car_make_v1/weights/best.pt")
    print(f"Last weights saved to: models/car_make/car_make_v1/weights/last.pt")
    print("="*60)

    # Copy best weights to the expected location
    import shutil
    os.makedirs('models/car_make/weights', exist_ok=True)
    shutil.copy(
        'models/car_make/car_make_v1/weights/best.pt',
        'models/car_make/weights/best.pt'
    )
    print("Best weights copied to: models/car_make/weights/best.pt")

    return results

if __name__ == "__main__":
    print("Starting Car Make Model Training...")
    print("Make sure you have placed your images in: datasets/car_make/images/train/")
    print("Make sure you have placed your labels in: datasets/car_make/labels/train/")
    print("="*60)

    train_car_make_model()
