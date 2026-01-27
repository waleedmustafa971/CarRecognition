import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
import yaml
import cv2

def standardize_brand_name(brand):
    """Standardize brand names across datasets"""
    brand_mapping = {
        'TOYOTA': 'Toyota',
        'toyota': 'Toyota',
        'HONDA': 'Honda',
        'honda': 'Honda',
        'BMW': 'BMW',
        'bmw': 'BMW',
        'NISSAN': 'Nissan',
        'nissan': 'Nissan',
        'Mercedes': 'Mercedes-Benz',
        'Mercedes-Benz': 'Mercedes-Benz',
        'Mercedes Benz': 'Mercedes-Benz',
        'Volkswagen': 'Volkswagen',
        'VW': 'Volkswagen',
        'Rolls-Royce': 'Rolls-Royce',
        'RollsRoyce': 'Rolls-Royce',
        'Rolls Royce': 'Rolls-Royce',
        'McLaren': 'McLaren',
        'Mclaren': 'McLaren',
        'mclaren': 'McLaren',
        'Bentley': 'Bentley',
        'bentley': 'Bentley',
        'Maserati': 'Maserati',
        'maserati': 'Maserati',
        'Ferrari': 'Ferrari',
        'ferrari': 'Ferrari',
        'Chevrolet': 'Chevrolet',
        'Chevy': 'Chevrolet',
    }
    return brand_mapping.get(brand, brand)

def crop_and_save_detection(image_path, label_path, class_names, output_folder, dataset_name, stats):
    """Crop detected objects from image and save to class folders"""
    saved_count = 0
    
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return 0
        
        img_height, img_width = image.shape[:2]
        
        # Parse labels
        with open(label_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    # Crop detection
                    if x2 > x1 and y2 > y1:
                        cropped = image[y1:y2, x1:x2]
                        
                        # Ensure minimum size
                        if cropped.shape[0] > 10 and cropped.shape[1] > 10:
                            # Get brand name
                            if class_id < len(class_names):
                                brand = class_names[class_id]
                                brand = standardize_brand_name(brand)
                                
                                # Create brand folder
                                brand_folder = output_folder / brand
                                brand_folder.mkdir(exist_ok=True)
                                
                                # Save cropped image
                                img_name = image_path.stem
                                output_file = brand_folder / f"{dataset_name}_{img_name}_{line_idx}.jpg"
                                cv2.imwrite(str(output_file), cropped)
                                saved_count += 1
                                
                                # Update stats
                                stats[brand] += 1
    except Exception as e:
        pass
    
    return saved_count

def process_standard_yolo_dataset(dataset_name, dataset_path, output_train, output_val, brand_to_id):
    """Process car_logo, car_make, stanford_cars format datasets"""
    
    data_yaml_path = dataset_path / 'data.yaml'
    if not data_yaml_path.exists():
        print(f"   ⚠️  data.yaml not found")
        return {}
    
    # Load class names
    with open(data_yaml_path, 'r') as f:
        data_info = yaml.safe_load(f)
    
    class_names = data_info.get('names', {})
    print(f"   Found {len(class_names)} classes in data.yaml")
    
    stats = {'train': defaultdict(int), 'val': defaultdict(int)}
    
    # Process train and val
    for split in ['train', 'val']:
        images_folder = dataset_path / 'images' / split
        labels_folder = dataset_path / 'labels' / split
        
        if not images_folder.exists():
            print(f"   ⚠️  {split} folder not found")
            continue
        
        # Get all images
        image_files = list(images_folder.glob("*.jpg")) + \
                     list(images_folder.glob("*.jpeg")) + \
                     list(images_folder.glob("*.png"))
        
        print(f"   Processing {len(image_files)} images in {split}...")
        
        output_folder = output_train if split == 'train' else output_val
        processed = 0
        
        for img_file in image_files:
            label_file = labels_folder / f"{img_file.stem}.txt"
            
            if label_file.exists():
                count = crop_and_save_detection(
                    img_file, 
                    label_file, 
                    class_names, 
                    output_folder, 
                    dataset_name,
                    stats[split]
                )
                if count > 0:
                    processed += 1
            
            if (processed + 1) % 500 == 0:
                print(f"      {processed + 1} images processed...")
        
        print(f"   ✅ {split}: {processed} images with detections")
    
    return stats

def process_compcars_dataset(dataset_path, output_train, output_val, brand_to_id):
    """Process CompCars numbered folder format"""
    
    print(f"   Processing CompCars (numbered folders)...")
    
    # CompCars has numbered folders (1-163)
    image_folder = dataset_path / 'image'
    
    if not image_folder.exists():
        print(f"   ⚠️  image folder not found")
        return {}
    
    # Get all numbered folders
    numbered_folders = [f for f in image_folder.iterdir() if f.is_dir() and f.name.isdigit()]
    print(f"   Found {len(numbered_folders)} brand folders")
    
    stats = {'train': defaultdict(int), 'val': defaultdict(int)}
    total_processed = 0
    
    for folder in sorted(numbered_folders, key=lambda x: int(x.name)):
        folder_num = folder.name
        
        # Get all images in this folder
        image_files = list(folder.glob("*.jpg")) + \
                     list(folder.glob("*.jpeg")) + \
                     list(folder.glob("*.png"))
        
        if len(image_files) == 0:
            continue
        
        # Use folder number as brand name for now (we'll map it later if needed)
        brand = f"CompCars_Class_{folder_num}"
        
        # Split into train/val (80/20)
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Process train images
        for img_file in train_files:
            output_folder = output_train / brand
            output_folder.mkdir(exist_ok=True)
            
            # Copy image directly (no cropping for CompCars full car images)
            output_file = output_folder / f"compcars_{img_file.name}"
            try:
                shutil.copy2(img_file, output_file)
                stats['train'][brand] += 1
                total_processed += 1
            except:
                pass
        
        # Process val images
        for img_file in val_files:
            output_folder = output_val / brand
            output_folder.mkdir(exist_ok=True)
            
            output_file = output_folder / f"compcars_{img_file.name}"
            try:
                shutil.copy2(img_file, output_file)
                stats['val'][brand] += 1
                total_processed += 1
            except:
                pass
        
        if (int(folder_num)) % 20 == 0:
            print(f"      Processed {folder_num}/163 folders ({total_processed} images)...")
    
    print(f"   ✅ CompCars: {total_processed} images processed")
    return stats

def merge_all_datasets(output_path):
    """Main merge function"""
    
    print("="*60)
    print("UNIFIED CAR BRAND DATASET CREATION")
    print("="*60)
    
    # Create output directories
    output_train = Path(output_path) / "train"
    output_val = Path(output_path) / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    brand_to_id = {}
    all_stats = {}
    
    # Process car_logo
    print(f"\n📂 Processing car_logo dataset...")
    dataset_path = Path("datasets/car_logo")
    if dataset_path.exists():
        stats = process_standard_yolo_dataset(
            'logo', dataset_path, output_train, output_val, brand_to_id
        )
        all_stats['logo'] = stats
    
    # Process car_make
    print(f"\n📂 Processing car_make dataset...")
    dataset_path = Path("datasets/car_make")
    if dataset_path.exists():
        stats = process_standard_yolo_dataset(
            'make', dataset_path, output_train, output_val, brand_to_id
        )
        all_stats['make'] = stats
    
    # Process stanford_cars
    print(f"\n📂 Processing stanford_cars dataset...")
    dataset_path = Path("datasets/stanford_cars")
    if dataset_path.exists():
        stats = process_standard_yolo_dataset(
            'stanford', dataset_path, output_train, output_val, brand_to_id
        )
        all_stats['stanford'] = stats
    
    # Process CompCars (different format)
    print(f"\n📂 Processing compcars dataset...")
    dataset_path = Path("datasets/compcars")
    if dataset_path.exists():
        stats = process_compcars_dataset(
            dataset_path, output_train, output_val, brand_to_id
        )
        all_stats['compcars'] = stats
    
    # Collect all brand names
    print(f"\n📊 Collecting brand statistics...")
    all_brands = set()
    for folder in output_train.iterdir():
        if folder.is_dir():
            all_brands.add(folder.name)
    for folder in output_val.iterdir():
        if folder.is_dir():
            all_brands.add(folder.name)
    
    # Assign IDs
    for brand in sorted(all_brands):
        brand_to_id[brand] = len(brand_to_id)
    
    # Create data.yaml
    print(f"\n📝 Creating data.yaml...")
    data_yaml = {
        'path': str(Path(output_path).absolute()),
        'train': 'train',
        'val': 'val',
        'nc': len(brand_to_id),
        'names': brand_to_id
    }
    
    yaml_path = Path(output_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    # Calculate final statistics
    print("\n" + "="*60)
    print("MERGE COMPLETE - Final Statistics")
    print("="*60)
    
    brand_stats = {}
    for brand in all_brands:
        train_count = len(list((output_train / brand).glob("*"))) if (output_train / brand).exists() else 0
        val_count = len(list((output_val / brand).glob("*"))) if (output_val / brand).exists() else 0
        brand_stats[brand] = {'train': train_count, 'val': val_count, 'total': train_count + val_count}
    
    total_train = sum(s['train'] for s in brand_stats.values())
    total_val = sum(s['val'] for s in brand_stats.values())
    
    print(f"\nTotal brands: {len(brand_stats)}")
    print(f"Total train images: {total_train:,}")
    print(f"Total val images: {total_val:,}")
    print(f"Total images: {total_train + total_val:,}")
    
    # Top 20 brands
    print("\n📊 Top 20 Brands:")
    print("-" * 60)
    sorted_brands = sorted(brand_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for i, (brand, stats) in enumerate(sorted_brands[:20], 1):
        print(f"{i:2}. {brand:30} → {stats['total']:5} (train: {stats['train']:4}, val: {stats['val']:4})")
    
    # Save statistics
    stats_file = Path(output_path) / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(brand_stats, f, indent=2)
    
    print(f"\n✅ Statistics saved: {stats_file}")
    print(f"✅ Config saved: {yaml_path}")
    print("\n" + "="*60)
    print("🎉 DATASET READY FOR TRAINING!")
    print("="*60)
    print(f"\nNext step: python train_unified.py")
    
    return yaml_path

if __name__ == "__main__":
    OUTPUT_PATH = "models/unified_car_brand"
    
    print("\n🚀 Starting Dataset Merge...")
    print(f"Output: {OUTPUT_PATH}\n")
    
    input("Press Enter to start or Ctrl+C to cancel...")
    
    merge_all_datasets(OUTPUT_PATH)