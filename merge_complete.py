import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
import yaml
import cv2
try:
    import scipy.io
    HAS_SCIPY = True
except:
    HAS_SCIPY = False
    print("⚠️  scipy not installed, using brand IDs instead of names")

def load_compcars_brand_names(dataset_path):
    """Load real brand names from CompCars metadata"""
    if not HAS_SCIPY:
        return {}
    
    misc_path = dataset_path / 'misc'
    make_model_file = misc_path / 'make_model_name.mat'
    
    if not make_model_file.exists():
        return {}
    
    try:
        print("📖 Loading CompCars brand names...")
        mat_data = scipy.io.loadmat(str(make_model_file))
        
        # Extract make names
        make_names = mat_data.get('make_names', [])
        brand_dict = {}
        
        for idx, name_array in enumerate(make_names, 1):
            if len(name_array) > 0 and len(name_array[0]) > 0:
                brand_name = str(name_array[0][0]).strip()
                brand_dict[idx] = brand_name
                
        print(f"✓ Loaded {len(brand_dict)} brand names")
        return brand_dict
    except Exception as e:
        print(f"⚠️  Could not load brand names: {e}")
        return {}

def standardize_brand_name(brand):
    """Standardize brand names"""
    brand_mapping = {
        'TOYOTA': 'Toyota',
        'toyota': 'Toyota',
        'HONDA': 'Honda',
        'honda': 'Honda',
        'BMW': 'BMW',
        'bmw': 'BMW',
        'Mercedes-Benz': 'Mercedes-Benz',
        'Mercedes Benz': 'Mercedes-Benz',
        'Benz': 'Mercedes-Benz',
        'AM General': 'AM General',
        'Acura': 'Acura',
        'Audi': 'Audi',
        'Aston Martin': 'Aston Martin',
        'Bentley': 'Bentley',
        'BMW': 'BMW',
        'Bugatti': 'Bugatti',
        'Buick': 'Buick',
        'Cadillac': 'Cadillac',
        'Chevrolet': 'Chevrolet',
        'Chrysler': 'Chrysler',
        'Dodge': 'Dodge',
        'Ferrari': 'Ferrari',
        'FIAT': 'FIAT',
        'Ford': 'Ford',
        'GMC': 'GMC',
        'Honda': 'Honda',
        'Hyundai': 'Hyundai',
        'Infiniti': 'Infiniti',
        'Jaguar': 'Jaguar',
        'Jeep': 'Jeep',
        'Kia': 'Kia',
        'Lamborghini': 'Lamborghini',
        'Land Rover': 'Land Rover',
        'Lexus': 'Lexus',
        'Lincoln': 'Lincoln',
        'Maserati': 'Maserati',
        'Mazda': 'Mazda',
        'McLaren': 'McLaren',
        'Mitsubishi': 'Mitsubishi',
        'Nissan': 'Nissan',
        'Porsche': 'Porsche',
        'Ram': 'Ram',
        'Rolls-Royce': 'Rolls-Royce',
        'Scion': 'Scion',
        'smart': 'Smart',
        'Subaru': 'Subaru',
        'Suzuki': 'Suzuki',
        'Tesla': 'Tesla',
        'Toyota': 'Toyota',
        'Volkswagen': 'Volkswagen',
        'Volvo': 'Volvo',
    }
    return brand_mapping.get(brand, brand)

def crop_logo_detection(image_path, label_path, class_names, output_folder, dataset_name, stats):
    """Crop logo detections from image"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return 0
        
        img_height, img_width = image.shape[:2]
        saved = 0
        
        with open(label_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to pixel coords
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    # Bounds check
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped = image[y1:y2, x1:x2]
                        
                        if cropped.shape[0] > 20 and cropped.shape[1] > 20:
                            if class_id in class_names:
                                brand = standardize_brand_name(class_names[class_id])
                                
                                brand_folder = output_folder / brand
                                brand_folder.mkdir(exist_ok=True)
                                
                                output_file = brand_folder / f"{dataset_name}_{image_path.stem}_{line_idx}.jpg"
                                cv2.imwrite(str(output_file), cropped)
                                saved += 1
                                stats[brand] += 1
        return saved
    except:
        return 0

def process_logo_dataset(dataset_path, output_train, output_val):
    """Process car-logo-detection-1 dataset"""
    print(f"\n📂 Processing car-logo-detection dataset...")
    
    data_yaml = dataset_path / 'data.yaml'
    if not data_yaml.exists():
        print("   ⚠️  data.yaml not found")
        return {}
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data.get('names', {})
    print(f"   Found {len(class_names)} classes")
    
    stats = {'train': defaultdict(int), 'val': defaultdict(int)}
    
    # Process train
    train_images = dataset_path / 'train' / 'images'
    train_labels = dataset_path / 'train' / 'labels'
    
    if train_images.exists():
        image_files = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
        print(f"   Processing {len(image_files)} train images...")
        
        processed = 0
        for img_file in image_files:
            label_file = train_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                count = crop_logo_detection(img_file, label_file, class_names, 
                                           output_train, 'logo', stats['train'])
                if count > 0:
                    processed += 1
            
            if (processed + 1) % 500 == 0:
                print(f"      {processed + 1} processed...")
        
        print(f"   ✅ Train: {processed} images")
    
    # Process valid (as val)
    val_images = dataset_path / 'valid' / 'images'
    val_labels = dataset_path / 'valid' / 'labels'
    
    if val_images.exists():
        image_files = list(val_images.glob("*.jpg")) + list(val_images.glob("*.png"))
        print(f"   Processing {len(image_files)} val images...")
        
        processed = 0
        for img_file in image_files:
            label_file = val_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                count = crop_logo_detection(img_file, label_file, class_names, 
                                           output_val, 'logo', stats['val'])
                if count > 0:
                    processed += 1
        
        print(f"   ✅ Val: {processed} images")
    
    return stats

def process_compcars(dataset_path, brand_names, output_train, output_val, train_split=0.8):
    """Process CompCars with real brand names"""
    print(f"\n📂 Processing CompCars dataset...")
    
    image_folder = dataset_path / 'image'
    brand_folders = sorted([f for f in image_folder.iterdir() if f.is_dir()], 
                          key=lambda x: int(x.name))
    
    print(f"   Found {len(brand_folders)} brands")
    
    stats = {'train': defaultdict(int), 'val': defaultdict(int)}
    
    for brand_folder in brand_folders:
        brand_id = int(brand_folder.name)
        brand_name = brand_names.get(brand_id, f"Brand_{brand_id}")
        brand_name = standardize_brand_name(brand_name)
        
        # Get all images
        all_images = list(brand_folder.glob("**/*.jpg"))
        
        if len(all_images) == 0:
            continue
        
        # Split
        split_idx = int(len(all_images) * train_split)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        # Copy train
        train_folder = output_train / brand_name
        train_folder.mkdir(exist_ok=True)
        for idx, img in enumerate(train_images):
            out = train_folder / f"compcars_{brand_id}_{idx:05d}.jpg"
            try:
                shutil.copy2(img, out)
                stats['train'][brand_name] += 1
            except:
                pass
        
        # Copy val
        val_folder = output_val / brand_name
        val_folder.mkdir(exist_ok=True)
        for idx, img in enumerate(val_images):
            out = val_folder / f"compcars_{brand_id}_{idx:05d}.jpg"
            try:
                shutil.copy2(img, out)
                stats['val'][brand_name] += 1
            except:
                pass
        
        if brand_id % 20 == 0:
            print(f"      Processed {brand_id}/163 brands...")
    
    print(f"   ✅ CompCars complete")
    return stats

def merge_all():
    """Main merge function"""
    print("="*60)
    print("UNIFIED CAR BRAND DATASET - COMPLETE MERGE")
    print("="*60)
    
    OUTPUT = Path("models/unified_car_brand")
    output_train = OUTPUT / "train"
    output_val = OUTPUT / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    # Load CompCars brand names
    compcars_brands = load_compcars_brand_names(Path("datasets/compcars"))
    
    # Process logo dataset
    logo_stats = process_logo_dataset(
        Path("datasets/car-logo-detection-1"),
        output_train,
        output_val
    )
    
    # Process CompCars
    compcars_stats = process_compcars(
        Path("datasets/compcars"),
        compcars_brands,
        output_train,
        output_val
    )
    
    # Collect all brands
    all_brands = set()
    for folder in output_train.iterdir():
        if folder.is_dir():
            all_brands.add(folder.name)
    
    brand_to_id = {brand: idx for idx, brand in enumerate(sorted(all_brands))}
    
    # Create data.yaml
    data_yaml = {
        'path': str(OUTPUT.absolute()),
        'train': 'train',
        'val': 'val',
        'nc': len(brand_to_id),
        'names': brand_to_id
    }
    
    yaml_path = OUTPUT / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    # Final stats
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    
    brand_stats = {}
    for brand in all_brands:
        train_count = len(list((output_train / brand).glob("*")))
        val_count = len(list((output_val / brand).glob("*")))
        brand_stats[brand] = {'train': train_count, 'val': val_count}
    
    total_train = sum(s['train'] for s in brand_stats.values())
    total_val = sum(s['val'] for s in brand_stats.values())
    
    print(f"\nTotal brands: {len(brand_stats)}")
    print(f"Train images: {total_train:,}")
    print(f"Val images: {total_val:,}")
    print(f"Total: {total_train + total_val:,}")
    
    # Top 20
    print("\n📊 Top 20 Brands:")
    print("-" * 60)
    sorted_brands = sorted(brand_stats.items(), 
                          key=lambda x: x[1]['train'] + x[1]['val'], 
                          reverse=True)
    for i, (brand, stats) in enumerate(sorted_brands[:20], 1):
        total = stats['train'] + stats['val']
        print(f"{i:2}. {brand:25} → {total:6} (train: {stats['train']:5}, val: {stats['val']:5})")
    
    # Save
    with open(OUTPUT / 'statistics.json', 'w') as f:
        json.dump(brand_stats, f, indent=2)
    
    print(f"\n✅ Config: {yaml_path}")
    print("="*60)
    print("🎉 READY TO TRAIN!")
    print("="*60)

if __name__ == "__main__":
    print("\n🚀 Merging car-logo-detection + CompCars")
    print("   This will take 15-30 minutes...\n")
    input("Press Enter to start...")
    merge_all()