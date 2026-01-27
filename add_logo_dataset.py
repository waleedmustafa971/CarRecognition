from pathlib import Path
import cv2
import yaml
from collections import defaultdict

def standardize_brand(brand):
    """Standardize brand names"""
    mapping = {
        'Toyota': 'Toyota',
        'Honda': 'Honda',
        'Ford': 'Ford',
        'Chevrolet': 'Chevrolet',
        'BMW': 'BMW',
        'Mercedes': 'Mercedes-Benz',
        'Audi': 'Audi',
        'Nissan': 'Nissan',
        'Hyundai': 'Hyundai',
        'Kia': 'KIA',
    }
    return mapping.get(brand, brand)

def crop_and_save_logo(image_path, label_path, class_names, output_folder):
    """Crop logo from image and save"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return 0
        
        img_h, img_w = image.shape[:2]
        saved = 0
        
        with open(label_path, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to pixels
                    x1 = int((x_center - width/2) * img_w)
                    y1 = int((y_center - height/2) * img_h)
                    x2 = int((x_center + width/2) * img_w)
                    y2 = int((y_center + height/2) * img_h)
                    
                    # Bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped = image[y1:y2, x1:x2]
                        
                        if cropped.shape[0] > 20 and cropped.shape[1] > 20:
                            # Handle both dict and list formats
                            if isinstance(class_names, dict):
                                brand = class_names.get(class_id, f"Class_{class_id}")
                            elif isinstance(class_names, list):
                                brand = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                            else:
                                brand = f"Class_{class_id}"
                            
                            brand = standardize_brand(brand)
                            
                            brand_folder = output_folder / brand
                            brand_folder.mkdir(exist_ok=True)
                            
                            output_file = brand_folder / f"logo_{image_path.stem}_{idx}.jpg"
                            cv2.imwrite(str(output_file), cropped)
                            saved += 1
        return saved
    except Exception as e:
        print(f"      Error: {e}")
        return 0

def add_logo_dataset():
    """Add logo dataset to unified dataset"""
    
    print("="*60)
    print("ADDING LOGO DATASET TO UNIFIED DATASET")
    print("="*60)
    
    logo_path = Path("datasets/car-logo-detection-1")
    output_path = Path("models/unified_car_brand")
    
    # Load logo dataset config
    data_yaml = logo_path / 'data.yaml'
    if not data_yaml.exists():
        print("❌ Logo dataset not found!")
        return
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data.get('names', {})
    
    print(f"\n📋 Logo dataset has {len(class_names)} classes:")
    
    # Handle both dict and list formats
    if isinstance(class_names, dict):
        for id, name in class_names.items():
            print(f"   {id}: {name}")
    elif isinstance(class_names, list):
        for id, name in enumerate(class_names):
            print(f"   {id}: {name}")
    
    stats = defaultdict(int)
    
    # Process train
    print(f"\n📂 Processing train set...")
    train_images = logo_path / 'train' / 'images'
    train_labels = logo_path / 'train' / 'labels'
    output_train = output_path / 'train'
    
    if train_images.exists():
        image_files = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
        print(f"   Found {len(image_files)} images")
        
        processed = 0
        for img_file in image_files:
            label_file = train_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                count = crop_and_save_logo(img_file, label_file, class_names, output_train)
                if count > 0:
                    processed += 1
                    stats['train'] += count
            
            if (processed + 1) % 100 == 0:
                print(f"      {processed + 1} images processed...")
        
        print(f"   ✅ Train: {processed} images processed, {stats['train']} logos extracted")
    
    # Process valid (as val)
    print(f"\n📂 Processing validation set...")
    val_images = logo_path / 'valid' / 'images'
    val_labels = logo_path / 'valid' / 'labels'
    output_val = output_path / 'val'
    
    if val_images.exists():
        image_files = list(val_images.glob("*.jpg")) + list(val_images.glob("*.png"))
        print(f"   Found {len(image_files)} images")
        
        processed = 0
        for img_file in image_files:
            label_file = val_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                count = crop_and_save_logo(img_file, label_file, class_names, output_val)
                if count > 0:
                    processed += 1
                    stats['val'] += count
            
            if (processed + 1) % 100 == 0:
                print(f"      {processed + 1} images processed...")
        
        print(f"   ✅ Val: {processed} images processed, {stats['val']} logos extracted")
    
    # Update statistics
    print("\n📊 Updating final statistics...")
    
    # Count all images per brand
    brand_stats = {}
    for split in ['train', 'val']:
        split_path = output_path / split
        for brand_folder in split_path.iterdir():
            if brand_folder.is_dir():
                brand = brand_folder.name
                if brand not in brand_stats:
                    brand_stats[brand] = {'train': 0, 'val': 0}
                
                count = len(list(brand_folder.glob("*")))
                brand_stats[brand][split] = count
    
    total_train = sum(s['train'] for s in brand_stats.values())
    total_val = sum(s['val'] for s in brand_stats.values())
    
    print(f"\n📈 Final Dataset Statistics:")
    print(f"   Total brands: {len(brand_stats)}")
    print(f"   Train images: {total_train:,}")
    print(f"   Val images: {total_val:,}")
    print(f"   Total: {total_train + total_val:,}")
    
    # Show which brands got logos added
    print(f"\n🎯 Brands with logo detections added:")
    brands_with_logos = []
    for brand in sorted(brand_stats.keys()):
        logo_count = len(list((output_path / 'train' / brand).glob("logo_*")))
        if logo_count > 0:
            brands_with_logos.append((brand, logo_count))
    
    for brand, count in brands_with_logos:
        print(f"   {brand}: +{count} logo crops")
    
    if not brands_with_logos:
        print("   ⚠️ No logo files found - check if extraction worked")
    
    print("\n" + "="*60)
    print("✅ LOGO DATASET ADDED!")
    print("="*60)
    print(f"\n🎯 Ready to train unified model!")

if __name__ == "__main__":
    print("\n📷 Adding logo dataset to unified dataset...")
    print("   This will extract and add logo crops\n")
    
    input("Press Enter to start...")
    
    add_logo_dataset()