import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
import yaml
import scipy.io

def load_compcars_metadata(dataset_path):
    """Load CompCars brand/model names from metadata"""
    misc_path = dataset_path / 'misc'
    
    # Try to find metadata file
    make_model_file = misc_path / 'make_model_name.mat'
    
    if make_model_file.exists():
        try:
            mat_data = scipy.io.loadmat(str(make_model_file))
            # Extract brand names
            # This will depend on the structure - we'll need to inspect it
            print("✓ Found make_model_name.mat")
            return mat_data
        except:
            pass
    
    return None

def process_compcars_raw(dataset_path, output_path, train_split=0.8):
    """Process raw CompCars dataset with 4-level structure"""
    
    print("="*60)
    print("COMPCARS DATASET PROCESSING")
    print("="*60)
    
    # Create output directories
    output_train = Path(output_path) / "train"
    output_val = Path(output_path) / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    image_folder = dataset_path / 'image'
    
    if not image_folder.exists():
        print("❌ Image folder not found!")
        return
    
    print(f"\n📂 Scanning CompCars structure...")
    
    # Get all brand folders (1-163)
    brand_folders = sorted([f for f in image_folder.iterdir() if f.is_dir()], 
                          key=lambda x: int(x.name))
    
    print(f"   Found {len(brand_folders)} brands")
    
    stats = defaultdict(lambda: {'train': 0, 'val': 0})
    brand_names = {}
    total_images = 0
    
    # Process each brand
    for brand_folder in brand_folders:
        brand_id = brand_folder.name
        brand_name = f"Brand_{brand_id}"  # We'll use ID for now
        brand_names[int(brand_id)] = brand_name
        
        print(f"\n📁 Processing Brand {brand_id}...")
        
        # Get all images recursively (across all models and years)
        all_images = list(brand_folder.glob("**/*.jpg")) + \
                    list(brand_folder.glob("**/*.jpeg")) + \
                    list(brand_folder.glob("**/*.png"))
        
        if len(all_images) == 0:
            print(f"   ⚠️  No images found")
            continue
        
        print(f"   Found {len(all_images)} images")
        
        # Split into train/val
        split_idx = int(len(all_images) * train_split)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        # Create brand folders
        train_brand_folder = output_train / brand_name
        val_brand_folder = output_val / brand_name
        train_brand_folder.mkdir(exist_ok=True)
        val_brand_folder.mkdir(exist_ok=True)
        
        # Copy train images
        for idx, img_file in enumerate(train_images):
            output_file = train_brand_folder / f"img_{idx:05d}.jpg"
            try:
                shutil.copy2(img_file, output_file)
                stats[brand_name]['train'] += 1
                total_images += 1
            except Exception as e:
                pass
        
        # Copy val images
        for idx, img_file in enumerate(val_images):
            output_file = val_brand_folder / f"img_{idx:05d}.jpg"
            try:
                shutil.copy2(img_file, output_file)
                stats[brand_name]['val'] += 1
                total_images += 1
            except Exception as e:
                pass
        
        print(f"   ✅ Copied {stats[brand_name]['train']} train, {stats[brand_name]['val']} val")
    
    # Create data.yaml
    print(f"\n📝 Creating data.yaml...")
    
    sorted_brands = sorted(brand_names.items())
    names_dict = {idx-1: name for idx, name in sorted_brands}  # 0-indexed
    
    data_yaml = {
        'path': str(Path(output_path).absolute()),
        'train': 'train',
        'val': 'val',
        'nc': len(brand_names),
        'names': names_dict
    }
    
    yaml_path = Path(output_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    # Statistics
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    
    print(f"\nTotal brands: {len(stats)}")
    print(f"Total train images: {total_train:,}")
    print(f"Total val images: {total_val:,}")
    print(f"Total images: {total_train + total_val:,}")
    
    # Top 20 brands
    print("\n📊 Top 20 Brands by Image Count:")
    print("-" * 60)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['train'] + x[1]['val'], reverse=True)
    for i, (brand, data) in enumerate(sorted_stats[:20], 1):
        total = data['train'] + data['val']
        print(f"{i:2}. {brand:20} → {total:5} (train: {data['train']:4}, val: {data['val']:4})")
    
    # Save statistics
    stats_file = Path(output_path) / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(dict(stats), f, indent=2)
    
    print(f"\n✅ Config: {yaml_path}")
    print(f"✅ Stats: {stats_file}")
    print("\n" + "="*60)
    print("🎉 READY TO TRAIN!")
    print("="*60)
    print(f"\nNext: python train_unified.py")

if __name__ == "__main__":
    DATASET_PATH = Path("datasets/compcars")
    OUTPUT_PATH = Path("models/unified_car_brand")
    
    print(f"\n🚀 Processing CompCars Dataset")
    print(f"   Input: {DATASET_PATH}")
    print(f"   Output: {OUTPUT_PATH}")
    print(f"\nThis will process ~136k images...")
    print(f"Estimated time: 10-20 minutes\n")
    
    input("Press Enter to start...")
    
    process_compcars_raw(DATASET_PATH, OUTPUT_PATH)