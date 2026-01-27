import os
import shutil
from pathlib import Path
from collections import defaultdict
import json
import yaml

def standardize_brand_name(brand):
    """Standardize brand names across datasets"""
    brand_mapping = {
        # Standardize case
        'TOYOTA': 'Toyota',
        'toyota': 'Toyota',
        'HONDA': 'Honda',
        'honda': 'Honda',
        'BMW': 'BMW',
        'bmw': 'BMW',
        'NISSAN': 'Nissan',
        'nissan': 'Nissan',
        'AUDI': 'Audi',
        'audi': 'Audi',
        
        # Fix typos/variations
        'Mercedes-Benz': 'Mercedes-Benz',
        'Mercedes Benz': 'Mercedes-Benz',
        'mercedes': 'Mercedes-Benz',
        'Volkswagen': 'Volkswagen',
        'VW': 'Volkswagen',
        'volkswagen': 'Volkswagen',
        'Chevrolet': 'Chevrolet',
        'Chevy': 'Chevrolet',
        'Lamborghini': 'Lamborghini',
        'Lamorghini': 'Lamborghini',  # Common typo
        'Porsche': 'Porsche',
        'porsche': 'Porsche',
        'Land Rover': 'Land Rover',
        'LandRover': 'Land Rover',
        'Range Rover': 'Land Rover',
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
        'Aston Martin': 'Aston Martin',
        'AstonMartin': 'Aston Martin',
    }
    
    return brand_mapping.get(brand, brand)

def merge_datasets(logo_path, compcars_path, make_path, stanford_path, output_path):
    """
    Merge multiple car brand detection datasets into one unified dataset
    """
    
    print("="*60)
    print("DATASET MERGER - Creating Unified Car Brand Model")
    print("="*60)
    
    # Create output directories
    output_train = Path(output_path) / "train"
    output_val = Path(output_path) / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'sources': set()})
    brand_to_id = {}
    current_id = 0
    
    # List of datasets to process
    datasets = [
        {'name': 'logo', 'path': logo_path, 'priority': 1},
        {'name': 'compcars', 'path': compcars_path, 'priority': 2},
        {'name': 'make', 'path': make_path, 'priority': 3},
        {'name': 'stanford', 'path': stanford_path, 'priority': 4},
    ]
    
    # Process each dataset
    for dataset in datasets:
        dataset_name = dataset['name']
        dataset_path = Path(dataset['path'])
        
        if not dataset_path.exists():
            print(f"⚠️  Skipping {dataset_name}: Path not found")
            continue
        
        print(f"\n📂 Processing {dataset_name} dataset...")
        
        for split in ['train', 'val']:
            split_path = dataset_path / split
            
            if not split_path.exists():
                print(f"   ⚠️  {split} folder not found, skipping...")
                continue
            
            # Get all class folders
            class_folders = [f for f in split_path.iterdir() if f.is_dir()]
            
            print(f"   Found {len(class_folders)} classes in {split}")
            
            for class_folder in class_folders:
                original_brand = class_folder.name
                standardized_brand = standardize_brand_name(original_brand)
                
                # Assign class ID
                if standardized_brand not in brand_to_id:
                    brand_to_id[standardized_brand] = current_id
                    current_id += 1
                
                # Create brand folder in output
                if split == 'train':
                    output_brand_folder = output_train / standardized_brand
                else:
                    output_brand_folder = output_val / standardized_brand
                
                output_brand_folder.mkdir(exist_ok=True)
                
                # Copy images
                image_files = list(class_folder.glob("*.jpg")) + \
                             list(class_folder.glob("*.jpeg")) + \
                             list(class_folder.glob("*.png"))
                
                for img_file in image_files:
                    # Create unique filename to avoid collisions
                    new_filename = f"{dataset_name}_{img_file.name}"
                    output_file = output_brand_folder / new_filename
                    
                    try:
                        shutil.copy2(img_file, output_file)
                        stats[standardized_brand][split] += 1
                        stats[standardized_brand]['sources'].add(dataset_name)
                    except Exception as e:
                        print(f"      Error copying {img_file}: {e}")
    
    # Create data.yaml for YOLO
    print("\n📝 Creating data.yaml configuration...")
    
    sorted_brands = sorted(brand_to_id.keys())
    names_dict = {brand_to_id[brand]: brand for brand in sorted_brands}
    
    data_yaml = {
        'path': str(Path(output_path).absolute()),
        'train': 'train',
        'val': 'val',
        'nc': len(brand_to_id),
        'names': names_dict
    }
    
    yaml_path = Path(output_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✅ Created {yaml_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("MERGE COMPLETE - Statistics")
    print("="*60)
    print(f"\nTotal unique brands: {len(stats)}")
    print(f"Total classes: {len(brand_to_id)}")
    
    # Calculate totals
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    
    print(f"\nTotal train images: {total_train:,}")
    print(f"Total val images: {total_val:,}")
    print(f"Total images: {total_train + total_val:,}")
    
    # Show top 20 brands by image count
    print("\n📊 Top 20 Brands by Image Count:")
    print("-" * 60)
    sorted_stats = sorted(stats.items(), 
                         key=lambda x: x[1]['train'] + x[1]['val'], 
                         reverse=True)
    
    for i, (brand, data) in enumerate(sorted_stats[:20], 1):
        total = data['train'] + data['val']
        sources = ', '.join(sorted(data['sources']))
        print(f"{i:2}. {brand:20} → {total:5} images (train: {data['train']:4}, val: {data['val']:4}) [{sources}]")
    
    # Show brands with few images (potential issues)
    print("\n⚠️  Brands with <50 images (may need more data):")
    print("-" * 60)
    low_count_brands = [(b, d) for b, d in stats.items() 
                        if d['train'] + d['val'] < 50]
    
    if low_count_brands:
        for brand, data in sorted(low_count_brands, 
                                 key=lambda x: x[1]['train'] + x[1]['val']):
            total = data['train'] + data['val']
            print(f"   {brand:20} → {total:3} images")
    else:
        print("   None! All brands have sufficient data.")
    
    # Save detailed statistics
    stats_file = Path(output_path) / 'merge_statistics.json'
    stats_dict = {
        brand: {
            'train': data['train'],
            'val': data['val'],
            'total': data['train'] + data['val'],
            'sources': list(data['sources'])
        }
        for brand, data in stats.items()
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"\n✅ Detailed statistics saved to: {stats_file}")
    print("\n" + "="*60)
    print("✅ DATASET MERGE COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review low-count brands and add more images if needed")
    print(f"2. Train unified model: python train_unified.py")
    print(f"3. Test on your images")
    
    return str(yaml_path)

if __name__ == "__main__":
    # Configure your dataset paths
    LOGO_PATH = "models/car_logo/dataset"  # Update this
    COMPCARS_PATH = "models/compcars/dataset"  # Update this
    MAKE_PATH = "models/car_make/dataset"  # Update this (if exists)
    STANFORD_PATH = "models/stanford_cars/dataset"  # Update this (if exists)
    OUTPUT_PATH = "models/unified_car_brand"
    
    print("\n🚀 Starting Dataset Merge Process...")
    print(f"\nSource datasets:")
    print(f"  - Logo: {LOGO_PATH}")
    print(f"  - CompCars: {COMPCARS_PATH}")
    print(f"  - Make: {MAKE_PATH}")
    print(f"  - Stanford: {STANFORD_PATH}")
    print(f"\nOutput: {OUTPUT_PATH}\n")
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    yaml_path = merge_datasets(
        logo_path=LOGO_PATH,
        compcars_path=COMPCARS_PATH,
        make_path=MAKE_PATH,
        stanford_path=STANFORD_PATH,
        output_path=OUTPUT_PATH
    )
    
    print(f"\n✅ Ready to train! Config file: {yaml_path}")