import os
import shutil
import random
from pathlib import Path
import yaml
from collections import defaultdict

def prepare_compcars_for_yolo(compcars_root, output_dir, samples_per_make=800):
    """
    Convert CompCars to YOLO format
    Focus on balanced dataset
    """
    
    compcars_path = Path(compcars_root)
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("Scanning CompCars dataset...")
    
    # Find image directory (adjust based on actual structure)
    # CompCars usually has: data/image/{make}/{model}/year/{image}.jpg
    image_root = compcars_path / 'data' / 'image'
    
    if not image_root.exists():
        # Try alternative structure
        image_root = compcars_path / 'image'
    
    if not image_root.exists():
        print(f"❌ Could not find image directory in {compcars_path}")
        print("Available directories:")
        for item in compcars_path.iterdir():
            print(f"  - {item.name}")
        return
    
    make_to_id = {}
    all_samples = defaultdict(list)
    
    # Scan all images
    print("Collecting images by make...")
    for make_dir in image_root.iterdir():
        if not make_dir.is_dir():
            continue
        
        make_name = make_dir.name
        if make_name not in make_to_id:
            make_to_id[make_name] = len(make_to_id)
        
        # Recursively find all jpg images
        for img_file in make_dir.rglob('*.jpg'):
            all_samples[make_name].append(img_file)
    
    print(f"\nFound {len(make_to_id)} makes:")
    for make, count in sorted([(m, len(all_samples[m])) for m in all_samples.keys()], 
                               key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {make}: {count} images")
    
    # Balance and split dataset
    dataset_splits = {'train': [], 'val': [], 'test': []}
    
    print(f"\nBalancing dataset (max {samples_per_make} per make)...")
    for make_name, images in all_samples.items():
        # Sample to balance
        if len(images) > samples_per_make:
            images = random.sample(images, samples_per_make)
        
        # Shuffle
        random.shuffle(images)
        
        # Split: 70% train, 15% val, 15% test
        n = len(images)
        train_n = int(n * 0.70)
        val_n = int(n * 0.15)
        
        dataset_splits['train'].extend([(img, make_name) for img in images[:train_n]])
        dataset_splits['val'].extend([(img, make_name) for img in images[train_n:train_n+val_n]])
        dataset_splits['test'].extend([(img, make_name) for img in images[train_n+val_n:]])
    
    # Copy files and create YOLO labels
    print("\nCopying files and creating labels...")
    for split, samples in dataset_splits.items():
        print(f"  {split}: {len(samples)} images...")
        
        for idx, (img_path, make_name) in enumerate(samples):
            # Create unique filename
            new_img_name = f"{make_name}_{idx:06d}.jpg"
            
            # Copy image
            shutil.copy(img_path, output_path / 'images' / split / new_img_name)
            
            # Create YOLO label (full image bbox since we don't have bbox annotations)
            label_content = f"{make_to_id[make_name]} 0.5 0.5 0.9 0.9\n"
            
            label_file = output_path / 'labels' / split / new_img_name.replace('.jpg', '.txt')
            with open(label_file, 'w') as f:
                f.write(label_content)
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(make_to_id),
        'names': [name for name, _ in sorted(make_to_id.items(), key=lambda x: x[1])]
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"\n✅ Dataset prepared!")
    print(f"   Classes: {len(make_to_id)}")
    print(f"   Train: {len(dataset_splits['train'])}")
    print(f"   Val: {len(dataset_splits['val'])}")
    print(f"   Test: {len(dataset_splits['test'])}")
    print(f"   Output: {output_path}")

if __name__ == "__main__":
    prepare_compcars_for_yolo(
        compcars_root="datasets/compcars",
        output_dir="datasets/compcars_yolo",
        samples_per_make=800
    )