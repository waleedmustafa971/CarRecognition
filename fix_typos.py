from pathlib import Path
import shutil

def fix_brand_typos(dataset_path):
    """Fix brand name typos in the unified dataset"""
    
    print("="*60)
    print("FIXING BRAND NAME TYPOS")
    print("="*60)
    
    typo_fixes = {
        'BWM': 'BMW',
        'Lamorghini': 'Lamborghini',
        'Chevy': 'Chevrolet',  # Standardize
    }
    
    for split in ['train', 'val']:
        split_path = Path(dataset_path) / split
        
        for typo, correct in typo_fixes.items():
            typo_folder = split_path / typo
            correct_folder = split_path / correct
            
            if typo_folder.exists():
                print(f"\n📁 {split}/{typo} → {split}/{correct}")
                
                # Create correct folder if doesn't exist
                correct_folder.mkdir(exist_ok=True)
                
                # Move all files from typo folder to correct folder
                files = list(typo_folder.glob("*"))
                print(f"   Moving {len(files)} files...")
                
                for file in files:
                    dest = correct_folder / file.name
                    shutil.move(str(file), str(dest))
                
                # Remove empty typo folder
                typo_folder.rmdir()
                print(f"   ✅ Fixed!")
    
    print("\n" + "="*60)
    print("TYPOS FIXED!")
    print("="*60)

if __name__ == "__main__":
    DATASET_PATH = "models/unified_car_brand"
    
    print("\n🔧 This will fix:")
    print("  - BWM → BMW")
    print("  - Lamorghini → Lamborghini")
    print("  - Chevy → Chevrolet\n")
    
    input("Press Enter to continue...")
    
    fix_brand_typos(DATASET_PATH)