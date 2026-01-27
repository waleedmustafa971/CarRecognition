"""
Brand Coverage Analysis Tool
Run this to see which brands each model knows about
"""

import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_model_brands(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ Not found: {model_path}")
        return set()
    
    try:
        model = YOLO(model_path)
        brands = set(model.names.values())
        print(f"✓ Loaded: {len(brands)} classes")
        print(f"\nBrands:")
        for brand in sorted(brands):
            print(f"  - {brand}")
        return brands
    except Exception as e:
        print(f"❌ Error: {e}")
        return set()


def main():
    models = {
        'UNIFIED (epoch10)': os.path.join(BASE_DIR, 'runs', 'classify', 'unified_car_brand', 'weights', 'epoch10.pt'),
        'UNIFIED (best)': os.path.join(BASE_DIR, 'runs', 'classify', 'unified_car_brand', 'weights', 'best.pt'),
        'LOGO': os.path.join(BASE_DIR, 'models', 'car_logo', 'weights', 'best.pt'),
        'COMPCARS': os.path.join(BASE_DIR, 'models', 'compcars', 'best.pt'),
    }
    
    all_brands = {}
    
    for name, path in models.items():
        brands = check_model_brands(path, name)
        all_brands[name] = brands
    
    print(f"\n{'='*60}")
    print("COVERAGE ANALYSIS")
    print(f"{'='*60}")
    
    uae_common = {
        'Toyota', 'TOYOTA', 'Nissan', 'NISSAN', 'Honda', 'HONDA',
        'Hyundai', 'HYUNDAI', 'Kia', 'KIA', 'Mitsubishi', 'MITSUBISHI',
        'Mazda', 'MAZDA', 'Lexus', 'LEXUS', 'Infiniti', 'INFINITI',
        'BMW', 'Mercedes-Benz', 'MERCEDES-BENZ', 'Mercedes', 'MERCEDES',
        'Audi', 'AUDI', 'Porsche', 'PORSCHE', 'Land Rover', 'LAND ROVER',
        'Ford', 'FORD', 'Chevrolet', 'CHEVROLET', 'GMC', 'Jeep', 'JEEP',
        'Volkswagen', 'VOLKSWAGEN', 'VW', 'Tesla', 'TESLA',
        'Bentley', 'BENTLEY', 'Rolls-Royce', 'ROLLS-ROYCE',
        'Ferrari', 'FERRARI', 'Lamborghini', 'LAMBORGHINI',
        'Jaguar', 'JAGUAR', 'Volvo', 'VOLVO', 'MG',
    }
    
    for name, brands in all_brands.items():
        if not brands:
            continue
        
        brands_upper = {b.upper() for b in brands}
        uae_upper = {b.upper() for b in uae_common}
        
        covered = brands_upper & uae_upper
        missing = uae_upper - brands_upper
        
        print(f"\n{name}:")
        print(f"  UAE brands covered: {len(covered)}/{len(uae_upper)}")
        
        if missing:
            print(f"  Missing UAE brands:")
            for brand in sorted(missing):
                print(f"    - {brand}")
    
    if 'UNIFIED (epoch10)' in all_brands and 'LOGO' in all_brands:
        unified = {b.upper() for b in all_brands['UNIFIED (epoch10)']}
        logo = {b.upper() for b in all_brands['LOGO']}
        
        only_unified = unified - logo
        only_logo = logo - unified
        both = unified & logo
        
        print(f"\n{'='*60}")
        print("MODEL OVERLAP")
        print(f"{'='*60}")
        print(f"In both models: {len(both)}")
        print(f"Only in UNIFIED: {len(only_unified)}")
        print(f"Only in LOGO: {len(only_logo)}")
        
        if only_logo:
            print(f"\nBrands only LOGO knows:")
            for brand in sorted(only_logo)[:20]:
                print(f"  - {brand}")


if __name__ == "__main__":
    main()