import os
from ultralytics import YOLO
from pathlib import Path

# ==================== MODEL CONFIGURATION ====================
MODEL_PATHS = {
    'logo': ['models/car_logo/weights/best.pt', 'car_logo.pt', 'models/car_logo.pt'],
    'make': ['models/car_make/weights/best.pt', 'car_make.pt', 'models/car_make.pt'],
    'stanford': ['models/stanford_cars/weights/best.pt', 'stanford_cars.pt', 'models/stanford_cars.pt'],
    'compcars': ['models/compcars/best.pt', 'runs/compcars/universal_v1/weights/best.pt'],
    'kia': ['models/kia/weights/best.pt', 'models/kia/kia_v1/weights/best.pt'],
    'color': ['models/car_color/weights/best.pt', 'models/car_color/car_color_v1/weights/best.pt'],
    # Disabled models
    'tesla': ['models/tesla/weights/best.pt', 'models/tesla/tesla_v1/weights/best.pt'],
    'mclaren': ['models/mclaren/weights/best.pt', 'models/mclaren/mclaren_v1/weights/best.pt'],
}

# Model weights from hierarchical system
MODEL_WEIGHTS = {
    'logo': 1.8,
    'make': 1.2,
    'stanford': 1.0,
    'compcars': 0.7,
    'kia': 'specialist',
    'tesla': 'specialist (disabled)',
    'mclaren': 'specialist (disabled)',
}

# Confidence thresholds
MODEL_THRESHOLDS = {
    'logo': 0.30,
    'make': 0.45,
    'stanford': 0.45,
    'compcars': 0.55,
    'kia': 0.85,
    'tesla': 0.85,
    'mclaren': 0.92,
}

def load_models():
    """Load all available models"""
    models = {}
    
    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    
    for model_name, possible_paths in MODEL_PATHS.items():
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    models[model_name] = YOLO(path)
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    print(f"✓ Loaded {model_name:15} from {path:50} ({size_mb:.1f} MB)")
                    break
                except Exception as e:
                    print(f"✗ Failed to load {path}: {e}")
                    continue
    
    if not models:
        print("\n⚠️  No models found!")
    
    print("="*80)
    print(f"Total models loaded: {len(models)}")
    print("="*80)
    print()
    
    return models

def get_brand_coverage(models):
    """Get which brands are covered by which models"""
    
    # Brand mapping (normalize names)
    brand_aliases = {
        'volkswagen': 'Volkswagen',
        'wolksvagen': 'Volkswagen',
        'vw': 'Volkswagen',
        'benz': 'Mercedes-Benz',
        'mercedes': 'Mercedes-Benz',
        'mercedes-benz': 'Mercedes-Benz',
        'range rover': 'Land Rover',
        'land rover': 'Land Rover',
        'mg': 'Morris Garages',
        'morris garages': 'Morris Garages',
        'peugeot': 'PEUGEOT',
        'pegeout': 'PEUGEOT',
        'porsche': 'Porsche',
        'porche': 'Porsche',
        'toyota': 'Toyota',
        'toyata': 'Toyota',
        'rolls-royce': 'Rolls-Royce',
        'rolls roys': 'Rolls-Royce',
    }
    
    brand_coverage = {}
    
    for model_name, model in models.items():
        if model_name == 'color':
            continue  # Skip color model
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        print(f"Weight: {MODEL_WEIGHTS.get(model_name, 'N/A')}")
        print(f"Threshold: {MODEL_THRESHOLDS.get(model_name, 'N/A')}")
        print(f"Total classes: {len(model.names)}")
        print(f"\nBrands in this model:")
        print("-"*80)
        
        # Get all class names
        brands_in_model = []
        for class_id, class_name in model.names.items():
            # Normalize brand name
            normalized = class_name.lower().strip()
            
            # Apply aliases
            if normalized in brand_aliases:
                normalized_name = brand_aliases[normalized]
            else:
                normalized_name = class_name
            
            brands_in_model.append(normalized_name)
            
            # Add to coverage dict
            if normalized_name not in brand_coverage:
                brand_coverage[normalized_name] = []
            
            brand_coverage[normalized_name].append({
                'model': model_name,
                'weight': MODEL_WEIGHTS.get(model_name, 1.0),
                'threshold': MODEL_THRESHOLDS.get(model_name, 0.5),
                'original_name': class_name
            })
        
        # Sort and display
        brands_in_model = sorted(set(brands_in_model))
        for idx, brand in enumerate(brands_in_model, 1):
            print(f"{idx:3}. {brand}")
    
    return brand_coverage

def analyze_working_brands(brand_coverage, working_brands):
    """Analyze which models detect the working brands"""
    
    print("\n\n" + "="*80)
    print("WORKING BRANDS ANALYSIS")
    print("="*80)
    print("\nThese brands are correctly detected by the system:")
    print("-"*80)
    
    for idx, brand in enumerate(working_brands, 1):
        # Normalize brand name
        normalized = brand.strip()
        
        # Find in coverage
        found = False
        for covered_brand, models in brand_coverage.items():
            if normalized.lower() in covered_brand.lower() or covered_brand.lower() in normalized.lower():
                found = True
                
                print(f"\n{idx}. {brand}")
                print(f"   Detected as: {covered_brand}")
                print(f"   Available in {len(models)} model(s):")
                
                for model_info in models:
                    status = "✓ ACTIVE" if model_info['model'] not in ['tesla', 'mclaren'] else "✗ DISABLED"
                    print(f"      - {model_info['model']:12} | Weight: {str(model_info['weight']):20} | Threshold: {model_info['threshold']:.2f} | {status}")
                
                break
        
        if not found:
            print(f"\n{idx}. {brand}")
            print(f"   ⚠️  NOT FOUND in any model!")
            print(f"   → Likely detected via API fallback")

def show_summary(models, brand_coverage):
    """Show summary statistics"""
    
    print("\n\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal models loaded: {len(models)}")
    print(f"Active models: {len([m for m in models.keys() if m not in ['tesla', 'mclaren', 'color']])}")
    print(f"Disabled models: 2 (tesla, mclaren)")
    
    print(f"\nTotal unique brands across all models: {len(brand_coverage)}")
    
    # Brands by model count
    single_model = [b for b, m in brand_coverage.items() if len(m) == 1]
    multi_model = [b for b, m in brand_coverage.items() if len(m) > 1]
    
    print(f"Brands in only 1 model: {len(single_model)}")
    print(f"Brands in multiple models: {len(multi_model)}")
    
    # Top covered brands
    top_brands = sorted(brand_coverage.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    print(f"\nTop 10 most covered brands:")
    print("-"*80)
    for idx, (brand, models) in enumerate(top_brands, 1):
        model_names = [m['model'] for m in models]
        print(f"{idx:2}. {brand:20} → {len(models)} models: {', '.join(model_names)}")

def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("CAR BRAND MODEL ANALYSIS")
    print("="*80)
    
    # Load all models
    models = load_models()
    
    if not models:
        print("\n⚠️  No models found! Please check MODEL_PATHS.")
        return
    
    # Get brand coverage
    brand_coverage = get_brand_coverage(models)
    
    # Your working brands list
    working_brands = [
        'Volkswagen',
        'Volvo',
        'Toyota',
        'SUZUKI',
        'Land Rover',
        'Renault',
        'Porsche',
        'NISSAN',
        'PEUGEOT',
        'Mitsubishi',
        'Rolls-Royce',
        'Morris Garages',
        'MAZDA',
        'LEXUS',
        'LINCOLN',
        'KIA',
        'Jeep',
        'HYUNDAI',
        'Infiniti',
        'HONDA',
        'GMC',
        'GEELY',
        'Ford',
        'Chevrolet',
        'BMW',
        'Mercedes-Benz',
        'Audi',
        'HAVAL'
    ]
    
    # Analyze working brands
    analyze_working_brands(brand_coverage, working_brands)
    
    # Show summary
    show_summary(models, brand_coverage)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()