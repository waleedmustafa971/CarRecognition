import os
import sys
from ultralytics import YOLO

print("\n" + "="*80)
print("MODEL BRANDS DIAGNOSTIC")
print("Check which brands each model can detect")
print("="*80 + "\n")

base_dir = r"C:\Users\Administrator\Desktop\car_recognition"

models_to_check = {
    "EPOCH10 (Old Models 2013-2019)": os.path.join(base_dir, "runs", "classify", "unified_car_brand", "weights", "epoch10.pt"),
    "BEST (Old Models 2013-2019)": os.path.join(base_dir, "runs", "classify", "unified_car_brand", "weights", "best.pt"),
    "LOGO (New Models)": os.path.join(base_dir, "models", "car_logo", "weights", "best.pt"),
    "KIA (Specialist)": os.path.join(base_dir, "models", "kia", "weights", "best.pt"),
}

all_brands = {}

for model_name, model_path in models_to_check.items():
    print(f"\n{model_name}")
    print("-" * 80)
    print(f"Path: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ NOT FOUND")
        continue
    
    try:
        model = YOLO(model_path)
        brands = list(model.names.values())
        
        print(f"✓ Loaded - {len(brands)} brands")
        
        all_brands[model_name] = set([b.upper() for b in brands])
        
        print(f"\nBrands (first 20):")
        for i, brand in enumerate(sorted(brands)[:20], 1):
            print(f"  {i:2d}. {brand}")
        
        if len(brands) > 20:
            print(f"  ... and {len(brands) - 20} more brands")
        
        print(f"\nSearching for specific brands:")
        search_brands = ['MG', 'VOLKSWAGEN', 'VW', 'TOYOTA', 'KIA', 'HYUNDAI', 'BMW', 'AUDI', 'MERCEDES']
        for search in search_brands:
            found = [b for b in brands if search.upper() in b.upper()]
            if found:
                print(f"  ✓ {search}: {', '.join(found)}")
            else:
                print(f"  ❌ {search}: Not found")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")

print("\n" + "="*80)
print("BRAND COVERAGE ANALYSIS")
print("="*80)

if len(all_brands) >= 2:
    print("\nChecking which brands are ONLY in specific models:\n")
    
    all_model_names = list(all_brands.keys())
    
    for model_name in all_model_names:
        other_models = [m for m in all_model_names if m != model_name]
        
        unique_brands = all_brands[model_name].copy()
        for other in other_models:
            unique_brands = unique_brands - all_brands[other]
        
        if unique_brands:
            print(f"{model_name} - UNIQUE brands ({len(unique_brands)}):")
            for brand in sorted(list(unique_brands))[:10]:
                print(f"  • {brand}")
            if len(unique_brands) > 10:
                print(f"  ... and {len(unique_brands) - 10} more")
            print()

print("\n" + "="*80)
print("MG DETECTION ANALYSIS")
print("="*80)

mg_found_in = []
for model_name, brands in all_brands.items():
    mg_variants = [b for b in brands if 'MG' in b]
    if mg_variants:
        mg_found_in.append(model_name)
        print(f"\n✓ {model_name}:")
        print(f"  MG variants: {', '.join(mg_variants)}")

if not mg_found_in:
    print("\n❌ MG NOT FOUND in any model!")
    print("   This explains why MG cars are misclassified")
    print("\n   SOLUTION:")
    print("   1. Check if logo model was trained on MG")
    print("   2. If not, retrain logo model with MG examples")
    print("   3. Or train a separate MG specialist model")
else:
    print(f"\n✓ MG found in {len(mg_found_in)} model(s)")
    print("   If MG cars are still misclassified, the issue is:")
    print("   1. Low confidence predictions")
    print("   2. Other models overriding with higher confidence")
    print("   3. Need to boost weight of models that know MG")

print()