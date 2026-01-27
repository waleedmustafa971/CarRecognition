from ultralytics import YOLO
import cv2
from pathlib import Path

def test_unified_model(test_folder="test_images"):
    print("="*60)
    print("TESTING UNIFIED MODEL")
    print("="*60)
    
    model_path = "runs/classify/unified_car_brand/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📦 Loading model...")
    model = YOLO(model_path)
    
    # Get test images
    test_path = Path(test_folder)
    if not test_path.exists():
        print(f"❌ Test folder not found: {test_folder}")
        print(f"   Create folder and add test images!")
        return
    
    image_files = list(test_path.glob("*.jpg")) + \
                 list(test_path.glob("*.jpeg")) + \
                 list(test_path.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {test_folder}")
        return
    
    print(f"\n📷 Testing on {len(image_files)} images...")
    
    for img_file in image_files:
        print(f"\n{'='*60}")
        print(f"Image: {img_file.name}")
        print(f"{'='*60}")
        
        # Run prediction
        results = model(str(img_file), verbose=False)
        
        # Get top 5 predictions
        for result in results:
            probs = result.probs
            
            if probs is not None:
                # Get top 5
                top5_idx = probs.top5
                top5_conf = probs.top5conf.cpu().numpy()
                
                print(f"\n🎯 Top 5 Predictions:")
                print("-" * 60)
                for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf), 1):
                    brand = model.names[idx]
                    print(f"  #{i}: {brand:25} → {conf:.3f} ({conf*100:.1f}%)")
                
                # Highlight if confident
                if top5_conf[0] > 0.8:
                    print(f"\n✅ HIGH CONFIDENCE: {model.names[top5_idx[0]]}")
                elif top5_conf[0] > 0.5:
                    print(f"\n⚠️  MEDIUM CONFIDENCE: {model.names[top5_idx[0]]}")
                else:
                    print(f"\n❌ LOW CONFIDENCE - Model unsure")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    test_folder = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    
    print(f"\n🧪 Testing unified model on: {test_folder}")
    print(f"   Place your test images in this folder\n")
    
    test_unified_model(test_folder)