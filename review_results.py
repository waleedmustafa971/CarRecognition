import pandas as pd
from pathlib import Path

def review_training_results():
    results_dir = Path("runs/classify/unified_car_brand")
    
    print("="*60)
    print("TRAINING RESULTS REVIEW")
    print("="*60)
    
    # Check if results exist
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    # Load results.csv
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        
        # Get final epoch metrics
        final = df.iloc[-1]
        
        print(f"\n📊 Final Metrics (Epoch {int(final['epoch'])}):")
        print(f"   Train Loss: {final['train/loss']:.4f}")
        
        if 'metrics/accuracy_top1' in df.columns:
            print(f"   Top-1 Accuracy: {final['metrics/accuracy_top1']:.2%}")
        if 'metrics/accuracy_top5' in df.columns:
            print(f"   Top-5 Accuracy: {final['metrics/accuracy_top5']:.2%}")
        
        # Best epoch
        if 'metrics/accuracy_top1' in df.columns:
            best_epoch = df['metrics/accuracy_top1'].idxmax()
            best = df.iloc[best_epoch]
            print(f"\n🏆 Best Performance (Epoch {int(best['epoch'])}):")
            print(f"   Top-1 Accuracy: {best['metrics/accuracy_top1']:.2%}")
            if 'metrics/accuracy_top5' in df.columns:
                print(f"   Top-5 Accuracy: {best['metrics/accuracy_top5']:.2%}")
    
    # Check model files
    weights_dir = results_dir / "weights"
    print(f"\n📦 Model Files:")
    if (weights_dir / "best.pt").exists():
        size = (weights_dir / "best.pt").stat().st_size / (1024**2)
        print(f"   ✅ best.pt ({size:.1f} MB)")
    if (weights_dir / "last.pt").exists():
        size = (weights_dir / "last.pt").stat().st_size / (1024**2)
        print(f"   ✅ last.pt ({size:.1f} MB)")
    
    # Check visualization files
    print(f"\n📈 Visualization Files:")
    viz_files = ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png']
    for viz in viz_files:
        if (results_dir / viz).exists():
            print(f"   ✅ {viz}")
    
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    review_training_results()