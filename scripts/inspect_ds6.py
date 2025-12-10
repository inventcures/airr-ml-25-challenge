import os
from pathlib import Path

def inspect_ds6():
    target_dir = Path("data/embeddings/ds6/test")
    print(f"🕵️‍♀️ Inspecting {target_dir}...")
    
    if not target_dir.exists():
        print("❌ Directory not found.")
        return

    files = sorted([f.name for f in target_dir.glob("*.npy")])
    count = len(files)
    print(f"📊 Total files: {count}")
    
    print("\n--- First 20 Files ---")
    for f in files[:20]:
        print(f)
        
    print("\n--- Last 20 Files ---")
    for f in files[-20:]:
        print(f)
        
    # Check for naming patterns
    hashes = [f for f in files if len(f) > 10 and "npy" in f]
    indices = [f for f in files if f.replace(".npy", "").isdigit()]
    
    print(f"\n--- Analysis ---")
    print(f"# Hash-like filenames: {len(hashes)}")
    print(f"# Index-like filenames: {len(indices)}")
    
    if len(hashes) > 0 and len(indices) > 0:
        print("⚠️  MIXED NAMING CONVENTIONS DETECTED!")
        print("This explains the extra files. You have both '0.npy' and 'abc123...npy'.")

if __name__ == "__main__":
    inspect_ds6()
