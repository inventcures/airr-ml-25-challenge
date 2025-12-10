import os
from pathlib import Path

BASE_DIR = Path("data/embeddings")

def enforce_clean_structure():
    print("🛡️  Enforcing Clean Structure Policy...")
    
    if not BASE_DIR.exists():
        print("❌ Base directory data/embeddings not found.")
        return

    # Iterate over ds1, ds2, ...
    for ds_dir in BASE_DIR.iterdir():
        if not ds_dir.is_dir() or not ds_dir.name.startswith("ds"):
            continue
            
        print(f"🔍 Inspecting {ds_dir.name}...")
        
        # Find loose .npy files in the root of ds_dir
        loose_files = list(ds_dir.glob("*.npy"))
        
        if loose_files:
            print(f"  ⚠️  Found {len(loose_files)} loose files in root. Cleaning up...")
            for f in loose_files:
                f.unlink()
            print(f"  ✅ Deleted {len(loose_files)} loose files.")
        else:
            print("  ✅ Clean.")
            
    print("\n✨ Structure enforcement complete. Only subdirectories remain.")

if __name__ == "__main__":
    enforce_clean_structure()
