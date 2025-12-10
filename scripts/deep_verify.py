import numpy as np
from pathlib import Path
import random
import sys

# Define expected structure
EXPECTED_COUNTS = [
    {"dataset": "ds1", "split": "train", "reps": 400},
    {"dataset": "ds2", "split": "train", "reps": 400},
    {"dataset": "ds3", "split": "train", "reps": 400},
    {"dataset": "ds4", "split": "train", "reps": 400},
    {"dataset": "ds5", "split": "train", "reps": 400},
    {"dataset": "ds6", "split": "train", "reps": 400},
    {"dataset": "ds7", "split": "train", "reps": 302},
    {"dataset": "ds8", "split": "train", "reps": 908},
    
    {"dataset": "ds1", "split": "test", "reps": 400},
    {"dataset": "ds2", "split": "test", "reps": 400},
    {"dataset": "ds3", "split": "test", "reps": 400},
    {"dataset": "ds4", "split": "test", "reps": 400},
    {"dataset": "ds5", "split": "test", "reps": 400},
    {"dataset": "ds6", "split": "test", "reps": 400},
    
    {"dataset": "ds7", "split": "1_test", "reps": 76},
    {"dataset": "ds7", "split": "2_test", "reps": 100},
    
    {"dataset": "ds8", "split": "1_test", "reps": 390},
    {"dataset": "ds8", "split": "2_test", "reps": 857},
    {"dataset": "ds8", "split": "3_test", "reps": 390},
]

BASE_DIR = Path("data/embeddings")
SAMPLES_PER_SPLIT = 5  # Check 5 random files per split

def deep_verify():
    print("🩺 Starting Deep Verification (Integrity & Shape Check)...")
    print(f"   Checking {SAMPLES_PER_SPLIT} random files per split.\n")
    
    all_good = True
    
    print(f"{'DATASET':<10} {'SPLIT':<10} {'STATUS':<10} {'DETAILS':<30}")
    print("-" * 60)
    
    for item in EXPECTED_COUNTS:
        ds = item["dataset"]
        split = item["split"]
        target_dir = BASE_DIR / ds / split
        
        if not target_dir.exists():
            print(f"{ds:<10} {split:<10} ❌ MISSING  Directory not found")
            all_good = False
            continue
            
        files = list(target_dir.glob("*.npy"))
        if not files:
            print(f"{ds:<10} {split:<10} ❌ EMPTY    No .npy files found")
            all_good = False
            continue
            
        # Sample files
        sample_files = random.sample(files, min(len(files), SAMPLES_PER_SPLIT))
        
        split_ok = True
        error_msg = ""
        
        for f in sample_files:
            try:
                data = np.load(f)
                # Check shape: Should be (N_sequences, 1280)
                if len(data.shape) != 2 or data.shape[1] != 1280:
                    split_ok = False
                    error_msg = f"Bad shape: {data.shape}"
                    break
                # Check for NaNs
                if np.isnan(data).any():
                    split_ok = False
                    error_msg = "Contains NaNs"
                    break
            except Exception as e:
                split_ok = False
                error_msg = f"Corrupt file: {e}"
                break
        
        if split_ok:
            print(f"{ds:<10} {split:<10} ✅ OK       Verified {len(sample_files)} samples")
        else:
            print(f"{ds:<10} {split:<10} ❌ FAILED   {error_msg}")
            all_good = False

    print("-" * 60)
    if all_good:
        print("\n✨ All systems go! Data is valid and ready for training. 🚀")
    else:
        print("\n⚠️  Issues detected. Do not start training yet.")

if __name__ == "__main__":
    deep_verify()
