import sys
from pathlib import Path

# Expected counts
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

BASE_LOCAL_DIR = Path("data/embeddings")

def count_local_files(ds, split):
    target_dir = BASE_LOCAL_DIR / ds / split
    if not target_dir.exists():
        return 0
    return len(list(target_dir.glob("*.npy")))

def main():
    print(f"{'DATASET':<10} {'SPLIT':<10} {'STATUS':<10} {'PROGRESS':<15}")
    print("-" * 50)
    
    all_good = True
    
    for item in EXPECTED_COUNTS:
        ds = item["dataset"]
        split = item["split"]
        expected = item["reps"]
        
        current = count_local_files(ds, split)
        
        if current >= expected:
            status = "✅ OK"
        elif current == 0:
            status = "❌ MISSING"
            all_good = False
        else:
            status = "⚠️  PARTIAL"
            all_good = False
            
        print(f"{ds:<10} {split:<10} {status:<10} {current}/{expected}")
        
    print("-" * 50)
    if all_good:
        print("🎉 All datasets verified successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some datasets are missing or incomplete.")
        sys.exit(1)

if __name__ == "__main__":
    main()
