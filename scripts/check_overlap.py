import pickle
from pathlib import Path
import sys
import os

# Ensure root is in path
sys.path.append(os.getcwd())
from data.load_data import Repertoire

def check_overlap(dataset_name="ds1"):
    data_dir = Path("data/processed")
    train_path = data_dir / f"{dataset_name}_train.pkl"
    test_path = data_dir / f"{dataset_name}_test.pkl"
    
    if not train_path.exists() or not test_path.exists():
        print(f"Missing files for {dataset_name}")
        return

    with open(train_path, "rb") as f:
        train_reps = pickle.load(f)
    with open(test_path, "rb") as f:
        test_reps = pickle.load(f)
        
    train_ids = set(r.rep_id for r in train_reps)
    test_ids = set(r.rep_id for r in test_reps)
    
    overlap = train_ids.intersection(test_ids)
    
    print(f"Dataset: {dataset_name}")
    print(f"Train IDs: {len(train_ids)}")
    print(f"Test IDs: {len(test_ids)}")
    print(f"Overlap: {len(overlap)}")
    
    if overlap:
        print(f"EXAMPLE OVERLAP: {list(overlap)[:5]}")
        print("⚠️  WARNING: IDs overlap! Files will be overwritten if stored in same folder.")
    else:
        print("✅ No overlap. IDs are unique.")

if __name__ == "__main__":
    check_overlap("ds1")
