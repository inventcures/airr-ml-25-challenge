import pickle
from pathlib import Path
import pandas as pd
import sys
import os

# Ensure root is in path
sys.path.append(os.getcwd())

from data.load_data import Repertoire

def count_stats():
    data_dir = Path("data/processed")
    
    # Define all datasets and their splits
    datasets = {
        "ds1": ["train", "test"],
        "ds2": ["train", "test"],
        "ds3": ["train", "test"],
        "ds4": ["train", "test"],
        "ds5": ["train", "test"],
        "ds6": ["train", "test"],
        "ds7": ["train", "1_test", "2_test"],
        "ds8": ["train", "1_test", "2_test", "3_test"]
    }

    train_stats = []
    test_stats = []

    for ds_name, splits in datasets.items():
        for split in splits:
            # Handle special naming for ds7/ds8 test sets
            if "_" in split: # e.g. 1_test
                filename = f"{ds_name}_{split}.pkl"
                display_split = split
            else:
                filename = f"{ds_name}_{split}.pkl"
                display_split = split

            pkl_path = data_dir / filename
            
            stat_entry = {
                "Dataset": ds_name,
                "Split": display_split,
                "Repertoires": "N/A",
                "Total Sequences": "N/A",
                "Avg Seqs/Rep": "N/A"
            }

            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        reps = pickle.load(f)
                    
                    n_reps = len(reps)
                    n_seqs = sum(len(r.junction_aa) for r in reps)
                    avg_seqs = int(n_seqs / n_reps) if n_reps > 0 else 0
                    
                    stat_entry = {
                        "Dataset": ds_name,
                        "Split": display_split,
                        "Repertoires": n_reps,
                        "Total Sequences": n_seqs,
                        "Avg Seqs/Rep": avg_seqs
                    }
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")

            if "train" in split:
                train_stats.append(stat_entry)
            else:
                test_stats.append(stat_entry)

    # Print Train Table
    print("\n=== TRAIN DATASETS ===")
    print(f"{'Dataset':<10} {'Split':<10} {'Repertoires':<15} {'Total Sequences':<20} {'Avg Seqs/Rep':<15}")
    print("-" * 75)
    for s in train_stats:
        print(f"{s['Dataset']:<10} {s['Split']:<10} {s['Repertoires']:<15} {s['Total Sequences']:<20} {s['Avg Seqs/Rep']:<15}")

    # Print Test Table
    print("\n=== TEST DATASETS ===")
    print(f"{'Dataset':<10} {'Split':<10} {'Repertoires':<15} {'Total Sequences':<20} {'Avg Seqs/Rep':<15}")
    print("-" * 75)
    for s in test_stats:
        print(f"{s['Dataset']:<10} {s['Split']:<10} {s['Repertoires']:<15} {s['Total Sequences']:<20} {s['Avg Seqs/Rep']:<15}")


if __name__ == "__main__":
    count_stats()
