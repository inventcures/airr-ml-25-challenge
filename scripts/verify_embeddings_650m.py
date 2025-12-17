
import logging
import argparse
from pathlib import Path
import sys
import numpy as np
import random
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.load_all_datasets import load_repertoires_pickle, TRAIN_DATASETS, TEST_DATASETS

# Config
EMBEDDINGS_DIR = Path("data/embeddings")
PROCESSED_DIR = Path("data/processed")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_dataset(ds_name: str, split: str = "train"):
    logging.info(f"--- Verifying {ds_name} {split} ---")
    
    # 1. Load Expected Repertoires
    suffix = f"_{split}" if split == "test" else "_train"
    pkl_path = PROCESSED_DIR / f"{ds_name}{suffix}.pkl"
    
    if not pkl_path.exists():
        # Fallback for old naming
        pkl_path = PROCESSED_DIR / f"{ds_name}.pkl"
        if not pkl_path.exists():
            logging.warning(f"  ⚠️ Skipping {ds_name} {split} (No Pickle found at {pkl_path})")
            return

    reps = load_repertoires_pickle(pkl_path)
    expected_count = len(reps)
    logging.info(f"  Expected Count: {expected_count}")

    # 2. Determine Candidate Paths (where NPY files might hide)
    # Similar to generate_ensemble_preds, we look in multiple places
    base_ds = ds_name.split("_")[0]
    candidate_roots = [
        EMBEDDINGS_DIR / ds_name,
        EMBEDDINGS_DIR / ds_name / split,
        EMBEDDINGS_DIR / base_ds / split,
    ]
    
    # Add manual sharded paths if they exist
    if split == "test":
        base_test = EMBEDDINGS_DIR / base_ds / "test"
        if base_test.exists():
            for i in range(1, 10):
                if (base_test / f"{i}_test").exists():
                    candidate_roots.append(base_test / f"{i}_test")

    valid_roots = [p for p in candidate_roots if p.exists()]
    
    if not valid_roots:
        logging.error(f"  ❌ No valid embedding directories found for {ds_name}. (Checked {candidate_roots})")
        return

    # 3. Check for Existence
    found_count = 0
    missing_ids = []
    
    # Check a random sample first for speed (if huge)
    # But since user wants sanity check, we iterate all (file system stat is fast enough for <100k files)
    
    for r in tqdm(reps, desc=f"Checking {ds_name}", leave=False):
        found = False
        for root in valid_roots:
            if (root / f"{r.rep_id}.npy").exists():
                found = True
                break
        
        if found:
            found_count += 1
        else:
            missing_ids.append(r.rep_id)

    # 4. Report
    completeness = (found_count / expected_count) * 100
    if completeness > 99.9:
        logging.info(f"  ✅ GREEN LIGHT: {found_count}/{expected_count} ({completeness:.1f}%) found.")
    elif completeness > 90:
        logging.warning(f"  ⚠️ AMBER LIGHT: {found_count}/{expected_count} ({completeness:.1f}%) found. (Missing {len(missing_ids)})")
    else:
        logging.error(f"  ❌ RED LIGHT: Only {found_count}/{expected_count} ({completeness:.1f}%) found.")
    
    # 5. Spot Check Data Integrity (Random 5 files)
    if found_count > 0:
        logging.info("  🔍 Spot checking 3 random files for corruption...")
        sample_ids = random.sample([r.rep_id for r in reps if r.rep_id not in missing_ids], min(3, found_count))
        for rid in sample_ids:
            # Find it again
            fpath = None
            for root in valid_roots:
                if (root / f"{rid}.npy").exists():
                    fpath = root / f"{rid}.npy"
                    break
            
            try:
                data = np.load(fpath, mmap_mode='r')
                shape = data.shape
                if shape[1] != 1280: # ESM 650m dim
                     logging.error(f"    ❌ CORRUPTION: {rid} has wrong shape {shape} (Expected 1280 dim)")
                else:
                     logging.info(f"    ✅ OK: {rid} shape {shape}")
            except Exception as e:
                logging.error(f"    ❌ CORRUPTION: {rid} could not be loaded: {e}")

    # Return failure status
    success = (completeness > 90) # Consider Amber light as "Pass" for now, or strict 99.9? User said "Accuracy" is key.
    # Let's be strict: If Red Light, we fail. Amber is warn.
    if completeness < 90:
        return False
    return True

def main():
    logging.info("🚀 Starting 650M Embedding Sanity Check...")
    
    failed_datasets = []

    # DS1-6 (Train/Test)
    for ds in ["ds1", "ds2", "ds3", "ds4", "ds5", "ds6"]:
        if not verify_dataset(ds, "train"): failed_datasets.append(f"{ds}_train")
        if not verify_dataset(ds, "test"): failed_datasets.append(f"{ds}_test")

    # DS7 (Train/Test)
    if not verify_dataset("ds7", "train"): failed_datasets.append("ds7_train")
    if not verify_dataset("ds7", "test"): failed_datasets.append("ds7_test")

    # DS8 (Train/Test)
    if not verify_dataset("ds8", "train"): failed_datasets.append("ds8_train")
    if not verify_dataset("ds8", "test"): failed_datasets.append("ds8_test")

    if failed_datasets:
        logging.error(f"❌ Sanity Check Failed for: {failed_datasets}")
        sys.exit(1)
    
    logging.info("🏁 Sanity Check Complete: ALL GREEN/AMBER.")

if __name__ == "__main__":
    main()
