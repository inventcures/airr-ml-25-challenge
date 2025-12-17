
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_all_datasets import load_repertoires_pickle, TRAIN_DATASETS, TEST_DATASETS, PROCESSED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, default="submission.csv", help="Path to submission file")
    parser.add_argument("--out", type=str, default="submission_filled.csv", help="Output path")
    parser.add_argument("-m", "--model_size", type=str, default="650")
    args = parser.parse_args()

    # Determine paths based on 650/35 switch (if needed, but submission.csv path is explicit usually)
    # If default is used, we try to be smart
    sub_path = Path(args.submission)
    if not sub_path.exists():
        # Try 650m default
        sub_path = Path("submission_650m.csv")
    
    if not sub_path.exists():
        logging.error(f"❌ Submission file not found at {sub_path}")
        return

    logging.info(f"Reading {sub_path}...")
    df = pd.read_csv(sub_path)
    
    existing_datasets = set(df["dataset"].unique())
    logging.info(f"Found datasets: {existing_datasets}")
    
    new_rows = []
    
    # 1. Check Train Datasets (Task 2)
    for ds_key, ds_name in TRAIN_DATASETS.items():
        if ds_name not in existing_datasets:
            logging.warning(f"⚠️ Missing TRAIN dataset: {ds_name}. Generating 50,000 dummy rows...")
            
            # For Train, we need 50k rows with IDs "dataset_seq_top_0"..."dataset_seq_top_49999"
            # And some random junction_aa/v/j
            
            # Load pickle just to get REAL sequences if possible? 
            # If we don't have rankings, we can't really pick "Top".
            # But we can pick 50,000 RANDOM sequences from the pickle to be safe?
            # Or just "A" * 10.
            
            pkl_path = PROCESSED_DIR / f"{ds_key}_train.pkl"
            if not pkl_path.exists(): pkl_path = PROCESSED_DIR / f"{ds_key}.pkl"
            
            dummy_seqs = ["CASSLIGGTYEQYF"] * 50000 # Default fallback
            dummy_v = ["TRBV20-1"] * 50000
            dummy_j = ["TRBJ2-7"] * 50000
            
            if pkl_path.exists():
                try:
                    reps = load_repertoires_pickle(pkl_path)
                    # Collect all sequences
                    all_seqs = []
                    all_v = []
                    all_j = []
                    for r in reps:
                        for s, v, j in zip(r.junction_aa, r.v_call, r.j_call):
                             all_seqs.append(s)
                             all_v.append(v)
                             all_j.append(j)
                             if len(all_seqs) >= 50000: break
                        if len(all_seqs) >= 50000: break
                    
                    if len(all_seqs) > 0:
                        # Pad or trim
                        if len(all_seqs) < 50000:
                            # Tile
                            factor = (50000 // len(all_seqs)) + 1
                            all_seqs = (all_seqs * factor)[:50000]
                            all_v = (all_v * factor)[:50000]
                            all_j = (all_j * factor)[:50000]
                        else:
                            all_seqs = all_seqs[:50000]
                            all_v = all_v[:50000]
                            all_j = all_j[:50000]
                        
                        dummy_seqs = all_seqs
                        dummy_v = all_v
                        dummy_j = all_j
                        logging.info(f"  ✅ Harvested real sequences for {ds_name} dummy filler.")
                except Exception as e:
                    logging.warning(f"  Could not load pickle for {ds_name}, using static dummies: {e}")

            # Generate IDs
            ids = [f"{ds_name}_seq_top_{i}" for i in range(50000)]
            
            dummy_df = pd.DataFrame({
                "ID": ids,
                "dataset": ds_name,
                "label_positive_probability": 0.5, # Safe neutral
                "junction_aa": dummy_seqs,
                "v_call": dummy_v,
                "j_call": dummy_j
            })
            new_rows.append(dummy_df)

    # 2. Check Test Datasets (Task 1)
    for ds_key, ds_name in TEST_DATASETS.items():
        if ds_name not in existing_datasets:
            logging.warning(f"⚠️ Missing TEST dataset: {ds_name}. Generating dummy rows for repertoire IDs...")
            
            # Load pickle to get IDs
            pkl_path = PROCESSED_DIR / f"{ds_key}_test.pkl"
            if not pkl_path.exists():
                logging.error(f"  ❌ Pickle not found {pkl_path}. Cannot guess Repertoire IDs! Skipping.")
                continue
                
            reps = load_repertoires_pickle(pkl_path)
            rep_ids = [str(r.rep_id) for r in reps]
            
            # Deduplicate just in case
            rep_ids = list(set(rep_ids))
            logging.info(f"  Found {len(rep_ids)} IDs for {ds_name}.")
            
            dummy_df = pd.DataFrame({
                "ID": rep_ids, # Task 1: ID is rep_id
                "dataset": ds_name,
                "label_positive_probability": 0.5,
                "junction_aa": "-999.0", # Masked for test
                "v_call": "-999.0",
                "j_call": "-999.0"
            })
            new_rows.append(dummy_df)

    if new_rows:
        logging.info(f"Injecting {len(new_rows)} missing datasets...")
        filled_df = pd.concat([df] + new_rows, ignore_index=True)
    else:
        logging.info("✅ No missing datasets found. File is complete.")
        filled_df = df

    # --- FINAL SAFETY: REMOVE ALL NULLS ---
    logging.info("🧹 Performing final Null check and fill...")
    
    # 1. Probability -> 0.5
    null_probs = filled_df["label_positive_probability"].isnull().sum()
    if null_probs > 0:
        logging.warning(f"  Found {null_probs} NULL probabilities. Filling with 0.5.")
        filled_df["label_positive_probability"] = filled_df["label_positive_probability"].fillna(0.5)

    # 2. String Columns -> "unknown" or "-999.0" depending on logic, but "unknown" is safer than NaN
    for col in ["junction_aa", "v_call", "j_call", "dataset", "ID"]:
        null_strs = filled_df[col].isnull().sum()
        if null_strs > 0:
             logging.warning(f"  Found {null_strs} NULLs in {col}. Filling with 'unknown'/-999.0.")
             # If it's a test row, ideally -999.0, but 'unknown' is better than crash.
             # Let's simple fill with -999.0 for sequence columns if they look like test data?
             # Just fill with string "unknown" for robustness.
             filled_df[col] = filled_df[col].fillna("unknown")

    # 3. Enforce Test Set Masking Rule again just in case concatenation broke it
    test_mask = filled_df['dataset'].str.contains('test', na=False)
    mask_cols = ['junction_aa', 'v_call', 'j_call']
    # Ensure they are -999.0
    filled_df.loc[test_mask, mask_cols] = "-999.0"

    filled_df.to_csv(args.out, index=False)
    logging.info(f"✅ Saved filled submission to {args.out}")

if __name__ == "__main__":
    main()
