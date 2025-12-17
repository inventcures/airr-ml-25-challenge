
import pandas as pd
from pathlib import Path
import sys
import logging
import argparse

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_size", type=str, choices=["35", "650", "35m", "650m"], default="650")
    args = parser.parse_args()

    if "35" in args.model_size:
        logging.info("🔵 Aggregating 35M Preds...")
        INPUT_DIR = Path("outputs/esm_seq_preds")
        OUTPUT_DIR = Path("outputs/submission")
    else:
        logging.info("🟣 Aggregating 650M Preds...")
        INPUT_DIR = Path("outputs/esm_seq_preds_650m")
        OUTPUT_DIR = Path("outputs/submission_650m")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "submission.csv"

    logging.info(f"Scanning {INPUT_DIR}...")
    
    # We expect files like ds1_train_oof_preds.csv, ds1_test_esm_preds.csv
    files = list(INPUT_DIR.glob("*.csv"))
    
    if not files:
        logging.error(f"❌ No prediction files found in {INPUT_DIR}!")
        sys.exit(1)

    dfs = []
    for f in files:
        logging.info(f"  Reading {f.name}...")
        try:
            df = pd.read_csv(f)
            # Ensure "dataset" column exists if possible, but generate_ensemble_preds usually doesn't add it?
            # Build_submission will look for repertoire_id + dataset match.
            # Wait, predictions usually just need {repertoire_id, probability}. 
            # build_submission.py merges on ["repertoire_id", "dataset"].
            # So we MUST ensure "dataset" column exists here OR build_submission infers it?
            # Let's check build_submission.py:
            # merged = df_task2.merge(df_task1, on=["repertoire_id", "dataset"], how="left")
            # So Task 1 MUST have 'dataset' column.
            
            if "dataset" not in df.columns:
                # Infer from filename
                # standard: ds1_train_oof_preds.csv -> ds1_train (mapped)
                # This is risky. Let's see if generate_ensemble_preds adds it.
                # It appends {"repertoire_id": ..., "p_esm": ...}. No dataset.
                
                # We need to inject 'dataset'.
                name = f.name
                ds_name_raw = name.split("_")[0] # ds1, ds7...
                ds_type = "train" if "train" in name else "test"
                
                # If ds7_shard1 or something? generate_ensemble_preds uses distinct ds names from config.
                
                # However, let's use the explicit mapping from load_all_datasets if possible.
                # Simpler: Just strip common suffixes.
                # Actually, build_submission relies on TRAIN_DATASETS values.
                
                # Let's import the mapping to be safe.
                from data.load_all_datasets import TRAIN_DATASETS, TEST_DATASETS
                
                mapped_name = None
                # Check Train
                for k, v in TRAIN_DATASETS.items():
                    if name.startswith(f"{k}_train") or name.startswith(k) and "train" in name:
                        mapped_name = v # e.g. "train_dataset_1"
                        break
                # Check Test
                if not mapped_name:
                    for k, v in TEST_DATASETS.items():
                        if name.startswith(f"{k}_test") or name.startswith(k) and "test" in name:
                            mapped_name = v
                            break
                            
                if mapped_name:
                    df["dataset"] = mapped_name
                else:
                    logging.warning(f"  ⚠️ Could not infer dataset name for {f.name}. Keeping as is (might fail merge).")

            dfs.append(df)
            
        except Exception as e:
            logging.error(f"  ❌ Error reading {f}: {e}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        # Unique by (repertoire_id, dataset)
        full_df = full_df.drop_duplicates(subset=["repertoire_id", "dataset"])
        
        full_df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"✅ Aggregated {len(full_df)} rows to {OUTPUT_FILE}")
    else:
        logging.error("❌ No data to save.")
        sys.exit(1)

if __name__ == "__main__":
    # Add project root to path for imports
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    main()
