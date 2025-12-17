
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import logging
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Logging setup
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/build_submission.log")
    ]
)

TASK1_SUBMISSION = Path("outputs/submission/submission.csv")
TASK2_DIR = Path("outputs/task2_ranking")
OUTPUT_CSV = Path("submission.csv")
SUBMISSIONS_ROOT = Path("outputs/submissions")


def validate_submission(df: pd.DataFrame, log_path: Path):
    """
    Performs strict validation checks against contest rules.
    Logs results to both console and summary log.
    """
    logging.info("\n[Validation] Starting strict checks...")
    issues = []
    
    # Check 1: Column Names
    expected_cols = {"ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        issues.append(f"❌ Missing columns: {missing_cols}")
    else:
        logging.info("✅ Columns check passed.")

    # Check 2: No Nulls in critical columns
    if df["dataset"].isnull().any():
        issues.append(f"❌ Null values in 'dataset' column: {df['dataset'].isnull().sum()}")
    if df["label_positive_probability"].isnull().any():
        issues.append(f"❌ Null values in 'label_positive_probability' column: {df['label_positive_probability'].isnull().sum()}")
    else:
        logging.info("✅ Null check passed.")
        
    # Check 3: Row Counts per Dataset
    # Train datasets must have exactly 50,000 sequences (for Task 2)
    # Test datasets just need 1 per repertoire (Task 1 focus)
    ds_counts = df["dataset"].value_counts()
    
    for ds_name, count in ds_counts.items():
        if "train" in ds_name:
            if count != 50000:
                issues.append(f"⚠️ warning: {ds_name} has {count} rows. Expected exactly 50,000 for Task 2.")
            else:
                logging.info(f"✅ {ds_name}: 50,000 rows (Perfect).")
        # Test datasets vary, so we skip specific count validation unless we know exact repertoire counts.

    # Check 4: Value Ranges
    # Task 1 probs should be 0-1. Task 2 dummy rows are ignored (usually same prob).
    # Since we broadcast prob to all rows, all rows should be 0-1.
    if not ((df["label_positive_probability"] >= 0) & (df["label_positive_probability"] <= 1)).all():
         issues.append("❌ Probabilities out of range [0, 1].")
    else:
         logging.info("✅ Probability range check passed.")

    # Final Report
    with open(log_path, "a") as f:
        f.write("\nValidation Report:\n")
        if issues:
            f.write("ISSUES FOUND:\n")
            for i in issues:
                f.write(i + "\n")
                logging.error(i)
            f.write("\nsubmission.csv might be INVALID.\n")
        else:
            f.write("✅ All Checks Passed.\n")
            logging.info("Click 'Submit'! 🚀")


def build_submission():
    logging.info("[Build Submission] Generating final submission file...")
    
    # 1. Load Task 1 results (Repertoire Level)
    if not TASK1_SUBMISSION.exists():
        logging.error(f"Task 1 submission not found: {TASK1_SUBMISSION}")
        return
        
    df_task1 = pd.read_csv(TASK1_SUBMISSION)
    logging.info(f"Loaded Task 1 results: {len(df_task1)} rows (Repertoires)")
    
    # 2. Load Task 2 results (Sequence Level - Top 125 per rep)
    # These files now contain BOTH Train and Test repertoires
    ranking_files = list(TASK2_DIR.glob("*_ranking.csv"))
    if not ranking_files:
        logging.warning("No Task 2 ranking files found. Submission will have missing Task 2 columns.")
    # 2. Load Task 2 Results (Dataset-Specific Ranking Files)
    # We expect files like: ds1_train_ranking.csv, ds1_test_ranking.csv
    # We must explicitly strict map them to 'dataset' name to avoid collision.
    task2_rows = []
    
    # Import mappings locally
    from data.load_all_datasets import TRAIN_DATASETS, TEST_DATASETS
    
    # helper to find mapping
    def get_dataset_name_from_file(fname):
        # fname example: ds1_train.csv or ds7_1_test_ranking.csv
        # We try to match keys
        name = fname.name
        
        # Check Train keys
        for k, v in TRAIN_DATASETS.items():
            if name.startswith(f"{k}_train"):
                return v, "train"
                
        # Check Test keys
        for k, v in TEST_DATASETS.items():
            if name.startswith(f"{k}_test"):
                return v, "test"
        
        # Fallback for old style "ds1_ranking.csv" - Try to guess
        # But this is risky. Let's rely on our new script's output pattern.
        return None, None

    ranking_files = list(TASK2_DIR.glob("*_ranking.csv"))
    logging.info(f"Found {len(ranking_files)} Task 2 ranking files.")
    
    for f in ranking_files:
        ds_name, ds_type = get_dataset_name_from_file(f)
        if not ds_name:
            logging.warning(f"  ⚠️ Skipping unrecognized file: {f.name}")
            continue
            
        df = pd.read_csv(f)
        df["dataset"] = ds_name # INJECT DATASET NAME HERE
        task2_rows.append(df)
        
    if task2_rows:
        df_task2 = pd.concat(task2_rows, ignore_index=True)
        # Rename sequence -> junction_aa immediately
        df_task2 = df_task2.rename(columns={"sequence": "junction_aa"})
        logging.info(f"Loaded Task 2 Results: {len(df_task2)} rows total.")
    else:
        # Emergency fallback or empty
        logging.error("No valid Task 2 files found! Submission will fail.")
        df_task2 = pd.DataFrame(columns=["repertoire_id", "junction_aa", "v_call", "j_call", "dataset"])
        
    # 3. Process & Merge
    # Task 1 (df_task1) is 1 row per repertoire.
    # df_task1 has ['repertoire_id', 'dataset', 'probability'] (Checked via logs)
    # Rename probability -> label_positive_probability
    if "probability" in df_task1.columns:
        df_task1 = df_task1.rename(columns={"probability": "label_positive_probability"})
        
    # We merge on BOTH [repertoire_id, dataset] to ensure safety
    # Reverting drop because it caused KeyError (ID not found implies it was already renamed)
    merged = df_task2.merge(df_task1, on=["repertoire_id", "dataset"], how="left")
    logging.info(f"Merged Data Shape: {merged.shape}")
    logging.info(f"Merged Columns: {merged.columns.tolist()}")
    
    final_rows = []
    
    # Iterate by dataset to handle specific logic
    for dataset_name, group in merged.groupby("dataset"):
        if "test" in dataset_name:
            # TEST DATASETS: STRICTLY 1 row per repertoire (Duplicate ID Forbidden)
            # Take the top ranked sequence
            group = group.sort_values("score", ascending=False)
            
            # Filter distinct repertoire_id
            distinct_reps = group.drop_duplicates(subset=["repertoire_id"])
            
            # Vectorized ID: Use repertoire_id
            distinct_reps = distinct_reps.copy()
            distinct_reps["ID"] = distinct_reps["repertoire_id"]
            
            logging.info(f"  Processed {dataset_name} (Test): kept {len(distinct_reps)} rows (1 per rep).")
            final_rows.append(distinct_reps)
                
        else:
            # TRAIN DATASETS: Synthetic IDs
            # Sort by score desc to be deterministic
            group = group.sort_values("score", ascending=False)
            
            logging.info(f"  Processed {dataset_name} (Train): kept {len(group)} rows with synthetic IDs.")
             
            # Vectorized ID Assignment
            # Create a 0-indexed range for this group
            ranks = range(len(group))
            group = group.copy()
            # USE SAMPLE FORMAT: {dataset}_seq_top_{i}
            group["ID"] = [f"{dataset_name}_seq_top_{i}" for i in ranks]
            
            # Debug check
            if "train_dataset_1" in dataset_name:
                first_id = group["ID"].iloc[0]
                logging.info(f"  [DEBUG-VEC] {dataset_name} first ID: {first_id}")

            final_rows.append(group)
            
    # Concatenate all groups
    final_df = pd.concat(final_rows, ignore_index=True)
    
    logging.info(f"[DEBUG] Final DF Head:\n{final_df.head()}")
    logging.info(f"[DEBUG] Final DF Columns: {final_df.columns.tolist()}")
    
    # 5. Sanitize Columns
    # Remove duplicate columns (e.g. ID, ID.1) by keeping first
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    # Select Final Columns
    final_cols = ["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
    
    # Ensure all columns exist
    for c in final_cols:
        if c not in final_df.columns:
            logging.warning(f"  Column {c} missing, filling with default.")
            final_df[c] = None
            
    final_df = final_df[final_cols]
    
    # MASK TEST SET DATA
    # Rule: "junction_aa must be -999.0 for non-train datasets"
    test_mask = final_df['dataset'].str.contains('test', na=False)
    mask_cols = ['junction_aa', 'v_call', 'j_call']
    logging.info(f"Masking {test_mask.sum()} test rows with -999.0 for columns {mask_cols}")
    final_df.loc[test_mask, mask_cols] = "-999.0"
    
    # Debug Train ID in Final DF
    train_sample = final_df[final_df['dataset'] == 'train_dataset_1']
    if not train_sample.empty:
        logging.info(f"[DEBUG-FINAL] train_dataset_1 Head ID:\n{train_sample['ID'].head()}")
    
    # Fill specific missing values if any
    final_df['dataset'] = final_df['dataset'].fillna("unknown")
    
    # Fill logic for sequences if Task 2 missing
    final_df['junction_aa'] = final_df['junction_aa'].fillna("")
    final_df['v_call'] = final_df['v_call'].fillna("unknown")
    final_df['j_call'] = final_df['j_call'].fillna("unknown")
    
    # SAFE FILL for probability - DO NOT DROP ROWS
    if final_df['label_positive_probability'].isnull().sum() > 0:
        logging.warning(f"  Found {final_df['label_positive_probability'].isnull().sum()} rows with NULL probability. Filling with 0.5.")
        final_df['label_positive_probability'] = final_df['label_positive_probability'].fillna(0.5)
        
    # Check for missing ID
    if final_df['ID'].isnull().sum() > 0:
        logging.error(f"  Found {final_df['ID'].isnull().sum()} rows with NULL ID! IDs cannot be missing.")
        # Try to rescue? Or failure.
        # Ensure ID at all costs
        null_id_mask = final_df['ID'].isnull()
        final_df.loc[null_id_mask, 'ID'] = "MISSING_ID_" + final_df.loc[null_id_mask].index.astype(str)
    
    # Sanity Check Row Count
    logging.info(f"Final Row Count: {len(final_df)}")
    
    # 5. Save & Versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = SUBMISSIONS_ROOT / f"submission_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = version_dir / "submission.csv"
    final_df.to_csv(out_path, index=False)
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    # 6. Generate Summary Log
    log_path = version_dir / "summary.log"
    with open(log_path, "w") as f:
        f.write(f"Submission Generated: {timestamp}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Rows: {len(final_df)}\n")
        f.write(f"Columns: {list(final_df.columns)}\n\n")
        f.write("Dataset Breakdown:\n")
        f.write(str(final_df["dataset"].value_counts()) + "\n\n")
        f.write("Missing Values:\n")
        f.write(str(final_df.isnull().sum()) + "\n\n")
        
    logging.info(f"✅ Saved final submission to {out_path}")
    logging.info(f"✅ Also saved to {OUTPUT_CSV}")

    # CALL VALIDATION
    validate_submission(final_df, log_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_size", type=str, choices=["35", "650", "35m", "650m"], default="650",
                      help="Embedding model size to use: 35 (35M) or 650 (650M). Adjusts INPUT and OUTPUT paths.")
    args = parser.parse_args()

    # Dynamic Configuration
    if "35" in args.model_size:
        # Legacy / Default
        logging.info("🔵 Selected 35M Workflow")
        # Keep globals as is (Task 1: outputs/submission/submission.csv, Task 2: outputs/task2_ranking)
        
    else:
        # 650M Namespaced
        logging.info("🟣 Selected 650M Workflow")
        
        # Adjust inputs to read from 650M namespaced folders
        TASK1_SUBMISSION = Path("outputs/submission_650m/submission.csv")
        TASK2_DIR = Path("outputs/task2_ranking_650m")
        SUBMISSIONS_ROOT = Path("outputs/submissions_650m")
        OUTPUT_CSV = Path("submission_650m.csv")
        
        logging.info(f"🟣 Inputs: Task1={TASK1_SUBMISSION}, Task2={TASK2_DIR}")
        logging.info(f"🟣 Output: {OUTPUT_CSV}")

    # Ensure root exists
    SUBMISSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    
    build_submission()
