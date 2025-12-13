
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
        logging.info(f"Loaded Task 2 Results: {len(df_task2)} rows total.")
    else:
        # Emergency fallback or empty
        logging.error("No valid Task 2 files found! Submission will fail.")
        df_task2 = pd.DataFrame(columns=["repertoire_id", "sequence", "v_call", "j_call", "dataset"])
        
    # 3. Process & Merge
    # Task 1 (df_task1) is 1 row per repertoire.
    # df_task1 has ['repertoire_id', 'dataset', 'label_positive_probability']
    # df_task2 has ['repertoire_id', 'sequence', 'score', 'dataset', ...]
    
    # We merge on BOTH [repertoire_id, dataset] to ensure safety
    # But Task 1 might have 'dataset' (e.g. test_dataset_1)
    
    merged = df_task2.merge(df_task1, on=["repertoire_id", "dataset"], how="left")
    
    # 4. Apply ID Logic
    final_rows = []
    
    # Iterate by dataset to handle specific logic
    for dataset_name, group in merged.groupby("dataset"):
        if "test" in dataset_name:
            # TEST DATASETS: STRICTLY 1 row per repertoire (Duplicate ID Forbidden)
            # Take the top ranked sequence (first one, since sorting is preserved or we force sort)
            # Assumption: Inputs are sorted by score desc? Rank script did sort.
            # But let's be safe. If 'score' is present, sort.
            if "score" in group.columns:
                group = group.sort_values("score", ascending=False)
            
            # Drop duplicates on repertoire_id, keeping first (best)
            group = group.drop_duplicates(subset=["repertoire_id"], keep="first")
            
            # ID is just Repertoire ID
            group = group.rename(columns={"repertoire_id": "ID"})
            final_rows.append(group)
            logging.info(f"  Processed {dataset_name} (Test): kept {len(group)} rows (1 per rep).")
            
        elif "train" in dataset_name:
            # TRAIN DATASETS: STRICTLY 50,000 rows (Task 2 Target)
            # ID must be UNIQUE. We use synthetic ID: dataset_name + "_seq_top_" + rank
            if len(group) != 50000:
                logging.warning(f"  ⚠️ {dataset_name} has {len(group)} rows. Expected 50,000.")
            
            # Sort by score for correct ranking
            if "score" in group.columns:
                 group = group.sort_values("score", ascending=False)
            
            # Generate Synthetic IDs
            # 0-indexed rank matches 0..49999
            ranks = range(len(group)) 
            group["ID"] = [f"{dataset_name}_seq_top_{i}" for i in ranks]
            
            final_rows.append(group)
            logging.info(f"  Processed {dataset_name} (Train): kept {len(group)} rows with synthetic IDs.")
            
    merged = pd.concat(final_rows)
    
    # 4. Format
    if "dataset" not in merged.columns:
        logging.warning("  'dataset' column missing from Task 1 predictions! Submission may be invalid.")
    
    # Rename columns
    merged = merged.rename(columns={
        "repertoire_id": "ID",
        "probability": "label_positive_probability",
        "sequence": "junction_aa"
    })
    
    # Fill missing Task 2 with defaults
    merged["junction_aa"] = merged["junction_aa"].fillna("-999.0")
    merged["v_call"] = merged["v_call"].fillna("-999.0")
    merged["j_call"] = merged["j_call"].fillna("-999.0")
    
    # Select columns
    final_cols = ["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
    
    # Ensure all cols exist (dataset might be null if mapping failed)
    if "dataset" not in merged.columns or merged["dataset"].isnull().any():
        logging.warning(f"  {merged['dataset'].isnull().sum()} rows have missing dataset mapping.")
        merged["dataset"] = merged["dataset"].fillna("unknown")
        
    final_df = merged[final_cols]
    
    # NO FILTERING. We submit everything (Train + Test).
    
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

    logging.info(f"✅ Saved final submission to {out_path}")
    logging.info(f"✅ Also saved to {OUTPUT_CSV}")
    
    validate_submission(final_df, log_path)

if __name__ == "__main__":
    build_submission()
