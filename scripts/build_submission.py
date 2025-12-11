
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import logging
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/build_submission.log")
    ]
)
Path("logs").mkdir(exist_ok=True)

TASK1_SUBMISSION = Path("outputs/submission/submission.csv")
TASK2_DIR = Path("outputs/task2_ranking")
OUTPUT_CSV = Path("submission.csv")
SUBMISSIONS_ROOT = Path("outputs/submissions")

def build_submission():
    logging.info("[Build Submission] Generating final submission file...")
    
    # 1. Load Task 1 results
    if not TASK1_SUBMISSION.exists():
        logging.error(f"Task 1 submission not found: {TASK1_SUBMISSION}")
        return
        
    df_task1 = pd.read_csv(TASK1_SUBMISSION)
    logging.info(f"Loaded Task 1 results: {len(df_task1)} rows")
    
    # 2. Load Task 2 results
    ranking_files = list(TASK2_DIR.glob("*_ranking.csv"))
    if not ranking_files:
        logging.warning("No Task 2 ranking files found. Submission will have missing Task 2 columns.")
    
    task2_rows = []
    for f in ranking_files:
        try:
            df = pd.read_csv(f)
            # Take top 1 per repertoire
            df = df.sort_values(["repertoire_id", "score"], ascending=[True, False])
            top1 = df.drop_duplicates(subset=["repertoire_id"], keep="first")
            task2_rows.append(top1)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
        
    if task2_rows:
        df_task2 = pd.concat(task2_rows)
        logging.info(f"Loaded Task 2 results: {len(df_task2)} rows from {len(ranking_files)} files")
    else:
        df_task2 = pd.DataFrame(columns=["repertoire_id", "sequence", "v_call", "j_call"])
        
    # 3. Merge
    merged = df_task1.merge(df_task2, on="repertoire_id", how="left")
    
    # 4. Format
    from data.load_all_datasets import TEST_DATASETS, PROCESSED_DIR, load_repertoires_pickle
    
    rep_to_dataset_real = {}
    logging.info("  Mapping repertoires to datasets...")
    for ds_name, dir_name in TEST_DATASETS.items():
        # Try both pickle naming conventions
        candidates = [
            PROCESSED_DIR / f"{ds_name}_test.pkl",
            PROCESSED_DIR / f"{ds_name}_1_test.pkl"
        ]
        for p in candidates:
            if p.exists():
                reps = load_repertoires_pickle(p)
                for r in reps:
                    rep_to_dataset_real[r.rep_id] = dir_name
                break # Found for this dataset

    merged["dataset"] = merged["repertoire_id"].map(rep_to_dataset_real)
    
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

if __name__ == "__main__":
    build_submission()
