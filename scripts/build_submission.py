import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

TASK1_SUBMISSION = Path("outputs/submission/submission.csv")
TASK2_DIR = Path("outputs/task2_ranking")
OUTPUT_CSV = Path("submission.csv")
SUBMISSIONS_ROOT = Path("outputs/submissions")

def build_submission():
    print("[Build Submission] Generating final submission file...")
    
    # 1. Load Task 1 results
    if not TASK1_SUBMISSION.exists():
        print(f"Task 1 submission not found: {TASK1_SUBMISSION}")
        return
        
    df_task1 = pd.read_csv(TASK1_SUBMISSION)
    # Columns: repertoire_id, probability
    
    # 2. Load Task 2 results
    # We need to iterate all ranking files and combine them
    ranking_files = list(TASK2_DIR.glob("*_ranking.csv"))
    if not ranking_files:
        print("No Task 2 ranking files found.")
        # We can still generate Task 1 only? No, we need all columns.
        # Fill with defaults?
        pass
        
    task2_rows = []
    for f in ranking_files:
        df = pd.read_csv(f)
        # Columns: repertoire_id, sequence, score, v_call, j_call
        # We take top 1 per repertoire
        # Sort by score just in case
        df = df.sort_values(["repertoire_id", "score"], ascending=[True, False])
        top1 = df.drop_duplicates(subset=["repertoire_id"], keep="first")
        task2_rows.append(top1)
        
    if task2_rows:
        df_task2 = pd.concat(task2_rows)
    else:
        df_task2 = pd.DataFrame(columns=["repertoire_id", "sequence", "v_call", "j_call"])
        
    # 3. Merge
    # Left join on Task 1 (which should have all test repertoires)
    merged = df_task1.merge(df_task2, on="repertoire_id", how="left")
    
    # 4. Format
    # ID,dataset,label_positive_probability,junction_aa,v_call,j_call
    
    # We need 'dataset' column.
    # Let's try to map repertoire_id -> dataset using the pickles.
    from data.load_all_datasets import TEST_DATASETS, PROCESSED_DIR, load_repertoires_pickle
    
    rep_to_dataset_real = {}
    print("  Mapping repertoires to datasets...")
    for ds_name, dir_name in TEST_DATASETS.items():
        pkl_path = PROCESSED_DIR / f"{ds_name}_test.pkl"
        if pkl_path.exists():
            reps = load_repertoires_pickle(pkl_path)
            for r in reps:
                rep_to_dataset_real[r.rep_id] = dir_name

    merged["dataset"] = merged["repertoire_id"].map(rep_to_dataset_real)
    
    # Rename columns
    merged = merged.rename(columns={
        "repertoire_id": "ID",
        "probability": "label_positive_probability",
        "sequence": "junction_aa"
    })
    
    # Fill missing Task 2 with defaults
    merged["junction_aa"] = merged["junction_aa"].fillna("-999.0") # as per sample?
    merged["v_call"] = merged["v_call"].fillna("-999.0")
    merged["j_call"] = merged["j_call"].fillna("-999.0")
    
    # Select columns
    final_cols = ["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
    final_df = merged[final_cols]
    
    # 5. Save & Versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = SUBMISSIONS_ROOT / f"submission_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = version_dir / "submission.csv"
    final_df.to_csv(out_path, index=False)
    
    # Also save to root for convenience
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
        
        f.write("Task 2 Coverage:\n")
        f.write(f"Repertoires with Task 2 sequences: {len(df_task2)}\n")
        f.write(f"Repertoires missing Task 2 sequences: {len(final_df) - len(df_task2)}\n")
        
    print(f"Saved final submission to {out_path}")
    print(f"Saved summary log to {log_path}")
    print(f"Also saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    build_submission()
