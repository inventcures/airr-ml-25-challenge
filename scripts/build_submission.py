import pandas as pd
from pathlib import Path

TASK1_SUBMISSION = Path("outputs/submission/submission.csv")
TASK2_DIR = Path("outputs/task2_ranking")
OUTPUT_CSV = Path("submission.csv")

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
    # We don't have it in df_task1 directly, but we can infer or we should have saved it.
    # `train_meta_and_predict.py` didn't save dataset name.
    # But we can look it up if we load the pickle metadata? Too slow.
    # Or we can just parse it from the input files if we had them.
    # Actually, `sample_submissions.csv` has it.
    # Maybe we should load `sample_submissions.csv` and fill it?
    # But we don't have the full sample submission (it's 34MB).
    # We can try to infer from `repertoire_id` if unique?
    # Or just leave it empty if not strictly required for scoring (but usually it is).
    
    # Better approach: In `train_meta_and_predict.py`, we iterated datasets.
    # We should have included dataset name in the output.
    # But we didn't.
    
    # Let's try to map repertoire_id -> dataset using the pickles.
    # We can load all test pickles and build a map.
    # This is reasonably fast (pickles are loaded quickly).
    
    from data.load_all_datasets import TEST_DATASETS, PROCESSED_DIR, load_repertoires_pickle
    
    rep_to_dataset = {}
    print("  Mapping repertoires to datasets...")
    for ds_name, _ in TEST_DATASETS.items():
        pkl_path = PROCESSED_DIR / f"{ds_name}_test.pkl"
        if pkl_path.exists():
            reps = load_repertoires_pickle(pkl_path)
            for r in reps:
                rep_to_dataset[r.rep_id] = ds_name # or the actual directory name?
                # sample_submissions.csv had "test_dataset_1".
                # TEST_DATASETS values are directory names like "test_dataset_1".
                # So we should use the value from TEST_DATASETS.
                
    # Actually, TEST_DATASETS is Dict[str, str]. Key=ds1, Value=test_dataset_1.
    # We want the Value.
    
    rep_to_dataset_real = {}
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
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved final submission to {OUTPUT_CSV} with {len(final_df)} rows.")

if __name__ == "__main__":
    build_submission()
