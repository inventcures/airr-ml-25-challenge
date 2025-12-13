
import pandas as pd
import numpy as np
import logging
import sys
import joblib
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TEST_DATASETS, TRAIN_DATASETS
from malid.esm_seq_model import ESMSequenceClassifier
from scripts.rank_sequences_task2_core import rank_sequences

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/task2_ranking.log")
    ]
)
Path("logs").mkdir(exist_ok=True)

MODELS_DIR = Path("models/esm_seq")
EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("outputs/task2_ranking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Task 2: Disease-Associated Sequence Discovery (Ranking/Retrieval)
"""

def rank_sequences_task2_all():
    # Process both Train and Test datasets separately to avoid overwriting files
    # We will generate specific output files: {ds}_train_ranking.csv and {ds}_test_ranking.csv
    
    tasks = []
    
    # 1. Schedule Test Datasets
    for test_ds in TEST_DATASETS.keys():
        tasks.append({
            "ds_name": test_ds,
            "type": "test",
            "pickle_suffix": "_test",
            "out_suffix": "_test_ranking.csv",
            "target_rows": None, # Will become 1 row/rep
            "base_ds": test_ds.split("_")[0] 
        })
        
    # 2. Schedule Train Datasets
    for train_ds in TRAIN_DATASETS.keys():
        tasks.append({
            "ds_name": train_ds,
            "type": "train",
            "pickle_suffix": "_train",
            "out_suffix": "_train_ranking.csv",
            "target_rows": 50000, 
            "base_ds": train_ds # Train DS is the base
        })
        
    ds_iter = tqdm(tasks, desc="Task 2 Rankings")
    
    for task in ds_iter:
        ds_name = task["ds_name"]
        ds_type = task["type"]
        ds_iter.set_description(f"Ranking {ds_name} ({ds_type})")
        
        out_csv = OUTPUT_DIR / f"{ds_name}{task['out_suffix']}"
        # if out_csv.exists():
        #     logging.info(f"  ✅ Ranking file exists for {ds_name}. Skipping.")
        #     continue

        logging.info(f"\n[Task 2] Processing {ds_name} ({ds_type})...")
        
        # Determine Train DS for model loading
        train_ds_model = task["base_ds"].split("_")[0] # ds7_1 -> ds7
        
        model_path = MODELS_DIR / f"{train_ds_model}_esm_seq_model.joblib"
        if not model_path.exists():
            logging.warning(f"  Model not found for {ds_name} at {model_path}. Skipping.")
            continue
            
        logging.info(f"  Loading model from {model_path}...")
        model = ESMSequenceClassifier.load(model_path)
        
        # Load Repertoires - STRICT loading based on type
        # For Test ds1, load ds1_test.pkl. For Train ds1, load ds1_train.pkl.
        pickle_candidates = [
            PROCESSED_DIR / f"{ds_name}{task['pickle_suffix']}.pkl",
            PROCESSED_DIR / f"{ds_name}.pkl" # Fallback
        ]
        
        reps = None
        for p in pickle_candidates:
            if p.exists():
                reps = load_repertoires_pickle(p)
                break
        
        if not reps:
            logging.warning(f"  Pickle not found for {ds_name} ({ds_type}). Candidates: {pickle_candidates}. Skipping.")
            continue
            
        logging.info(f"  Loaded {len(reps)} repertoires from {p.name}")

        # Load embeddings with robust checking
        base_ds = task["base_ds"]
        
        # Define candidate roots
        candidate_roots = [
            EMBEDDINGS_DIR / ds_name,
            EMBEDDINGS_DIR / ds_name / ds_type,
            EMBEDDINGS_DIR / base_ds,
            EMBEDDINGS_DIR / base_ds / ds_type,
            # Handle multipart if needed (ds7/test/1_test)
            EMBEDDINGS_DIR / base_ds / "test" / "1_test",
            EMBEDDINGS_DIR / base_ds / "test" / "2_test",
            EMBEDDINGS_DIR / base_ds / "test" / "3_test",
        ]
        
        # Filter existing roots
        valid_roots = [p for p in candidate_roots if p.exists()]
            
        # Checkpoint Setup - Separate checkpoint for each task type
        ckpt_path = OUTPUT_DIR / f"{ds_name}_{ds_type}_checkpoint.pkl"
        
        if ckpt_path.exists():
            logging.warning(f"  ⚠️ Deleting old checkpoint {ckpt_path} to ensure clean run.")
            ckpt_path.unlink()
            
        per_rep_top_k = 1
        target_total_rows = task["target_rows"]
        
        if reps:
             num_reps = len(reps)
             if ds_type == "train":
                 # Distribute 50k rows
                 import math
                 per_rep_top_k = math.ceil(50000 / num_reps)
                 logging.info(f"  [Limit] Train: 50k Goal. Using top_k={per_rep_top_k} per rep.")
             else:
                 # Test: 1 per rep
                 per_rep_top_k = 1
                 logging.info(f"  [Limit] Test: 1 per rep.")

        processed_ids = set()
        all_rankings = []
        
        pbar_reps = tqdm(reps, desc=f"  Ranking {ds_name}", leave=False)
        
        for i, r in enumerate(pbar_reps):
            # Find embedding
            emb = None
            for root in valid_roots:
                 p = root / f"{r.rep_id}.npy"
                 if p.exists():
                     try:
                         emb = np.load(p, mmap_mode='r')
                         break
                     except:
                         continue
            
            if emb is not None:
                if emb.ndim == 1:
                    if len(emb) > 0:
                        emb = emb.reshape(1, -1)
                    else:
                        emb = None 
                
                if emb is not None:
                    # Filter valid sequences
                    valid_seqs = [s for s in r.junction_aa if len(s) > 0]
                    
                    if len(valid_seqs) > 0:
                        # Truncate mismatch
                        if len(valid_seqs) != len(emb):
                            min_len = min(len(valid_seqs), len(emb))
                            valid_seqs = valid_seqs[:min_len]
                            emb = emb[:min_len]
                        
                        # Rank
                        df_rank = rank_sequences(valid_seqs, emb, model, top_k=per_rep_top_k)
                        df_rank["repertoire_id"] = r.rep_id
                        
                        # Merge V/J info
                        rep_df = pd.DataFrame({
                            "sequence": r.junction_aa,
                            "v_call": r.v_call,
                            "j_call": r.j_call
                        })
                        rep_df = rep_df[rep_df["sequence"].str.len() > 0]
                        # Dedup rep_df on sequence to avoid explosion if local duplicate
                        rep_df = rep_df.drop_duplicates(subset=["sequence"])
                        
                        # Merge 
                        merged = df_rank.merge(rep_df, on="sequence", how="left")
                        
                        all_rankings.append(merged)
            
            # Update State
            processed_ids.add(r.rep_id)
            
            # Checkpoint every 50 reps
            if (i + 1) % 50 == 0:
                joblib.dump({'processed_ids': processed_ids, 'all_rankings': all_rankings}, ckpt_path)
                pbar_reps.set_postfix({'saved': len(processed_ids)})
            
        if all_rankings:
            final_df = pd.concat(all_rankings)
            
            # Post-Process for Train: Trim to exactly 50,000 if we overshot
            if ds_type == "train" and len(final_df) > 50000:
                original_len = len(final_df)
                final_df = final_df.sort_values("score", ascending=False)
                final_df = final_df.head(50000)
                logging.info(f"  trimmed {original_len} -> {len(final_df)} rows (Exact 50k target).")
            
            final_df = final_df[["repertoire_id", "sequence", "score", "v_call", "j_call"]]
            final_df.to_csv(out_csv, index=False)
            logging.info(f"  Saved rankings to {out_csv}")
            
            if ckpt_path.exists():
                ckpt_path.unlink()
        else:
            logging.warning("  No rankings generated.")

if __name__ == "__main__":
    rank_sequences_task2_all()
