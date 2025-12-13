
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
    # Process both Train and Test datasets because submission requires rows for ALL repertoires (125 per rep)
    # 50,000 rows / 400 reps = 125 rows/rep
    all_datasets = list(TEST_DATASETS.keys()) + list(TRAIN_DATASETS.keys())
    ds_iter = tqdm(all_datasets, desc="Task 2 Rankings")
    
    for test_ds in ds_iter:
        ds_iter.set_description(f"Ranking {test_ds}")
        
        # Check if output exists. 
        # CAUTION: If we changed top_k, we usually should regenerate. 
        # We will NOT skip if file exists, to ensure we get the new 125-rank files.
        out_csv = OUTPUT_DIR / f"{test_ds}_ranking.csv"
        # if out_csv.exists():
        #     logging.info(f"  ✅ Ranking file exists for {test_ds}. Skipping.")
        #     continue
        
        logging.info(f"\n[Task 2] Processing {test_ds}...")
        
        # Determine corresponding model (from train ds)
        # If test_ds is actually a training set (e.g. "ds1"), then train_ds is "ds1".
        train_ds = test_ds.split("_")[0]
        # Handle ds7_1 -> ds7
        # Check if train_ds model exists.
        
        # Mapping logic:
        # ds1 -> ds1 (train) -> model: ds1_esm_seq_model.joblib
        # ds7_1 -> ds7 (train) -> model: ds7_esm_seq_model.joblib
        
        model_path = MODELS_DIR / f"{train_ds}_esm_seq_model.joblib"
        if not model_path.exists():
            logging.warning(f"  Model not found for {test_ds} (expected {train_ds} at {model_path}). Skipping.")
            continue
            
        logging.info(f"  Loading model from {model_path}...")
        model = ESMSequenceClassifier.load(model_path)
        
        # Load repertoires (Check both Train and Test pickle patterns)
        candidates = [
            PROCESSED_DIR / f"{test_ds}_test.pkl",
            PROCESSED_DIR / f"{test_ds}_1_test.pkl",
            PROCESSED_DIR / f"{test_ds}_train.pkl", # Support Train DS
            PROCESSED_DIR / f"{test_ds}.pkl",       # Fallback
        ]
        reps = None
        for p in candidates:
            if p.exists():
                reps = load_repertoires_pickle(p)
                break
                
        if not reps:
            logging.warning(f"  Pickle not found for {test_ds}. Skipping.")
            continue
            
        # Load embeddings with robust checking
        base_ds = test_ds.split("_")[0]
        
        # Define candidate roots
        candidate_roots = [
            EMBEDDINGS_DIR / test_ds,
            EMBEDDINGS_DIR / test_ds / "test",
            EMBEDDINGS_DIR / test_ds / "train", # Support Train
            EMBEDDINGS_DIR / base_ds,
            EMBEDDINGS_DIR / base_ds / "test",
            EMBEDDINGS_DIR / base_ds / "train", # Support Train
            # Handle multipart if needed (ds7/test/1_test)
            EMBEDDINGS_DIR / base_ds / "test" / "1_test",
            EMBEDDINGS_DIR / base_ds / "test" / "2_test",
            EMBEDDINGS_DIR / base_ds / "test" / "3_test",
        ]
        
        # Filter existing roots
        valid_roots = [p for p in candidate_roots if p.exists()]
            
        # Checkpoint Setup
        ckpt_path = OUTPUT_DIR / f"{test_ds}_checkpoint.pkl"
        
        # CLEANUP: Always remove stale checkpoint if this is a fresh start for this dataset
        # However, we might want to resume if the script crashed mid-way effectively.
        # But user asked for "clean overwrite".
        # If we always delete checkpoint, we lose resume capability.
        # A better approach: If the final output file exists, we delete it AND the checkpoint to start fresh.
        # If final output doesn't exist but checkpoint does, we resume.
        # Wait, I removed the "if out_csv exists: continue" block earlier (it is commented out).
        # So we represent "Force Rerun".
        
        # Decision: To satisfy "cleanly stores or overwrites", we should probably reset unless a flag says otherwise.
        # Or simpler: If we are starting iteration 0, we can ignore checkpoint.
        # But we load checkpoint BEFORE loop.
        
        # Let's trust the standard logic:
        # If we are re-running, likely the user wants to re-compute.
        # But if it crashed after 4 hours, they want resume.
        
        # Compromise: Check if the *Task 2* logic changed.
        # Since I am editing the script NOW, the previous checkpoint is likely from the OLD logic (Top 100).
        # Resuming from a Top-100 checkpoint while trying to build Top-125 is DANGEROUS/WRONG.
        # FIX: We MUST invalidated old checkpoints.
        
        if ckpt_path.exists():
            # Verify if checkpoint is compatible? Hard.
            # Safest: Nuke it.
            logging.warning(f"  ⚠️ Deleting old checkpoint {ckpt_path} to ensure clean Top-top_k run.")
        # Determine limits based on dataset type
        # Test Datasets: 1 row per repertoire (according to sample_submission)
        # Train Datasets: Exactly 50,000 rows total (Global top 50k or distributed)
        
        is_train = "train" in test_ds or test_ds in TRAIN_DATASETS
        
        target_total_rows = None
        per_rep_top_k = 1 # Default for Test
        
        # Calculate dynamic top_k
        if reps:
            num_reps = len(reps)
            if is_train:
                target_total_rows = 50000
                import math
                per_rep_top_k = math.ceil(50000 / num_reps)
                logging.info(f"  [Train Limit] Goal: 50,000 rows. Repertoires: {num_reps}. Using top_k={per_rep_top_k} per rep.")
            else:
                logging.info(f"  [Test Limit] Goal: 1 row per rep. Using top_k=1.")
                per_rep_top_k = 1
        
        # Checkpoint Setup
        ckpt_path = OUTPUT_DIR / f"{test_ds}_checkpoint.pkl"
        
        # CLEANUP: Delete stale checkpoint to force fresh run with new logic
        if ckpt_path.exists():
            logging.warning(f"  ⚠️ Deleting old checkpoint {ckpt_path} to ensure clean Top-top_k run.")
            ckpt_path.unlink()
            
        processed_ids = set()
        all_rankings = []
        
        # Filter reps
        reps_to_process = [r for r in reps if r.rep_id not in processed_ids]
        if not reps_to_process:
            if not all_rankings:
                logging.warning("  No repertoires to process (and no checkpoint data). Skipping.")
                continue
            else:
                 logging.info("  All repertoires processed in checkpoint. assembling final CSV.")
        
        pbar_reps = tqdm(reps_to_process, desc=f"  Ranking {test_ds}", leave=False)
        
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
                        emb = None # Skip empty
                
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
                        
                        # Merge (left join on rank to keep only top k)
                        merged = df_rank.merge(rep_df, on="sequence", how="left")
                        merged = merged.drop_duplicates(subset=["sequence"])
                        
                        all_rankings.append(merged)
            
            # Update State
            processed_ids.add(r.rep_id)
            
            # Checkpoint every 50 reps
            if (i + 1) % 50 == 0:
                joblib.dump({'processed_ids': processed_ids, 'all_rankings': all_rankings}, ckpt_path)
                pbar_reps.set_postfix({'saved': len(processed_ids)})
            
        if all_rankings:
            final_df = pd.concat(all_rankings)
            
            # Post-Process for Train: Trim to exactly 50,000 if we overshot (due to ceil)
            if is_train and target_total_rows and len(final_df) > target_total_rows:
                original_len = len(final_df)
                # Sort by score global descending
                final_df = final_df.sort_values("score", ascending=False)
                final_df = final_df.head(target_total_rows)
                logging.info(f"  trimmed {original_len} -> {len(final_df)} rows (Exact 50k target).")
            
            final_df = final_df[["repertoire_id", "sequence", "score", "v_call", "j_call"]]
            final_df.to_csv(out_csv, index=False)
            logging.info(f"  Saved rankings to {out_csv}")
            
            # Clean up checkpoint
            if ckpt_path.exists():
                ckpt_path.unlink()
        else:
            logging.warning("  No rankings generated.")

if __name__ == "__main__":
    rank_sequences_task2_all()
