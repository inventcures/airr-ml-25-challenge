
import pandas as pd
import numpy as np
import logging
import sys
import joblib
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TEST_DATASETS
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

SYSTEM ARCHITECTURE
-------------------
This script addresses the second competition task: Identifying specific TCR sequences associated with the disease.

METHODOLOGY:
Instead of training a separate model, we leverage the interpretability of our "Stream 2" (ESM-MIL) models.
1. Model Loading: We load the robust `ESMSequenceClassifier` trained on the corresponding dataset.
2. Embedding Scoring:
   - The classifier (SGD/Logistic) has learned a hyperplane defined by coefficients `w`.
   - For a sequence with embedding `x`, the score is `w . x`.
   - High positive scores indicate strong association with the positive class (disease).
3. Ranking:
   - We scan the target test repertoires.
   - We compute scores for every sequence.
   - We retrieve the Top-K sequences with the highest scores.

why THIS WORKS:
Since ESM embeddings capture biological properties, the linear decision boundary effectively separates "healthy-like" motifs from "disease-like" motifs. By retrieving the furthest points on the positive side of the hyperplane, we identify the most confident disease biomarkers.
"""

def rank_sequences_task2_all():
    ds_iter = tqdm(TEST_DATASETS.keys(), desc="Task 2 Rankings")
    
    for test_ds in ds_iter:
        ds_iter.set_description(f"Ranking {test_ds}")
        
        # Check if output exists
        out_csv = OUTPUT_DIR / f"{test_ds}_ranking.csv"
        if out_csv.exists():
            logging.info(f"  ✅ Ranking file exists for {test_ds}. Skipping.")
            continue
        
        logging.info(f"\n[Task 2] Processing {test_ds}...")
        
        # Determine corresponding model (from train ds)
        train_ds = test_ds.split("_")[0]
        # Handle ds7_1 -> ds7
        if train_ds not in MODELS_DIR.glob(f"{train_ds}_esm_seq_model.joblib"):
             if "_" in test_ds:
                 train_ds = test_ds.split("_")[0]
        
        model_path = MODELS_DIR / f"{train_ds}_esm_seq_model.joblib"
        if not model_path.exists():
            logging.warning(f"  Model not found for {test_ds} (expected {train_ds}). Skipping.")
            continue
            
        logging.info(f"  Loading model from {model_path}...")
        model = ESMSequenceClassifier.load(model_path)
        
        # Load repertoires
        candidates = [
            PROCESSED_DIR / f"{test_ds}_test.pkl",
            PROCESSED_DIR / f"{test_ds}_1_test.pkl",
        ]
        reps = None
        for p in candidates:
            if p.exists():
                reps = load_repertoires_pickle(p)
                break
                
        if not reps:
            logging.warning(f"  Pickle not found for {test_ds}. Skipping.")
            continue
            
        # Load embeddings
        # Load embeddings with robust checking
        base_ds = test_ds.split("_")[0]
        
        # Define candidate roots
        candidate_roots = [
            EMBEDDINGS_DIR / test_ds,
            EMBEDDINGS_DIR / test_ds / "test",
            EMBEDDINGS_DIR / base_ds,
            EMBEDDINGS_DIR / base_ds / "test",
            # Handle multipart if needed
            EMBEDDINGS_DIR / base_ds / "test" / "1_test",
            EMBEDDINGS_DIR / base_ds / "test" / "2_test",
            EMBEDDINGS_DIR / base_ds / "test" / "3_test",
        ]
        
        # Filter existing roots
        valid_roots = [p for p in candidate_roots if p.exists()]
            
        # Checkpoint Setup
        ckpt_path = OUTPUT_DIR / f"{test_ds}_checkpoint.pkl"
        processed_ids = set()
        all_rankings = []
        
        if ckpt_path.exists():
            try:
                data = joblib.load(ckpt_path)
                processed_ids = data['processed_ids']
                all_rankings = data['all_rankings']
                logging.info(f"  🔄 Resuming from checkpoint: {len(processed_ids)} repertoires already processed.")
            except Exception as e:
                logging.warning(f"  Failed to load checkpoint: {e}. Starting from scratch.")
        
        # Filter reps
        reps_to_process = [r for r in reps if r.rep_id not in processed_ids]
        if not reps_to_process and not all_rankings:
            logging.warning("  No repertoires to process (and no checkpoint data). Skipping.")
            continue
            
        if not reps_to_process:
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
                        df_rank = rank_sequences(valid_seqs, emb, model, top_k=100)
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
