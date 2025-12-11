
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from tqdm import tqdm

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
        # Robust embedding finding logic from other scripts
        emb_dir = EMBEDDINGS_DIR / test_ds
        if not emb_dir.exists():
             base_ds = test_ds.split("_")[0]
             if (EMBEDDINGS_DIR / base_ds).exists():
                 emb_dir = EMBEDDINGS_DIR / base_ds
                 
        if not emb_dir.exists():
            logging.warning(f"  Embeddings dir not found: {emb_dir}")
            continue
            
        all_rankings = []
        
        pbar_reps = tqdm(reps, desc=f"  Ranking Reps", leave=False)
        for r in pbar_reps:
            emb_path = emb_dir / f"{r.rep_id}.npy"
            # Try subfolder if needed? Assuming flat now.
            if not emb_path.exists():
                # Attempt recursive search if flat search fails?
                # For now assume flat or correct structure.
                continue
                
            try:
                emb = np.load(emb_path)
                if emb.ndim == 1:
                    if len(emb) == 0:
                        continue
                    emb = emb.reshape(1, -1)
            except:
                continue
                
            # Filter valid sequences
            valid_seqs = [s for s in r.junction_aa if len(s) > 0]
            if len(valid_seqs) == 0:
                continue
                
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
            
        if all_rankings:
            final_df = pd.concat(all_rankings)
            final_df = final_df[["repertoire_id", "sequence", "score", "v_call", "j_call"]]
            final_df.to_csv(out_csv, index=False)
            logging.info(f"  Saved rankings to {out_csv}")
        else:
            logging.warning("  No rankings generated.")

if __name__ == "__main__":
    rank_sequences_task2_all()
