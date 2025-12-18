
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

# Dynamic Global Configuration (Defaults)
MODELS_DIR = Path("models/esm_seq")
ENSEMBLE_DIR = Path("models/esm_seq_ensemble")
EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("outputs/task2_ranking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Task 2: Disease-Associated Sequence Discovery (Ranking/Retrieval)
"""

class EnsembleModel:
    def __init__(self, models):
        self.models = models
        
    def predict_proba(self, X):
        # Average probabilities
        p_sum = None
        for m in self.models:
            p = m.predict_proba(X)
            if p_sum is None: p_sum = p
            else: p_sum += p
        return p_sum / len(self.models)
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba_sequences(self, X_seq):
        """
        Average sequence-level probabilities across all models.
        Values are [0, 1] for class 1 likelihood.
        """
        p_sum = None
        for m in self.models:
            # Each model returns (N, 2)
            p = m.predict_proba_sequences(X_seq)
            if p_sum is None: 
                p_sum = p
            else: 
                p_sum += p
        return p_sum / len(self.models)

def rank_sequences_task2_all(force=False):
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
        if out_csv.exists() and not force:
            logging.info(f"  ✅ Ranking file exists for {ds_name}. Skipping.")
            continue

        logging.info(f"\n[Task 2] Processing {ds_name} ({ds_type})...")
        
        # Determine Train DS for model loading
        train_ds_model = task["base_ds"].split("_")[0] # ds7_1 -> ds7
        
        # --- MODEL LOADING LOGIC (ENSEMBLE vs SINGLE) ---
        ensemble_models = []
        ensemble_dir = ENSEMBLE_DIR
        
        # Check for 5 folds
        folds_found = 0
        for k in range(5):
             fold_path = ensemble_dir / f"{train_ds_model}_fold{k}.joblib"
             if fold_path.exists():
                 folds_found += 1
        
        if folds_found == 5:
            logging.info(f"  🚀 Found 5/5 Ensemble Folds for {train_ds_model}! Using Ensemble.")
            for k in range(5):
                fold_path = ensemble_dir / f"{train_ds_model}_fold{k}.joblib"
                ensemble_models.append(ESMSequenceClassifier.load(fold_path))
        else:
            # Fallback
            model_path = MODELS_DIR / f"{train_ds_model}_esm_seq_model.joblib"
            if not model_path.exists():
                logging.warning(f"  Model not found for {ds_name} at {model_path}. Skipping.")
                continue
            logging.info(f"  Using Single Model from {model_path}...")
            ensemble_models.append(ESMSequenceClassifier.load(model_path))
        
        # Load Repertoires 
        pickle_candidates = [
            PROCESSED_DIR / f"{ds_name}{task['pickle_suffix']}.pkl",
            PROCESSED_DIR / f"{ds_name}.pkl" 
        ]
        
        reps = None
        for p in pickle_candidates:
            if p.exists():
                reps = load_repertoires_pickle(p)
                break
                
        if not reps:
            logging.warning(f"  Pickle not found for {ds_name}. Skipping.")
            continue
            
        logging.info(f"  Loaded {len(reps)} repertoires. Ranking with {len(ensemble_models)} models...")
        
        # Wrap models
        ensemble_wrapper = EnsembleModel(ensemble_models)
        
        # Run Ranking (Logic moved to per-repertoire loop below)
        # df_res = rank_sequences(reps, ensemble_wrapper, ds_name, task["base_ds"], task["target_rows"])

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
        
        processed_ids = set()
        all_rankings = []
        
        # Resumption Logic
        if ckpt_path.exists():
            if force:
                logging.warning(f"  ⚠️ Force enabled: Deleting old checkpoint {ckpt_path}.")
                ckpt_path.unlink()
            else:
                try:
                    logging.info(f"  🔄 Found checkpoint {ckpt_path}. Loading state...")
                    state = joblib.load(ckpt_path)
                    processed_ids = state.get('processed_ids', set())
                    all_rankings = state.get('all_rankings', [])
                    logging.info(f"  ✅ Resumed from {len(processed_ids)} repertoires.")
                except Exception as e:
                    logging.error(f"  ❌ Corrupt checkpoint {ckpt_path}: {e}. Starting fresh.")
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
            if r.rep_id in processed_ids:
                continue

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
                # Sanitize: Check for NaNs or Inf
                if not np.isfinite(emb).all():
                    logging.warning(f"  ⚠️ Embeddings for {r.rep_id} contain NaNs/Inf. Replacing with zeros.")
                    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

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
                        df_rank = rank_sequences(valid_seqs, emb, ensemble_wrapper, top_k=per_rep_top_k)
                        df_rank["repertoire_id"] = r.rep_id
                        
                        # Merge V/J info
                        rep_df = pd.DataFrame({
                            "sequence": r.junction_aa,
                            "v_call": r.v_call,
                            "j_call": r.j_call
                        })
                        rep_df = rep_df[rep_df["sequence"].str.len() > 0]
                        
                        # Safety: Ensure we don't crash on missing sequences in lookup
                        # map rank result back to full info
            
            # --- FALLBACK LOGIC ---
            if emb is None:
                # If embedding missing/empty, just pick first valid sequence(s) as default
                valid_seqs = [s for s in r.junction_aa if len(s) > 0]
                if valid_seqs:
                    logging.warning(f"  ⚠️ No embedding for {r.rep_id}. Using fallback (taking first {per_rep_top_k} seqs).")
                    selected = valid_seqs[:per_rep_top_k]
                    
                    df_rank = pd.DataFrame({
                        "sequence": selected,
                        "score": 0.0, # Low score to indicate fallback
                        "rank": range(1, len(selected) + 1),
                        "repertoire_id": r.rep_id,
                        "model_name": "fallback"
                    })
                    
                    rep_df = pd.DataFrame({
                        "sequence": r.junction_aa,
                        "v_call": r.v_call,
                        "j_call": r.j_call
                    })
                    rep_df = rep_df[rep_df["sequence"].str.len() > 0]
                else:
                     logging.error(f"  ❌ No valid sequences {r.rep_id} even for fallback.")
                     continue

            # --- END FALLBACK ---

            
            # --- SHARED MERGE LOGIC ---
            if 'df_rank' in locals() and 'rep_df' in locals():
                # Dedup rep_df on sequence to avoid explosion if local duplicate
                rep_df = rep_df.drop_duplicates(subset=["sequence"])
                
                # Merge 
                merged = df_rank.merge(rep_df, on="sequence", how="left")
                
                all_rankings.append(merged)
                
                # Cleanup for next loop
                del df_rank
                del rep_df
            
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("-m", "--model_size", type=str, choices=["35", "650", "35m", "650m"], default="650",
                      help="Embedding model size to use: 35 (35M) or 650 (650M). Adjusts EMBEDDINGS_DIR and OUTPUT DIRS.")
    args = parser.parse_args()

    logging.info(f"Arguments: {args}")

    # Dynamic Configuration
    if "35" in args.model_size:
        # Check if we are on the 'New Pod' structure where 35m was moved to subdir
        if Path("data/embeddings/35m").exists():
            EMBEDDINGS_DIR = Path("data/embeddings/35m")
            logging.info(f"🔵 Selected 35M Embeddings from {EMBEDDINGS_DIR}")
        else:
            EMBEDDINGS_DIR = Path("data/embeddings")
            logging.info(f"🔵 Selected 35M Embeddings (Fallback/Legacy) from {EMBEDDINGS_DIR}")
            
        # Keep Default Outputs for 35M
        MODELS_DIR = Path("models/esm_seq")
        ENSEMBLE_DIR = Path("models/esm_seq_ensemble")
        OUTPUT_DIR = Path("outputs/task2_ranking")
        
    else:
        # 650M Namespaced Outputs
        EMBEDDINGS_DIR = Path("data/embeddings") 
        logging.info(f"🟣 Selected 650M Embeddings from {EMBEDDINGS_DIR}")
        
        MODELS_DIR = Path("models/esm_seq_650m")           # Though we might use ensemble, best to map both for fallback
        ENSEMBLE_DIR = Path("models/esm_seq_ensemble_650m") # Where we expect trained models from Stream 2
        OUTPUT_DIR = Path("outputs/task2_ranking_650m")
        logging.info(f"🟣 Outputs redirected to: {OUTPUT_DIR}")
        logging.info(f"🟣 Loading models from: {ENSEMBLE_DIR} (or {MODELS_DIR})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    rank_sequences_task2_all(force=args.force)
