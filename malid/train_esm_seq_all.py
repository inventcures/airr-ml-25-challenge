import pandas as pd
import numpy as np
import sys
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional
import joblib
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_all_datasets import load_repertoires_pickle, TRAIN_DATASETS, PROCESSED_DIR, TEST_DATASETS
from malid.esm_seq_model import ESMSequenceClassifier

MODELS_DIR = Path("models/esm_seq")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR = Path("outputs/esm_seq_preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path("data/embeddings")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/esm_seq_train.log")
    ]
)
Path("logs").mkdir(exist_ok=True)

BATCH_SIZE = 50 

def load_embeddings(dataset_name: str, rep_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Load embeddings for a list of repertoires.
    Returns dict: rep_id -> embedding array (N, D)
    """
    emb_dir = EMBEDDINGS_DIR / dataset_name
    embeddings = {}
    
    # Map ds7_1 -> ds7 if needed
    if not emb_dir.exists():
        base_ds = dataset_name.split("_")[0]
        if (EMBEDDINGS_DIR / base_ds).exists():
            emb_dir = EMBEDDINGS_DIR / base_ds
    
    if not emb_dir.exists():
        logging.warning(f"  Embeddings dir not found: {emb_dir}")
        return {}
        
    # We do NOT want to scan all files. That takes time and maybe OS resources.
    # Just assume file names match rep_ids.
    
    for rid in rep_ids:
        path = emb_dir / f"{rid}.npy"
        if path.exists():
            try:
                emb = np.load(path)
                # Ensure 2D
                if emb.ndim == 1:
                    if len(emb) == 0:
                        emb = np.zeros((0, 1280)) # ESM2-650M dim is 1280
                    else:
                        emb = emb.reshape(1, -1)
                embeddings[rid] = emb
            except Exception as e:
                logging.error(f"  Failed to load {path}: {e}")
        else:
            # Try recursive search only if direct fail? No, assumes flat for speed.
            pass
            
    return embeddings

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def train_esm_seq_all():
    # Progress bar for datasets
    ds_iterator = tqdm(TRAIN_DATASETS.keys(), desc="Datasets")
    
    for ds_name in ds_iterator:
        ds_iterator.set_description(f"Processing {ds_name}")
        logging.info(f"Processing {ds_name}...")
        
        clf_main = None # Ensure clear scope
        
        # Check if TRAINING OOF preds ALREADY EXIST
        train_preds_csv = PREDS_DIR / f"{ds_name}_train_esm_preds.csv"
        model_path = MODELS_DIR / f"{ds_name}_esm_seq_model.joblib"
        
        # We need both the model and the OOF preds.
        if train_preds_csv.exists() and model_path.exists():
            logging.info(f"  ✅ Training artifacts (Model + OOF Preds) exist for {ds_name}. Skipping training.")
        else:
            # --- TRAINING PHASE (STREAMING) ---
            pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
            if not pkl_path.exists():
                logging.error(f"  Skipping {ds_name}, pickle not found at {pkl_path}")
                continue
                
            reps = load_repertoires_pickle(pkl_path)
            
            # Filter labeled
            labeled_reps = [r for r in reps if r.label is not None]
            if not labeled_reps:
                logging.warning("  No labeled data.")
                continue
            
            # Assign Folds upfront
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_all = [r.label for r in labeled_reps]
            
            # fold_map: rep_id -> fold_index
            fold_map = {}
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labeled_reps, y_all)):
                 for i in val_idx:
                     fold_map[labeled_reps[i].rep_id] = fold_idx
            
            # Initialize Models
            logging.info("  Initializing streaming models...")
            clf_main = ESMSequenceClassifier(random_state=42)
            clf_folds = [ESMSequenceClassifier(random_state=42) for _ in range(5)]
            classes = np.array([0, 1])
            
            # Loop Batches for TRAINING
            logging.info(f"  Training (Streaming Batches of {BATCH_SIZE})...")
            
            # Shuffle reps for better SGD convergence? Yes.
            # But we must keep track of IDs.
            # actually SKF split shuffles indices.
            # So dealing with them in random order is fine.
            # Let's just shuffle the list once.
            import random
            random.seed(42)
            train_order_reps = list(labeled_reps)
            random.shuffle(train_order_reps)
            
            batch_iter = tqdm(chunk_list(train_order_reps, BATCH_SIZE), total=(len(train_order_reps)//BATCH_SIZE)+1, desc="  Training Batches", leave=False)
            
            count_processed = 0
            
            for batch_reps in batch_iter:
                if not batch_reps: continue
                batch_ids = [r.rep_id for r in batch_reps]
                
                # Load Embeddings for this batch
                emb_map = load_embeddings(ds_name, batch_ids)
                if not emb_map: continue
                
                # Prepare X, y for clf_main
                X_batch_list = []
                y_batch_list = []
                
                # Prepare lists for folds
                # folds_data[k] = (X_list, y_list)
                folds_data = {k: ([], []) for k in range(5)}
                
                for r in batch_reps:
                    if r.rep_id not in emb_map: continue
                    emb = emb_map[r.rep_id]
                    if len(emb) == 0: continue
                    
                    y_seq = np.full(len(emb), r.label, dtype=int)
                    
                    # Add to Main (All Data)
                    X_batch_list.append(emb)
                    y_batch_list.append(y_seq)
                    
                    # Add to Folds
                    # Represents VAL fold. So it should be used to train OTHER folds.
                    r_fold = fold_map.get(r.rep_id, -1)
                    if r_fold != -1:
                        for k in range(5):
                            if k != r_fold: # If r is in val set of k, DON'T train k on it.
                                folds_data[k][0].append(emb)
                                folds_data[k][1].append(y_seq)
                                
                if not X_batch_list: continue
                
                # Update Main Model
                X_main = np.vstack(X_batch_list)
                y_main = np.concatenate(y_batch_list)
                clf_main.partial_fit(X_main, y_main, classes=classes)
                
                # Update Fold Models
                for k in range(5):
                    if folds_data[k][0]:
                        X_k = np.vstack(folds_data[k][0])
                        y_k = np.concatenate(folds_data[k][1])
                        clf_folds[k].partial_fit(X_k, y_k, classes=classes)
                
                count_processed += len(batch_reps)
                
                # release memory
                del emb_map, X_batch_list, y_batch_list, X_main, y_main, folds_data
                gc.collect()
            
            logging.info("  Training complete. Saving main model...")
            clf_main.save(model_path)
            
            # Release Fold Training Memory? No we need models for inference.
            
            # --- INFERENCE PHASE (STREAMING) ---
            logging.info("  Generating OOF Predictions (Streaming)...")
            oof_preds = []
            
            # Iterate original order or any order, doesn't matter.
            batch_iter_inf = tqdm(chunk_list(labeled_reps, BATCH_SIZE), total=(len(labeled_reps)//BATCH_SIZE)+1, desc="  OOF Batches", leave=False)
            
            for batch_reps in batch_iter_inf:
                batch_ids = [r.rep_id for r in batch_reps]
                emb_map = load_embeddings(ds_name, batch_ids)
                if not emb_map: continue
                
                for r in batch_reps:
                    if r.rep_id not in emb_map: continue
                    emb = emb_map[r.rep_id]
                    if len(emb) == 0: continue
                    
                    r_fold = fold_map.get(r.rep_id, -1)
                    if r_fold == -1: continue # Should not happen
                    
                    # Predict using model that did NOT see r
                    p = clf_folds[r_fold].predict_repertoire(emb)
                    oof_preds.append({"repertoire_id": r.rep_id, "label": r.label, "p_esm": p})
                
                del emb_map
                gc.collect()
                
            df_preds = pd.DataFrame(oof_preds)
            df_preds.to_csv(train_preds_csv, index=False)
            logging.info(f"  Saved CV preds to {train_preds_csv}")
            
            # Free fold models
            del clf_folds
            gc.collect()
        
        # --- TEST INFERENCE PHASE (STREAMING) ---
        logging.info("  Starting Test Inference...")
        
        # Load model if needed
        if clf_main is None:
            logging.info("  Loading model for test inference...")
            clf_main = ESMSequenceClassifier.load(model_path)
            
        # Find matching test datasets
        matching_test_ds = []
        for test_ds in TEST_DATASETS.keys():
            if test_ds == ds_name:
                matching_test_ds.append(test_ds)
            elif "_" in test_ds and test_ds.split("_")[0] == ds_name:
                matching_test_ds.append(test_ds)
                
        for test_ds in matching_test_ds:
            out_csv = PREDS_DIR / f"{test_ds}_test_esm_preds.csv"
            if out_csv.exists():
                logging.info(f"    ✅ Test preds exist for {test_ds}. Skipping.")
                continue

            # Try likely pickle candidates
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
                logging.warning(f"    Could not find pickle for {test_ds}. Skipping.")
                continue
                
            logging.info(f"    Inferring {test_ds} (Streaming)...")
            test_preds = []
            
            batch_iter_test = tqdm(chunk_list(reps, BATCH_SIZE), total=(len(reps)//BATCH_SIZE)+1, desc=f"    {test_ds} Batches", leave=False)
            
            for batch_reps in batch_iter_test:
                batch_ids = [r.rep_id for r in batch_reps]
                emb_map = load_embeddings(test_ds, batch_ids)
                if not emb_map: continue
                
                for r in batch_reps:
                    if r.rep_id not in emb_map: continue
                    emb = emb_map[r.rep_id]
                    if len(emb) == 0: continue
                    
                    p = clf_main.predict_repertoire(emb)
                    test_preds.append({
                        "repertoire_id": r.rep_id,
                        "p_esm": p
                    })
                del emb_map
                gc.collect()
            
            if test_preds:
                df_test = pd.DataFrame(test_preds)
                df_test.to_csv(out_csv, index=False)
                logging.info(f"    Saved test preds to {out_csv}")
            else:
                logging.warning("    No predictions generated (missing embeddings?).")

if __name__ == "__main__":
    train_esm_seq_all()
