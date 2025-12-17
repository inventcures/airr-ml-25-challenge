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
import random

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_all_datasets import load_repertoires_pickle, TRAIN_DATASETS, PROCESSED_DIR, TEST_DATASETS
from malid.esm_seq_model import ESMSequenceClassifier

# --- CONFIG ---
MODELS_DIR = Path("models/esm_seq_ensemble") # NEW DIRECTORY FOR FOLDS
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path("data/embeddings")

BATCH_SIZE = 5 # Strict memory optimization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/retrain_folds.log")
    ]
)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_embeddings(dataset_name: str, rep_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Load embeddings for a list of repertoires.
    """
    base_ds = dataset_name.split("_")[0]
    
    candidate_roots = [
        EMBEDDINGS_DIR / dataset_name,
        EMBEDDINGS_DIR / dataset_name / "train",
        EMBEDDINGS_DIR / dataset_name / "test",
        EMBEDDINGS_DIR / base_ds,
        EMBEDDINGS_DIR / base_ds / "train",
        EMBEDDINGS_DIR / base_ds / "test",
    ]
    
    # Specific handling for multipart test sets
    if base_ds in ["ds7", "ds8"]:
        base_test = EMBEDDINGS_DIR / base_ds / "test"
        if base_test.exists():
            candidate_roots.append(base_test / "1_test")
            candidate_roots.append(base_test / "2_test")
            candidate_roots.append(base_test / "3_test")

    valid_roots = [p for p in candidate_roots if p.exists()]
    embeddings = {}
    
    for rid in rep_ids:
        found = False
        for root in valid_roots:
            p = root / f"{rid}.npy"
            if p.exists():
                try:
                    # Use mmap_mode='r' to prevent loading full file into RAM
                    emb = np.load(p, mmap_mode='r')
                    if emb.ndim == 1:
                        if len(emb) == 0:
                            emb = np.zeros((0, 1280)) 
                        else:
                            emb = emb.reshape(1, -1)
                    embeddings[rid] = emb
                    found = True
                    break
                except Exception as e:
                    pass
            if found: break
    return embeddings

def retrain_and_save_folds():
    ds_iterator = tqdm(TRAIN_DATASETS.keys(), desc="Datasets")
    
    for ds_name in ds_iterator: # e.g. ds1_train
        ds_iterator.set_description(f"Retraining {ds_name}")
        
        # Check if Folds already exist
        all_folds_exist = True
        for k in range(5):
            fold_path = MODELS_DIR / f"{ds_name}_fold{k}.joblib"
            if not fold_path.exists():
                all_folds_exist = False
                break
        
        if all_folds_exist:
            logging.info(f"✅ All 5 folds exist for {ds_name}. Skipping.")
            continue
            
        logging.info(f"🔄 Retraining folds for {ds_name}...")
        
        # Load Data
        pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl" # Expect run on RunPod where these exist
        if not pkl_path.exists():
            # Fallback for naming convention
            pkl_path = PROCESSED_DIR / f"{ds_name}.pkl"
            
        if not pkl_path.exists():
            logging.error(f"❌ Pickle not found for {ds_name} at {pkl_path}")
            continue
            
        reps = load_repertoires_pickle(pkl_path)
        labeled_reps = [r for r in reps if r.label is not None]
        if not labeled_reps:
            continue
            
        # Assign Folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_all = [r.label for r in labeled_reps]
        fold_map = {} # rep_id -> fold_index (VAL fold)
        
        # Correctly assign fold indices
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labeled_reps, y_all)):
             for i in val_idx:
                 fold_map[labeled_reps[i].rep_id] = fold_idx
        
        # Initialize Models
        clf_folds = [ESMSequenceClassifier(random_state=42) for _ in range(5)]
        classes = np.array([0, 1])
        
        # Shuffle Training Order
        random.seed(42)
        train_order = list(labeled_reps)
        random.shuffle(train_order)
        
        # --- TRAINING LOOP ---
        batch_iter = tqdm(chunk_list(train_order, BATCH_SIZE), total=(len(train_order)//BATCH_SIZE)+1, desc="Training Batches", leave=False)
        
        for batch_reps in batch_iter:
            if not batch_reps: continue
            batch_ids = [r.rep_id for r in batch_reps]
            emb_map = load_embeddings(ds_name, batch_ids)
            if not emb_map: continue
            
            # Prepare data for each fold
            # fold k is trained on data where r_fold != k
            folds_data = {k: ([], []) for k in range(5)}
            
            for r in batch_reps:
                if r.rep_id not in emb_map: continue
                emb = emb_map[r.rep_id]
                if len(emb) == 0: continue
                
                y_seq = np.full(len(emb), r.label, dtype=int)
                r_fold = fold_map.get(r.rep_id, -1)
                
                if r_fold != -1:
                    for k in range(5):
                        if k != r_fold: # Train fold k on everything EXCEPT its val set
                            folds_data[k][0].append(emb)
                            folds_data[k][1].append(y_seq)
            
            # Update Models
            for k in range(5):
                if folds_data[k][0]:
                    X_k = np.vstack(folds_data[k][0])
                    y_k = np.concatenate(folds_data[k][1])
                    clf_folds[k].partial_fit(X_k, y_k, classes=classes)
            
            del emb_map, folds_data
            gc.collect()
            
        # SAVE MODELS
        for k in range(5):
            out_path = MODELS_DIR / f"{ds_name}_fold{k}.joblib"
            clf_folds[k].save(out_path)
            logging.info(f"  Saved fold {k} to {out_path}")
            
        del clf_folds
        gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_size", type=str, choices=["35", "650", "35m", "650m"], default="650",
                      help="Embedding model size to use: 35 (35M) or 650 (650M).")
    args = parser.parse_args()
    
    # Config
    if "35" in args.model_size:
        logging.info("🔵 Training 35M Models...")
        MODELS_DIR = Path("models/esm_seq_ensemble")
        # Ensure 35M embeddings path
        if Path("data/embeddings/35m").exists():
            EMBEDDINGS_DIR = Path("data/embeddings/35m")
        else:
            EMBEDDINGS_DIR = Path("data/embeddings")
    else:
        logging.info("🟣 Training 650M Models...")
        MODELS_DIR = Path("models/esm_seq_ensemble_650m")
        EMBEDDINGS_DIR = Path("data/embeddings")
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"  Models Output: {MODELS_DIR}")
    logging.info(f"  Embeddings Input: {EMBEDDINGS_DIR}")
    
    retrain_and_save_folds()
