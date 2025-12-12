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
    Handles nested structures: ds1/train, ds1/test, ds7/test/1_test, etc.
    Returns dict: rep_id -> embedding array (N, D)
    """
    base_ds = dataset_name.split("_")[0]
    
    # Define search candidates
    # We prioritize specific {dataset_name} matches, then {base_ds}
    candidate_roots = [
        EMBEDDINGS_DIR / dataset_name,
        EMBEDDINGS_DIR / dataset_name / "train",
        EMBEDDINGS_DIR / dataset_name / "test",
        EMBEDDINGS_DIR / base_ds,
        EMBEDDINGS_DIR / base_ds / "train",
        EMBEDDINGS_DIR / base_ds / "test",
    ]
    
    # Specific handling for multipart test sets if applicable (ds7, ds8)
    # User mentioned "1_test", "2_test" inside "test" folder
    if base_ds in ["ds7", "ds8"]:
        base_test = EMBEDDINGS_DIR / base_ds / "test"
        if base_test.exists():
            # Add subfolders generic check is tricky without glob
            # We explicitly add likely ones
            candidate_roots.append(base_test / "1_test")
            candidate_roots.append(base_test / "2_test")
            candidate_roots.append(base_test / "3_test")
            # Also "1_train"? Unlikely based on description but harmless to add
            # candidate_roots.append(EMBEDDINGS_DIR / base_ds / "train" / "1_train")

    # Filter non-existent to save IO checks
    valid_roots = [p for p in candidate_roots if p.exists()]
    
    embeddings = {}
    
    # Optimization: Cache logic would be nice, but simple path checking is okay for batch=50
    
    for rid in rep_ids:
        found = False
        for root in valid_roots:
            p = root / f"{rid}.npy"
            if p.exists():
                try:
                    # Use mmap_mode='r' to prevent loading full file into RAM
                    emb = np.load(p, mmap_mode='r')
                    # Ensure 2D
                    if emb.ndim == 1:
                        if len(emb) == 0:
                            emb = np.zeros((0, 1280)) 
                        else:
                            emb = emb.reshape(1, -1)
                    embeddings[rid] = emb
                    found = True
                    break
                except Exception as e:
                    logging.error(f"  Failed to load {path}: {e}")
                    # Continue searching other paths? No, if found but corrupt, break? 
                    # Usually if found, it's the one.
                    break
        
        if not found:
            # logging.warning(f"Embedding not found for {rid} in {dataset_name}")
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
        
        skip_training = False
        if train_preds_csv.exists() and model_path.exists():
            try:
                # Validate model format
                ESMSequenceClassifier.load(model_path)
                logging.info(f"  ✅ Training artifacts (Model + OOF Preds) exist and are valid for {ds_name}. Skipping training.")
                skip_training = True
            except Exception as e:
                logging.warning(f"  ⚠️ Found existing model for {ds_name} but it is invalid/old ({e}). Retraining...")
        
        if skip_training:
             pass
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
                 for i in val_idx: # Correction: loop over val_idx to assign fold ID
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
