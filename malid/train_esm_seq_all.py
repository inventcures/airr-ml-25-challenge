
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List
import joblib
from tqdm import tqdm

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
        
    # Pre-scan for all .npy files
    file_map = {}
    # Use rglob just in case of nested structures, though we expect flat now
    for p in emb_dir.rglob("*.npy"):
        file_map[p.stem] = p
        
    for rid in rep_ids:
        # Check map
        if str(rid) in file_map:
            path = file_map[str(rid)]
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
            pass
    return embeddings

def train_esm_seq_all():
    # Progress bar for datasets
    ds_iterator = tqdm(TRAIN_DATASETS.keys(), desc="Datasets")
    
    for ds_name in ds_iterator:
        ds_iterator.set_description(f"Processing {ds_name}")
        logging.info(f"Processing {ds_name}...")
        
        # Reset model variable to prevent leakage from previous iteration
        clf = None
        
        # Check if TRAINING OOF preds ALREADY EXIST
        train_preds_csv = PREDS_DIR / f"{ds_name}_train_esm_preds.csv"
        model_path = MODELS_DIR / f"{ds_name}_esm_seq_model.joblib"
        
        # We need both the model and the OOF preds.
        if train_preds_csv.exists() and model_path.exists():
            logging.info(f"  ✅ Training artifacts (Model + OOF Preds) exist for {ds_name}. Skipping training.")
        else:
            # --- TRAINING PHASE ---
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
                
            # Load embeddings
            logging.info("  Loading embeddings...")
            rep_ids = [r.rep_id for r in labeled_reps]
            embeddings_map = load_embeddings(ds_name, rep_ids)
            
            if not embeddings_map:
                logging.warning("  No embeddings found. Skipping training.")
                continue
                
            # Prepare sequence-level dataset
            X_seq_list = []
            y_seq_list = []
            valid_reps = []
            
            for r in labeled_reps:
                if r.rep_id not in embeddings_map:
                    continue
                emb = embeddings_map[r.rep_id]
                if len(emb) == 0:
                    continue
                    
                X_seq_list.append(emb)
                # Broadcast label
                y_seq_list.append(np.full(len(emb), r.label, dtype=int))
                valid_reps.append(r)
                
            if not X_seq_list:
                logging.warning("  No valid embeddings after filtering.")
                continue
                
            X_seq_all = np.vstack(X_seq_list)
            y_seq_all = np.concatenate(y_seq_list)
            
            logging.info(f"  Training on {len(X_seq_all)} sequences from {len(valid_reps)} repertoires...")
            
            # 1. Train final model
            clf = ESMSequenceClassifier(random_state=42)
            clf.fit(X_seq_all, y_seq_all)
            clf.save(model_path)
            logging.info(f"  Saved model to {model_path}")
            
            # 2. Generate OOF predictions via CV
            logging.info("  Generating CV predictions (OOF)...")
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            y_reps = [r.label for r in valid_reps]
            
            cv_preds = [] 
            
            # Tqdm for folds
            fold_iter = tqdm(skf.split(valid_reps, y_reps), total=5, desc="CV Folds", leave=False)
            
            for fold, (train_idx, val_idx) in enumerate(fold_iter):
                train_reps = [valid_reps[i] for i in train_idx]
                val_reps = [valid_reps[i] for i in val_idx]
                
                # Build train seq dataset
                X_train_list = []
                y_train_list = []
                for r in train_reps:
                    emb = embeddings_map[r.rep_id]
                    X_train_list.append(emb)
                    y_train_list.append(np.full(len(emb), r.label, dtype=int))
                    
                X_train = np.vstack(X_train_list)
                y_train = np.concatenate(y_train_list)
                
                # Train fold model
                clf_fold = ESMSequenceClassifier(random_state=42)
                clf_fold.fit(X_train, y_train)
                
                # Predict val repertoires
                for r in val_reps:
                    emb = embeddings_map[r.rep_id]
                    p = clf_fold.predict_repertoire(emb)
                    cv_preds.append({"repertoire_id": r.rep_id, "label": r.label, "p_esm": p})
                    
            df_preds = pd.DataFrame(cv_preds)
            df_preds.to_csv(train_preds_csv, index=False)
            logging.info(f"  Saved CV preds to {train_preds_csv}")
        
        # --- INFERENCE PHASE ---
        # Find matching test datasets
        matching_test_ds = []
        for test_ds in TEST_DATASETS.keys():
            # ds1 -> ds1
            # ds7_1 -> ds7
            if test_ds == ds_name:
                matching_test_ds.append(test_ds)
            elif "_" in test_ds and test_ds.split("_")[0] == ds_name:
                matching_test_ds.append(test_ds)
                
        for test_ds in matching_test_ds:
            out_csv = PREDS_DIR / f"{test_ds}_test_esm_preds.csv"
            if out_csv.exists():
                logging.info(f"    ✅ Test preds exist for {test_ds}. Skipping.")
                continue

            logging.info(f"  Inferring on test dataset: {test_ds}...")
            
            # Load model (make sure it's loaded)
            if clf is None:
                clf = ESMSequenceClassifier.load(model_path)
            
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
                
            logging.info(f"    Loading test embeddings for {test_ds}...")
            rep_ids = [r.rep_id for r in reps]
            test_embeddings_map = load_embeddings(test_ds, rep_ids)
            
            test_preds = []
            for r in tqdm(reps, desc=f"Inferring {test_ds}", leave=False):
                if r.rep_id not in test_embeddings_map:
                    continue
                    
                emb = test_embeddings_map[r.rep_id]
                if len(emb) == 0:
                    continue
                    
                p = clf.predict_repertoire(emb)
                test_preds.append({
                    "repertoire_id": r.rep_id,
                    "p_esm": p
                })
            
            if test_preds:
                df_test = pd.DataFrame(test_preds)
                df_test.to_csv(out_csv, index=False)
                logging.info(f"    Saved test preds to {out_csv}")
            else:
                logging.warning("    No predictions generated (missing embeddings?).")

if __name__ == "__main__":
    train_esm_seq_all()
