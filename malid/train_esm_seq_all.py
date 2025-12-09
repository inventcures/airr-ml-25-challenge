import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import joblib

from data.load_all_datasets import load_repertoires_pickle, TRAIN_DATASETS, PROCESSED_DIR
from malid.esm_seq_model import ESMSequenceClassifier

MODELS_DIR = Path("models/esm_seq")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR = Path("outputs/esm_seq_preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path("data/embeddings")

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
        print(f"  Warning: Embeddings dir not found: {emb_dir}")
        return {}
        
    # Pre-scan for all .npy files
    file_map = {}
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
                print(f"  Failed to load {path}: {e}")
        else:
            # print(f"  Missing embedding for {rid}")
            pass
    return embeddings

def train_esm_seq_all():
    for ds_name in TRAIN_DATASETS.keys():
        print(f"\n[train_esm_seq_all] Processing {ds_name}...")
        
        # Load data
        pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
        if not pkl_path.exists():
            print(f"  Skipping {ds_name}, pickle not found.")
            continue
            
        model_path = MODELS_DIR / f"{ds_name}_esm_seq_model.joblib"
        if model_path.exists():
            print(f"  Model already exists: {model_path}. Skipping.")
            continue
            
        reps = load_repertoires_pickle(pkl_path)
        
        # Filter labeled
        labeled_reps = [r for r in reps if r.label is not None]
        if not labeled_reps:
            print("  No labeled data.")
            continue
            
        # Load embeddings
        print("  Loading embeddings...")
        rep_ids = [r.rep_id for r in labeled_reps]
        embeddings_map = load_embeddings(ds_name, rep_ids)
        
        if not embeddings_map:
            print("  No embeddings found. Skipping training.")
            continue
            
        # Prepare sequence-level dataset
        X_seq_list = []
        y_seq_list = []
        
        # Keep track of which sequences belong to which repertoire for CV aggregation
        # But for training the sequence classifier, we just need the bag of sequences.
        
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
            print("  No valid embeddings after filtering.")
            continue
            
        X_seq_all = np.vstack(X_seq_list)
        y_seq_all = np.concatenate(y_seq_list)
        
        print(f"  Training on {len(X_seq_all)} sequences from {len(valid_reps)} repertoires...")
        
        # 1. Train final model
        clf = ESMSequenceClassifier(random_state=42)
        clf.fit(X_seq_all, y_seq_all)
        
        model_path = MODELS_DIR / f"{ds_name}_esm_seq_model.joblib"
        clf.save(model_path)
        print(f"  Saved model to {model_path}")
        
        # 2. Generate predictions (CV-like?)
        # ---------------------------------------------------------------------
        # EDUCATIONAL NOTE: Cross-Validation (CV) for Meta-Ensembling
        # ---------------------------------------------------------------------
        # We need to generate predictions for our TRAINING data to train the Meta-Ensemble.
        # BUT, if we use the model we just trained on all data, the predictions will be
        # "too good" (overfitted) because the model has already seen the answers.
        #
        # Solution: K-Fold Cross-Validation
        # 1. Split data into K parts (e.g., 5 folds).
        # 2. For each fold:
        #    - Train a temporary model on the OTHER 4 folds.
        #    - Predict on the current fold (which the model hasn't seen).
        # 3. Combine these "Out-of-Fold" (OOF) predictions.
        #
        # Result: We get predictions for the entire dataset where the model never saw
        # the specific sample it was predicting on during training. This simulates
        # how the model will perform on new, unseen test data.
        # ---------------------------------------------------------------------
        
        print("  Generating CV predictions...")
        from sklearn.model_selection import StratifiedKFold
        # StratifiedKFold ensures each fold has the same percentage of positive/negative samples.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # We need to split repertoires, not sequences
        y_reps = [r.label for r in valid_reps]
        rep_ids_valid = [r.rep_id for r in valid_reps]
        
        cv_preds = [] # (rep_id, p_esm)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(valid_reps, y_reps)):
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
        out_csv = PREDS_DIR / f"{ds_name}_train_esm_preds.csv"
        df_preds.to_csv(out_csv, index=False)
        print(f"  Saved CV preds to {out_csv}")
        
        # 3. Inference on Test Sets
        # Find all test datasets that correspond to this train dataset
        # e.g. ds1 -> ds1 (test), ds7 -> ds7_1 (test), ds7_2 (test)
        
        from data.load_all_datasets import TEST_DATASETS
        
        # Heuristic: Find test datasets that start with the train dataset name
        # But be careful: ds1 matches ds1, but ds10 would match ds1 if we just check startswith?
        # Our naming is ds1..ds8.
        
        matching_test_ds = []
        for test_ds in TEST_DATASETS.keys():
            # ds1 -> ds1
            # ds7_1 -> ds7
            if test_ds == ds_name:
                matching_test_ds.append(test_ds)
            elif "_" in test_ds and test_ds.split("_")[0] == ds_name:
                matching_test_ds.append(test_ds)
                
        for test_ds in matching_test_ds:
            print(f"  Inferring on test dataset: {test_ds}...")
            pkl_path = PROCESSED_DIR / f"{test_ds}_test.pkl" # Note: load_all_datasets saves as {name}_{split}.pkl. For ds7_1, split is "1_test"? No, wait.
            # In load_all_datasets:
            # ds7_1: split="1_test" -> pickle name "ds7_1_1_test.pkl"? 
            # Let's check load_all_datasets.py logic or just check file existence.
            # Actually, the user provided table says: ds7 split "1_test".
            # But in load_all_datasets, we might have flattened it.
            # Let's try standard naming first.
            
            # Actually, let's use the load_repertoires_pickle directly on the expected path
            # The keys in TEST_DATASETS are "ds1", "ds7_1", etc.
            # The splits are usually "test".
            # But for ds7_1, the split name in the object might be "1_test".
            # The pickle filename is what matters.
            
            # Try likely candidates
            candidates = [
                PROCESSED_DIR / f"{test_ds}_test.pkl",
                PROCESSED_DIR / f"{test_ds}_1_test.pkl", # if split was part of name?
            ]
            
            reps = None
            for p in candidates:
                if p.exists():
                    reps = load_repertoires_pickle(p)
                    break
            
            if not reps:
                # Fallback: check data/data
                # But we should have processed it.
                print(f"    Could not find pickle for {test_ds}. Skipping.")
                continue
                
            # Load embeddings
            # We use the robust load_embeddings which handles ds7_1 -> ds7 mapping internally if we pass ds7_1
            # Wait, load_embeddings(dataset_name, ...)
            # We should pass test_ds (e.g. ds7_1) and it will map to ds7 folder.
            rep_ids = [r.rep_id for r in reps]
            test_embeddings_map = load_embeddings(test_ds, rep_ids)
            
            test_preds = []
            for r in reps:
                if r.rep_id not in test_embeddings_map:
                    # Missing embedding
                    # print(f"    Missing embedding for {r.rep_id}")
                    continue
                    
                emb = test_embeddings_map[r.rep_id]
                if len(emb) == 0:
                    continue
                    
                # Predict
                p = clf.predict_repertoire(emb)
                test_preds.append({
                    "repertoire_id": r.rep_id,
                    # "label": r.label, # Test might not have label, or we don't use it
                    "p_esm": p
                })
            
            if test_preds:
                df_test = pd.DataFrame(test_preds)
                # Output filename: matches what train_meta_and_predict expects
                # It expects: {dataset_name}_{split}_esm_preds.csv
                # For ds7_1, split is "test" (implicitly)? 
                # train_meta_and_predict loads: f"{dataset_name}_{split}_esm_preds.csv"
                # If we run it for ds7_1, split "test".
                out_csv = PREDS_DIR / f"{test_ds}_test_esm_preds.csv"
                df_test.to_csv(out_csv, index=False)
                print(f"    Saved test preds to {out_csv}")
            else:
                print("    No predictions generated (missing embeddings?).")

if __name__ == "__main__":
    train_esm_seq_all()
