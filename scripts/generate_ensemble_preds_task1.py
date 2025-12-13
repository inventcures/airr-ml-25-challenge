
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
MODELS_DIR = Path("models/esm_seq_ensemble")
OUTPUT_DIR = Path("outputs/esm_seq_preds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path("data/embeddings")

BATCH_SIZE = 16 # Faster inference than training

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/generate_ensemble_preds.log")
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

def predict_repertoire_ensemble(ensemble_models: List[ESMSequenceClassifier], emb: np.ndarray) -> float:
    """
    Predict using all models in ensemble and Average.
    """
    if len(emb) == 0:
        return 0.5
        
    probs = []
    for model in ensemble_models:
        probs.append(model.predict_repertoire(emb))
        
    return float(np.mean(probs))

def generate_train_oof_preds(ds_name: str):
    logging.info(f"Generating OOF Preds for {ds_name}...")
    
    pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
    if not pkl_path.exists(): pkl_path = PROCESSED_DIR / f"{ds_name}.pkl"
    
    if not pkl_path.exists():
        logging.error(f"  ❌ Pickle not found: {pkl_path}")
        return

    reps = load_repertoires_pickle(pkl_path)
    labeled_reps = [r for r in reps if r.label is not None]
    
    if not labeled_reps:
        logging.warning(f"  No labeled reps for {ds_name}.")
        return

    # Models will be loaded after filtering to save time
        
    # 2. Re-create Splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_all = [r.label for r in labeled_reps]
    
    results = []
    
    # 3. Predict per Fold (OOF)
    # The split yields (train_idx, val_idx). 
    # For val_idx, we use fold `k` model?
    # Wait, in retrain_save_folds:
    #   if k != r_fold: train fold k on everything EXCEPT its val set
    #   So fold k model was trained on [All - Fold K Data].
    #   Thus, fold k model has NOT seen Fold K Data.
    #   So for data in Fold K, we usage Model K.
    

        
    work_items = []
    fold_map = {} # just to identify which models we truly need if we wanted to be lazy, but let's load all.
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labeled_reps, y_all)):
         for i in val_idx:
            rep = labeled_reps[i]
            if str(rep.rep_id) not in processed_ids:
                work_items.append((rep, fold_idx)) 

    if not work_items:
        logging.info(f"  ✅ All items already processed for {ds_name}.")
        return

    # Valid work found, load models now
    models = []
    for k in range(5):
        m_path = MODELS_DIR / f"{ds_name}_fold{k}.joblib"
        if not m_path.exists():
            logging.error(f"  ❌ Model fold {k} missing for {ds_name} at {m_path}")
            return
        models.append(ESMSequenceClassifier.load(m_path))

    logging.info(f"  Processing {len(work_items)} remaining items for {ds_name}...")
    
    # Process in chunks
    # Work items is list of (rep, fold_idx)
    batch_iter = tqdm(chunk_list(work_items, BATCH_SIZE), total=(len(work_items)//BATCH_SIZE)+1, desc=f"OOF {ds_name}")
    
    for batch_items in batch_iter:
        batch_reps = [x[0] for x in batch_items]
        batch_fold_idxs = [x[1] for x in batch_items]
        batch_ids = [r.rep_id for r in batch_reps]
        
        emb_map = load_embeddings(ds_name, batch_ids)
        
        batch_results = []
        for i, r in enumerate(batch_reps):
            if r.rep_id not in emb_map:
                batch_results.append({"repertoire_id": r.rep_id, "p_esm": 0.5, "label": r.label})
                continue
                
            emb = emb_map[r.rep_id]
            model = models[batch_fold_idxs[i]] # Select correct fold model
            p = model.predict_repertoire(emb)
            
            batch_results.append({"repertoire_id": r.rep_id, "p_esm": p, "label": r.label})
            
        # Write Batch
        df_batch = pd.DataFrame(batch_results)
        header = not out_path.exists()
        df_batch.to_csv(out_path, mode='a', header=header, index=False)
        
    logging.info(f"  ✅ Completed OOF preds for {ds_name}")

def generate_test_ensemble_preds(ds_name: str, force: bool = False):
    logging.info(f"Generating TEST Ensemble Preds for {ds_name}...")
    
    pkl_path = PROCESSED_DIR / f"{ds_name}_test.pkl"
    if not pkl_path.exists():
        logging.error(f"  ❌ Pickle not found: {pkl_path}")
        return

    out_path = OUTPUT_DIR / f"{ds_name}_test_esm_preds.csv"

    if force and out_path.exists():
        logging.info(f"  ⚠️ Force enabled: Deleting existing file {out_path}")
        out_path.unlink()
        
    # Check resumption
    processed_ids = set()
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            if "repertoire_id" in existing_df.columns:
                processed_ids = set(existing_df["repertoire_id"].astype(str))
                logging.info(f"  🔄 Resuming {ds_name}: Found {len(processed_ids)} already processed.")
        except Exception as e:
            logging.warning(f"  ⚠️ Could not read existing file {out_path}: {e}")

    # Load Data
    reps = load_repertoires_pickle(pkl_path)
    
    # Filter work
    reps_to_process = [r for r in reps if str(r.rep_id) not in processed_ids]
    
    if not reps_to_process:
        logging.info(f"  ✅ All items already processed for {ds_name}.")
        return

    # Identify Base Train DS (to load models)
    if "ds7_" in ds_name: base_ds = "ds7"
    elif "ds8_" in ds_name: base_ds = "ds8"
    else: base_ds = ds_name # ds1, ds2...
    
    # Load Models
    models = []
    for k in range(5):
        m_path = MODELS_DIR / f"{base_ds}_fold{k}.joblib"
        if not m_path.exists():
             logging.warning(f"  ⚠️ Model fold {k} missing for {base_ds}. Ensemble incomplete.")
             continue
        models.append(ESMSequenceClassifier.load(m_path))
        
    if not models:
        logging.error(f"  ❌ No models found for {base_ds}. Skipping {ds_name}.")
        return

    logging.info(f"  Processing {len(reps_to_process)} remaining items for {ds_name}...")

    # Predict
    batch_iter = tqdm(chunk_list(reps_to_process, BATCH_SIZE), total=(len(reps_to_process)//BATCH_SIZE)+1, desc=f"Predicting {ds_name}")
    
    for batch_reps in batch_iter:
        batch_ids = [r.rep_id for r in batch_reps]
        emb_map = load_embeddings(ds_name, batch_ids)
        
        batch_results = []
        for r in batch_reps:
            if r.rep_id not in emb_map:
                batch_results.append({"repertoire_id": r.rep_id, "p_esm": 0.5})
                continue
                
            emb = emb_map[r.rep_id]
            p_avg = predict_repertoire_ensemble(models, emb)
            
            batch_results.append({"repertoire_id": r.rep_id, "p_esm": p_avg})
            
        # Write Batch
        df_batch = pd.DataFrame(batch_results)
        header = not out_path.exists()
        df_batch.to_csv(out_path, mode='a', header=header, index=False)
        
    logging.info(f"  ✅ Completed Ensemble preds for {ds_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate Ensemble Predictions")
    parser.add_argument("--force", action="store_true", help="Overwrite existing prediction files")
    args = parser.parse_args()

    # 1. Process Train (OOF)
    for ds_name in TRAIN_DATASETS.keys():
        generate_train_oof_preds(ds_name, force=args.force)
        
    # 2. Process Test (Ensemble)
    for ds_name in TEST_DATASETS.keys():
        generate_test_ensemble_preds(ds_name, force=args.force)

if __name__ == "__main__":
    main()
