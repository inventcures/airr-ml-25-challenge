
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
import joblib
from tqdm import tqdm

from data.load_all_datasets import TRAIN_DATASETS, TEST_DATASETS
from malid.meta_ensemble import MetaEnsembleClassifier

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/meta_ensemble.log")
    ]
)
Path("logs").mkdir(exist_ok=True)

STATS_PREDS_DIR = Path("outputs/stats_preds")
ESM_PREDS_DIR = Path("outputs/esm_seq_preds")
DEEPRC_PREDS_DIR = Path("outputs/deeprc_cv_preds")
CLUSTER_PREDS_DIR = Path("outputs/cluster_preds_lancedb")

MODELS_DIR = Path("models/meta")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR = Path("outputs/submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# New: Intermediate directory to prevent data loss
PARTS_DIR = SUBMISSION_DIR / "parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

def load_preds(dataset_name: str, split: str) -> pd.DataFrame:
    """
    Load and merge predictions from all models for a given dataset and split.
    """
    dfs = []
    
    # 1. Stats
    stats_path = STATS_PREDS_DIR / f"{dataset_name}_{split}_stats_preds.csv"
    if stats_path.exists():
        df = pd.read_csv(stats_path)
        dfs.append(df.set_index("repertoire_id")[["p_stats"]])
        
    # 2. ESM
    esm_path = ESM_PREDS_DIR / f"{dataset_name}_{split}_esm_preds.csv"
    if esm_path.exists():
        df = pd.read_csv(esm_path)
        dfs.append(df.set_index("repertoire_id")[["p_esm"]])
        
    # 3. DeepRC (Updated for CV)
    if split == "train":
        deeprc_path = DEEPRC_PREDS_DIR / f"{dataset_name}_oof.csv"
    else:
        deeprc_path = DEEPRC_PREDS_DIR / f"{dataset_name}_{split}_deeprc_preds.csv"
        
    if deeprc_path.exists():
        df = pd.read_csv(deeprc_path)
        dfs.append(df.set_index("repertoire_id")[["p_deeprc"]])
    else:
        logging.warning(f"  Warning: DeepRC file missing: {deeprc_path}")
        
    # 4. Cluster
    cluster_path = CLUSTER_PREDS_DIR / f"{dataset_name}_{split}_cluster_preds.csv"
    if cluster_path.exists():
        df = pd.read_csv(cluster_path)
        dfs.append(df.set_index("repertoire_id")[["p_cluster"]])
        
    if not dfs:
        return pd.DataFrame()
        
    # Merge all
    merged = pd.concat(dfs, axis=1)
    
    # Find label
    label_found = False
    for path in [stats_path, esm_path, deeprc_path, cluster_path]:
        if path.exists():
            df = pd.read_csv(path)
            if "label" in df.columns:
                labels = df.set_index("repertoire_id")["label"]
                merged["label"] = labels
                label_found = True
                break
                
    return merged.reset_index()

def train_meta_and_predict():
    all_test_preds = []
    
    # Check for existing parts to resume
    existing_parts = list(PARTS_DIR.glob("*_meta_pred.csv"))
    if existing_parts:
        logging.info(f"Found {len(existing_parts)} existing part files. These will be included in final submission.")
    
    test_ds_iter = tqdm(TEST_DATASETS.keys(), desc="Meta-Ensemble")
    
    for test_ds in test_ds_iter:
        test_ds_iter.set_description(f"Processing {test_ds}")
        
        # Check if part exists
        part_path = PARTS_DIR / f"{test_ds}_meta_pred.csv"
        if part_path.exists():
            logging.info(f"  ✅ Part file exists for {test_ds}. Skipping computation.")
            all_test_preds.append(pd.read_csv(part_path))
            continue
        
        logging.info(f"\nProcessing {test_ds}...")
        
        # Determine corresponding train dataset
        train_ds = test_ds.split("_")[0]
        if train_ds not in TRAIN_DATASETS:
            pass
             
        # Load Train Preds (CV) for Training Meta Model
        # One model per train dataset
        meta_model_path = MODELS_DIR / f"{train_ds}_meta_model.joblib"
        
        if meta_model_path.exists():
            # Load existing meta model
            clf = MetaEnsembleClassifier.load(meta_model_path)
            # Make sure we have feature cols from somewhere. 
            # We can infer them from test data or hardcode/save them. 
            # Alternatively, re-load train data to get cols.
            # Loading train data is safer to ensure consistent features.
            train_df = load_preds(train_ds, "train")
            feature_cols = [c for c in train_df.columns if c.startswith("p_")]
        else:
            # Need to train
            logging.info(f"  Loading train preds for {train_ds}...")
            train_df = load_preds(train_ds, "train")
            
            if train_df.empty or "label" not in train_df.columns:
                logging.warning(f"  Missing train preds or labels for {train_ds}. Skipping.")
                continue
                
            feature_cols = [c for c in train_df.columns if c.startswith("p_")]
            logging.info(f"  Features: {feature_cols}")
            
            X_train = train_df[feature_cols].fillna(0.5)
            y_train = train_df["label"]
            
            clf = MetaEnsembleClassifier(random_state=42)
            clf.fit(X_train, y_train)
            
            clf.save(meta_model_path)
            logging.info(f"  Saved meta model to {meta_model_path}")
        
        # Load Test Preds
        logging.info(f"  Loading test preds for {test_ds}...")
        test_df = load_preds(test_ds, "test")
        
        if test_df.empty:
            logging.warning(f"  Missing test preds for {test_ds}. Skipping.")
            continue
            
        # Predict
        for col in feature_cols:
            if col not in test_df.columns:
                logging.warning(f"  Warning: {col} missing in test set. Filling with 0.5.")
                test_df[col] = 0.5
                
        X_test = test_df[feature_cols].fillna(0.5)
        
        probs = clf.predict_proba(X_test)[:, 1]
        
        # Prepare result
        results = pd.DataFrame({
            "repertoire_id": test_df["repertoire_id"],
            "probability": probs
        })
        
        # SAVE PART IMMEDIATELY
        results.to_csv(part_path, index=False)
        logging.info(f"  Saved part file to {part_path}")
        
        all_test_preds.append(results)
        
    if all_test_preds:
        submission = pd.concat(all_test_preds)
        out_csv = SUBMISSION_DIR / "submission.csv"
        submission.to_csv(out_csv, index=False)
        logging.info(f"\nSaved submission to {out_csv} with {len(submission)} rows.")
    else:
        logging.warning("\nNo predictions generated.")

if __name__ == "__main__":
    train_meta_and_predict()
