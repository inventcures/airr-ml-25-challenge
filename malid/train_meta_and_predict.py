import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from data.load_all_datasets import TRAIN_DATASETS, TEST_DATASETS
from malid.meta_ensemble import MetaEnsembleClassifier

STATS_PREDS_DIR = Path("outputs/stats_preds")
ESM_PREDS_DIR = Path("outputs/esm_seq_preds")
DEEPRC_PREDS_DIR = Path("outputs/deeprc_preds")
CLUSTER_PREDS_DIR = Path("outputs/cluster_preds")

MODELS_DIR = Path("models/meta")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR = Path("outputs/submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

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
        
    # 3. DeepRC
    deeprc_path = DEEPRC_PREDS_DIR / f"{dataset_name}_{split}_deeprc_preds.csv"
    if deeprc_path.exists():
        df = pd.read_csv(deeprc_path)
        dfs.append(df.set_index("repertoire_id")[["p_deeprc"]])
        
    # 4. Cluster (TODO)
    cluster_path = CLUSTER_PREDS_DIR / f"{dataset_name}_{split}_cluster_preds.csv"
    if cluster_path.exists():
        df = pd.read_csv(cluster_path)
        dfs.append(df.set_index("repertoire_id")[["p_cluster"]])
        
    if not dfs:
        return pd.DataFrame()
        
    # Merge all
    merged = pd.concat(dfs, axis=1)
    
    # We need labels for training
    # For training, one of the files should have 'label'.
    # Let's try to find it.
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
    
    # Iterate over TEST datasets to produce submission
    # We assume each test dataset maps to a train dataset model.
    
    for test_ds in TEST_DATASETS.keys():
        print(f"\nProcessing {test_ds}...")
        
        # Determine corresponding train dataset
        # Heuristic: ds1 -> ds1, ds7_1 -> ds7
        train_ds = test_ds.split("_")[0]
        if train_ds not in TRAIN_DATASETS:
             # Try stripping suffix if present
             pass
             
        # Load Train Preds (CV)
        print(f"  Loading train preds for {train_ds}...")
        train_df = load_preds(train_ds, "train")
        
        if train_df.empty or "label" not in train_df.columns:
            print(f"  Missing train preds or labels for {train_ds}. Skipping.")
            continue
            
        # Train Meta Model
        # ---------------------------------------------------------------------
        # EDUCATIONAL NOTE: Stacking / Meta-Ensembling
        # ---------------------------------------------------------------------
        # We are now training a "Meta-Model" (Logistic Regression) that takes the
        # predictions of our base models (Stats, ESM, DeepRC) as INPUTS.
        #
        # Why?
        # 1. Diversity: Different models learn different things.
        #    - Stats model sees global repertoire features (V-gene usage).
        #    - ESM model sees sequence-level patterns (embeddings).
        #    - DeepRC sees MIL patterns (bags of sequences).
        # 2. Robustness: If one model makes a mistake, the others can correct it.
        # 3. Calibration: The meta-model learns which base model to trust more.
        #
        # Input (X): [p_stats, p_esm, p_deeprc]
        # Output (y): Disease Label (0 or 1)
        # ---------------------------------------------------------------------
        feature_cols = [c for c in train_df.columns if c.startswith("p_")]
        print(f"  Features: {feature_cols}")
        
        X_train = train_df[feature_cols].fillna(0.5)
        y_train = train_df["label"]
        
        clf = MetaEnsembleClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        # Save model
        model_path = MODELS_DIR / f"{train_ds}_meta_model.joblib"
        clf.save(model_path)
        
        # Load Test Preds
        print(f"  Loading test preds for {test_ds}...")
        test_df = load_preds(test_ds, "test")
        
        if test_df.empty:
            print(f"  Missing test preds for {test_ds}. Skipping.")
            continue
            
        # Predict
        X_test = test_df[feature_cols] # Will fill missing with 0.5 inside predict_proba if needed
        # But we should ensure columns match what was trained
        
        probs = clf.predict_proba(X_test)[:, 1]
        
        # Prepare submission rows
        # We need repertoire_id
        results = pd.DataFrame({
            "repertoire_id": test_df["repertoire_id"],
            "probability": probs
        })
        all_test_preds.append(results)
        
    if all_test_preds:
        submission = pd.concat(all_test_preds)
        out_csv = SUBMISSION_DIR / "submission.csv"
        submission.to_csv(out_csv, index=False)
        print(f"\nSaved submission to {out_csv} with {len(submission)} rows.")
    else:
        print("\nNo predictions generated.")

if __name__ == "__main__":
    train_meta_and_predict()
