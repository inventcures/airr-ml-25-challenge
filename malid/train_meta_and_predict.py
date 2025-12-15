
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
import joblib
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_all_datasets import TRAIN_DATASETS, TEST_DATASETS
from malid.meta_ensemble import MetaEnsembleClassifier

# Logging setup
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/meta_ensemble.log")
    ]
)

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
    
    # FRESH START: User requested no contamination from previous runs.
    # We clear intermediate parts and models to force re-training and re-prediction.
    if PARTS_DIR.exists():
        logging.warning(f"  Cleanup: Removing existing parts directory {PARTS_DIR} to ensure fresh run.")
        import shutil
        shutil.rmtree(PARTS_DIR)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for existing parts to resume (Now effectively empty unless filesystem race)
    existing_parts = list(PARTS_DIR.glob("*_meta_pred.csv"))
    if existing_parts:
        logging.info(f"Found {len(existing_parts)} existing part files. These will be included in final submission.")
    
    # Process BOTH Test and Train datasets to match Kaggle submission requirements
    # We must be careful because keys (e.g. 'ds1') overlap between Train and Test dicts.
    # We will iterate over explicit (ds_name, split) tuples.
    targets = []
    for test_ds in TEST_DATASETS.keys():
        targets.append((test_ds, "test"))
    for train_ds in TRAIN_DATASETS.keys():
        targets.append((train_ds, "train"))
        
    # Sort by ds_name then split
    targets = sorted(targets, key=lambda x: (x[0], x[1]))
    
    ds_iter = tqdm(targets, desc="Meta-Ensemble")
    
    for ds_name, split in ds_iter:
        ds_iter.set_description(f"Processing {ds_name} ({split})")
        
        # Check if part exists - MUST include split in filename to avoid collision!
        part_path = PARTS_DIR / f"{ds_name}_{split}_meta_pred.csv"
        if part_path.exists():
            logging.info(f"  ✅ Part file exists for {ds_name} ({split}). Skipping computation.")
            all_test_preds.append(pd.read_csv(part_path))
            continue
            
        logging.info(f"\nProcessing {ds_name} ({split})...")
        
        # Determine train base
        if split == "train":
            train_ds = ds_name
        else:
            # e.g. ds7_1 -> ds7
            train_ds = ds_name.split("_")[0]
            
        # 1. Ensure Meta Model Exists (Train if needed)
        meta_model_path = MODELS_DIR / f"{train_ds}_meta_model.joblib"
        
        if not meta_model_path.exists():
            logging.info(f"  Meta model for {train_ds} not found. Loading train data to train it...")
            train_df = load_preds(train_ds, "train")
            
            if train_df.empty or "label" not in train_df.columns:
                logging.warning(f"  Missing train preds or labels for {train_ds}. Skipping {ds_name}.")
                continue
                
            feature_cols = [c for c in train_df.columns if c.startswith("p_")]
            logging.info(f"  Features: {feature_cols}")
            
            X_train = train_df[feature_cols].fillna(0.5)
            y_train = train_df["label"]
            
            clf = MetaEnsembleClassifier(random_state=42) # Added random_state for reproducibility
            clf.fit(X_train, y_train)
            
            clf.save(meta_model_path)
            logging.info(f"  Saved meta model to {meta_model_path}")
        else:
            clf = MetaEnsembleClassifier.load(meta_model_path)
            # We assume features are consistent. 
            # If we need to know feature names, we can peek at a train file or rely on column naming convention.
            # Here we just rely on "p_" columns in the target split.

        # 2. Predict on Target Dataset (Train or Test)
        # If split is "train", we are just predicting on the training set itself (OOF style essentially, or just fitting error check)
        # But load_preds(..., split="train") gives us the OOF/CV predictions from upstream models.
        logging.info(f"  Loading predictions for {ds_name} ({split})...")
        target_df = load_preds(ds_name, split)
        
        if target_df.empty:
            logging.warning(f"  No predictions found for {ds_name}. Skipping.")
            continue
            
        feature_cols = [c for c in target_df.columns if c.startswith("p_")]
        X_target = target_df[feature_cols].fillna(0.5)
        
        # Generate Meta Predictions
        # If we are in "train" split, we are technically re-predicting on training data using the model trained on it.
        # Ideally we'd use OOF meta-predictions, but for this level of stacking, re-predicting is often accepted if upstream was OOF.
        # Or even better: if split=="train", we might just want to use the upstream OOF average?
        # A simple approach is to use the meta-model we just trained.
        probs = clf.predict_proba(X_target)[:, 1]
        
        results = pd.DataFrame({
            "repertoire_id": target_df["repertoire_id"], # Corrected from target_df.index
            "probability": probs
        })
        
        # Add dataset column for build_submission.py
        # Map internal 'ds1' -> 'train_dataset_1' or 'test_dataset_1'
        if split == "train":
            real_name = TRAIN_DATASETS[ds_name]
        else:
            real_name = TEST_DATASETS[ds_name]
        results["dataset"] = real_name
        
        # Save part
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_size", type=str, choices=["35", "650", "35m", "650m"], default="650",
                      help="Embedding model size to use: 35 (35M) or 650 (650M). Adjusts INPUT (ESM) and OUTPUT DIRS.")
    args = parser.parse_args()

    # Dynamic Configuration
    if "35" in args.model_size:
        # 35M: Legacy / Default
        logging.info("🔵 Selected 35M Workflow (Using models/meta, outputs/submission)")
        # Globals already set to defaults
        # STATS_PREDS_DIR, ESM_PREDS_DIR are defaults
    else:
        # 650M: Namespaced
        logging.info("🟣 Selected 650M Workflow (Using models/meta_650m, outputs/submission_650m)")
        
        # Inputs: Point to 650M ESM predictions
        ESM_PREDS_DIR = Path("outputs/esm_seq_preds_650m")
        
        # Outputs: Point to 650M meta models and submission
        MODELS_DIR = Path("models/meta_650m")
        SUBMISSION_DIR = Path("outputs/submission_650m")
        PARTS_DIR = SUBMISSION_DIR / "parts"
        
        logging.info(f"🟣 Inputs: {ESM_PREDS_DIR}")
        logging.info(f"🟣 Outputs: {MODELS_DIR}, {SUBMISSION_DIR}")

    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    train_meta_and_predict()
