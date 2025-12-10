import sys
from pathlib import Path

# Add project root to path to allow importing 'data' module
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import logging
import joblib
from tqdm import tqdm
from typing import Dict, List

from data.load_all_datasets import load_repertoires_pickle, TRAIN_DATASETS, PROCESSED_DIR
from malid.stats_model import RepertoireStatsClassifier

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/stats_training.log")
    ]
)

# Ensure directories exist
Path("logs").mkdir(exist_ok=True)
MODELS_DIR = Path("models/stats")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR = Path("outputs/stats_preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)

def train_stats_all():
    """
    Trains statistical models (V-gene usage, etc.) for all datasets.
    
    Workflow for each dataset:
    1. Check if model already exists (skip if so, for resumability).
    2. Load training data (repertoires).
    3. Generate Cross-Validation (CV) predictions.
       - WHY? The Meta-Ensemble needs 'unbiased' predictions on the training set
         to learn how to weigh this model. Predicting on data the model was
         trained on would lead to overfitting.
    4. Train the Final Model on ALL training data.
       - This is the model that will be used for inference on the Test set.
    """
    
    # We iterate over known training datasets (ds1..ds8)
    # Using tqdm to show overall progress across datasets
    dataset_names = list(TRAIN_DATASETS.keys())
    
    for ds_name in tqdm(dataset_names, desc="Processing Datasets"):
        logging.info(f"--- Starting {ds_name} ---")
        
        # ---------------------------------------------------------------------
        # 1. Resumability Check
        # ---------------------------------------------------------------------
        model_path = MODELS_DIR / f"{ds_name}_stats_model.joblib"
        preds_path = PREDS_DIR / f"{ds_name}_train_stats_preds.csv"
        
        # If both model and CV preds exist, we can safely skip
        if model_path.exists() and preds_path.exists():
            logging.info(f"  ✅ Model and preds already exist for {ds_name}. Skipping.")
            continue
            
        # ---------------------------------------------------------------------
        # 2. Data Loading
        # ---------------------------------------------------------------------
        pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
        if not pkl_path.exists():
            logging.warning(f"  ⚠️ Pickle not found: {pkl_path}. Skipping.")
            continue
            
        logging.info(f"  Loading repertoires from {pkl_path}...")
        reps = load_repertoires_pickle(pkl_path)
        
        # Filter for labeled data only
        labeled_reps = [r for r in reps if r.label is not None]
        if not labeled_reps:
            logging.warning("  ⚠️ No labeled data found. Skipping.")
            continue
            
        # Extract targets and IDs
        y = [r.label for r in labeled_reps]
        rep_ids = [r.rep_id for r in labeled_reps]
        
        logging.info(f"  Loaded {len(labeled_reps)} labeled repertoires.")
        
        # ---------------------------------------------------------------------
        # 3. Cross-Validation Predictions (for Meta-Ensemble)
        # ---------------------------------------------------------------------
        if not preds_path.exists():
            logging.info("  Generating CV predictions (5-fold)...")
            clf = RepertoireStatsClassifier(random_state=42)
            
            try:
                # The custom stats classifier usually runs fast, but for large datasets
                # it might take a moment.
                # cross_val_predict internally does K-Fold.
                y_proba_cv = clf.cross_val_predict(labeled_reps, y, cv=5)
                
                # Extract probability of class 1 (Positive/Disease)
                p_stats = y_proba_cv[:, 1]
                
                df_preds = pd.DataFrame({
                    "repertoire_id": rep_ids,
                    "label": y,
                    "p_stats": p_stats
                })
                df_preds.to_csv(preds_path, index=False)
                logging.info(f"  ✅ Saved CV preds to {preds_path}")
                
            except Exception as e:
                logging.error(f"  ❌ CV failed for {ds_name}: {e}")
                # We continue to try training the final model even if CV fails,
                # though usually both would fail if there's a data issue.
        else:
            logging.info("  ✅ CV preds already exist.")

        # ---------------------------------------------------------------------
        # 4. Final Model Training
        # ---------------------------------------------------------------------
        if not model_path.exists():
            logging.info("  Training final model on full dataset...")
            clf_final = RepertoireStatsClassifier(random_state=42)
            
            try:
                clf_final.fit(labeled_reps, y)
                clf_final.save(model_path)
                logging.info(f"  ✅ Saved final model to {model_path}")
            except Exception as e:
                logging.error(f"  ❌ Final training failed for {ds_name}: {e}")
        else:
            logging.info("  ✅ Final model already exists.")

    logging.info("🎉 All stats models processed.")

if __name__ == "__main__":
    train_stats_all()