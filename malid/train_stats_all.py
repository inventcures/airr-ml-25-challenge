import pandas as pd
from pathlib import Path
from typing import Dict
import joblib

from data.load_all_datasets import load_repertoires_pickle, TRAIN_DATASETS, PROCESSED_DIR
from malid.stats_model import RepertoireStatsClassifier

MODELS_DIR = Path("models/stats")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR = Path("outputs/stats_preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)

def train_stats_all():
    for ds_name in TRAIN_DATASETS.keys():
        print(f"\n[train_stats_all] Processing {ds_name}...")
        
        # Load data
        pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
        if not pkl_path.exists():
            print(f"  Skipping {ds_name}, pickle not found: {pkl_path}")
            continue
            
        model_path = MODELS_DIR / f"{ds_name}_stats_model.joblib"
        if model_path.exists():
            print(f"  Model already exists: {model_path}. Skipping.")
            continue
            
        reps = load_repertoires_pickle(pkl_path)
        
        # Filter out repertoires without labels (if any, though train should have them)
        labeled_reps = [r for r in reps if r.label is not None]
        if len(labeled_reps) < len(reps):
            print(f"  Warning: Dropped {len(reps) - len(labeled_reps)} unlabeled repertoires.")
            
        if not labeled_reps:
            print("  No labeled data found.")
            continue
            
        y = [r.label for r in labeled_reps]
        rep_ids = [r.rep_id for r in labeled_reps]
        
        # 1. Cross-validation predictions for Meta-Ensemble
        clf = RepertoireStatsClassifier(random_state=42)
        print(f"  Generating CV predictions for {len(labeled_reps)} samples...")
        try:
            # Returns (N, 2) array of probabilities
            y_proba_cv = clf.cross_val_predict(labeled_reps, y, cv=5)
            # We only need prob of positive class (1)
            p_stats = y_proba_cv[:, 1]
            
            # Save predictions
            df_preds = pd.DataFrame({
                "repertoire_id": rep_ids,
                "label": y,
                "p_stats": p_stats
            })
            out_csv = PREDS_DIR / f"{ds_name}_train_stats_preds.csv"
            df_preds.to_csv(out_csv, index=False)
            print(f"  Saved CV preds to {out_csv}")
            
        except Exception as e:
            print(f"  CV failed: {e}")
            # Fallback? Or just fail.
        
        # 2. Train final model on full dataset
        print("  Training final model...")
        clf_final = RepertoireStatsClassifier(random_state=42)
        clf_final.fit(labeled_reps, y)
        
        model_path = MODELS_DIR / f"{ds_name}_stats_model.joblib"
        clf_final.save(model_path)
        print(f"  Saved model to {model_path}")

if __name__ == "__main__":
    train_stats_all()
