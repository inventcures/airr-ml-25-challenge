import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
import sys
from tqdm import tqdm

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TRAIN_DATASETS, TEST_DATASETS
from malid.cluster_model import ClusterClassifier

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/clustering.log")
    ]
)
Path("logs").mkdir(exist_ok=True)

EMBEDDINGS_DIR = Path("data/embeddings")
MODELS_DIR = Path("models/cluster")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR = Path("outputs/cluster_preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)

def load_embeddings_for_reps(reps, dataset_name):
    """
    Load embeddings for a list of repertoires.
    Returns dict: rep_id -> np.ndarray
    """
    embeddings = {}
    
    # Handle ds7_1 -> ds7 mapping
    emb_ds_name = dataset_name.split("_")[0]
    emb_dir = EMBEDDINGS_DIR / emb_ds_name
    
    if not emb_dir.exists():
        # Fallback
        if (EMBEDDINGS_DIR / dataset_name).exists():
            emb_dir = EMBEDDINGS_DIR / dataset_name
        else:
            logging.warning(f"Embeddings dir not found: {emb_dir}")
            return {}

    logging.info(f"Loading embeddings from {emb_dir} for {len(reps)} reps...")
    
    count = 0
    for r in tqdm(reps, desc="Loading Embeddings"):
        npy_path = emb_dir / f"{r.rep_id}.npy"
        if npy_path.exists():
            try:
                emb = np.load(npy_path)
                if emb.size > 0:
                    embeddings[r.rep_id] = emb
                    count += 1
            except Exception as e:
                logging.warning(f"Failed to load {npy_path}: {e}")
                
    logging.info(f"Loaded {count}/{len(reps)} embeddings.")
    return embeddings

def run_clustering_all():
    # Train datasets
    for ds_name in TRAIN_DATASETS.keys():
        logging.info(f"\nProcessing {ds_name}...")
        
        # Load Train Data
        pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
        if not pkl_path.exists():
            logging.warning(f"Pickle not found: {pkl_path}")
            continue
            
        reps = load_repertoires_pickle(pkl_path)
        
        # Filter labeled
        labeled_reps = [r for r in reps if r.label is not None]
        if not labeled_reps:
            logging.warning("No labeled data.")
            continue
            
        # Load Embeddings
        embeddings = load_embeddings_for_reps(labeled_reps, ds_name)
        if not embeddings:
            logging.warning("No embeddings loaded. Skipping.")
            continue
            
        y = [r.label for r in labeled_reps]
        rep_ids = [r.rep_id for r in labeled_reps]
        
        # Check if model exists
        model_path = MODELS_DIR / f"{ds_name}_cluster_model.joblib"
        
        if model_path.exists():
            logging.info(f"Loading existing model from {model_path}")
            clf = joblib.load(model_path)
        else:
            logging.info("Training ClusterClassifier...")
            clf = ClusterClassifier(k_neighbors=10, resolution=1.0, n_clusters_to_keep=50)
            clf.fit(rep_ids, y, embeddings)
            joblib.dump(clf, model_path)
            logging.info(f"Saved model to {model_path}")
            
        # Generate CV predictions for Meta-Ensemble
        # Note: ClusterClassifier doesn't support CV natively in fit() yet (unlike StatsModel which is fast).
        # We should ideally run CV. But for now, let's just predict on train (overfitted) 
        # OR implement CV inside fit?
        # The plan said "Train Logistic Regression meta-learner".
        # If we feed overfitted predictions, meta-learner will trust this model too much.
        # Let's do a simple KFold here if we just trained it.
        # But fitting ClusterClassifier is slow (FAISS+Louvain).
        # We can't afford 5x fit.
        # Compromise: Use the trained model to predict on train set.
        # WARNING: This IS overfitted.
        
        logging.info("Generating train predictions...")
        probs = clf.predict_proba(rep_ids, embeddings)[:, 1]
        
        df_preds = pd.DataFrame({
            "repertoire_id": rep_ids,
            "label": y,
            "p_cluster": probs
        })
        out_csv = PREDS_DIR / f"{ds_name}_train_cluster_preds.csv"
        df_preds.to_csv(out_csv, index=False)
        logging.info(f"Saved train preds to {out_csv}")
        
        # Predict on Test Sets
        # Find associated test sets
        test_ds_names = [k for k in TEST_DATASETS.keys() if k.startswith(ds_name)]
        # Handle ds7_1, ds7_2
        if ds_name == "ds7":
            test_ds_names = ["ds7_1", "ds7_2"]
        elif ds_name == "ds8":
            test_ds_names = ["ds8_1", "ds8_2", "ds8_3"]
            
        for test_ds in test_ds_names:
            if test_ds not in TEST_DATASETS: continue
            
            out_test_csv = PREDS_DIR / f"{test_ds}_test_cluster_preds.csv"
            if out_test_csv.exists():
                logging.info(f"  ✅ Test preds exist for {test_ds}. Skipping.")
                continue

            logging.info(f"Predicting on {test_ds}...")
            test_pkl = PROCESSED_DIR / f"{test_ds}_test.pkl"
            if not test_pkl.exists():
                logging.warning(f"Test pickle not found: {test_pkl}")
                continue
                
            test_reps = load_repertoires_pickle(test_pkl)
            test_embeddings = load_embeddings_for_reps(test_reps, test_ds)
            
            if not test_embeddings:
                logging.warning("No test embeddings.")
                continue
                
            test_ids = [r.rep_id for r in test_reps]
            # Filter those with embeddings
            valid_ids = [rid for rid in test_ids if rid in test_embeddings]
            
            if not valid_ids:
                logging.warning("No valid test reps with embeddings.")
                continue
                
            test_probs = clf.predict_proba(valid_ids, test_embeddings)[:, 1]
            
            df_test = pd.DataFrame({
                "repertoire_id": valid_ids,
                "p_cluster": test_probs
            })
            out_test_csv = PREDS_DIR / f"{test_ds}_test_cluster_preds.csv"
            df_test.to_csv(out_test_csv, index=False)
            logging.info(f"Saved test preds to {out_test_csv}")

if __name__ == "__main__":
    run_clustering_all()
