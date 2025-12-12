import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
import sys
import gc
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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

def load_embedding_single(dataset_name: str, rep_id: str):
    """
    Load a single embedding file.
    """
    emb_ds_name = dataset_name.split("_")[0]
    
    # Try direct matches first
    candidates = [
        EMBEDDINGS_DIR / dataset_name / f"{rep_id}.npy",
        EMBEDDINGS_DIR / emb_ds_name / f"{rep_id}.npy"
    ]
    
    for p in candidates:
        if p.exists():
            try:
                emb = np.load(p)
                if emb.ndim == 1:
                    if len(emb) == 0: return None
                    emb = emb.reshape(1, -1)
                return emb
            except:
                return None
    return None

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
            
        model_path = MODELS_DIR / f"{ds_name}_cluster_model.joblib"
        
        # We need to perform subsampling first to build the clusters
        logging.info("  Phase 1: Subsampling sequences for clustering...")
        
        # Target: 200k sequences
        TOTAL_TARGET = 200000
        n_per_rep = max(1, TOTAL_TARGET // len(labeled_reps))
        
        X_subsampled = []
        origin_labels = []
        
        # List of rep_ids and labels for later
        rep_ids = []
        y_labels = []
        
        for r in tqdm(labeled_reps, desc="  Subsampling"):
            rep_ids.append(r.rep_id)
            y_labels.append(r.label)
            
            emb = load_embedding_single(ds_name, r.rep_id)
            if emb is not None and len(emb) > 0:
                # Sample
                n = len(emb)
                if n > n_per_rep:
                    idx = np.random.choice(n, n_per_rep, replace=False)
                    sampled = emb[idx]
                else:
                    sampled = emb
                X_subsampled.append(sampled)
                origin_labels.extend([r.label] * len(sampled))
                
                del emb
                
        if not X_subsampled:
            logging.warning("  No embeddings found. Skipping.")
            continue
            
        X_sub = np.vstack(X_subsampled).astype('float32')
        y_origin = np.array(origin_labels)
        del X_subsampled
        gc.collect()
        
        # Initialize and Cluster
        logging.info("  Phase 2: Running Clustering (FAISS + Louvain)...")
        clf = ClusterClassifier(k_neighbors=10, resolution=1.0, n_clusters_to_keep=50)
        clf.fit_clustering(X_sub, y_origin)
        
        del X_sub, y_origin
        gc.collect()
        
        # Phase 3: Featurize all repertoires
        logging.info("  Phase 3: Featurizing repertoires...")
        X_features = []
        valid_indices = [] # keep track of which reps had embeddings
        
        for i, r in enumerate(tqdm(labeled_reps, desc="  Featurizing")):
            emb = load_embedding_single(ds_name, r.rep_id)
            if emb is not None:
                feat = clf.transform_repertoire(emb)
                X_features.append(feat)
                valid_indices.append(i)
                del emb
            else:
                # If missing, maybe zero vector? or skip?
                # If we skip, we lose alignment with y.
                # Let's clean y.
                pass
        
        # Filter y to match X_features
        y_train = np.array([y_labels[i] for i in valid_indices])
        X_train_feat = np.vstack(X_features)
        
        del X_features
        gc.collect()
        
        # Phase 4: Train Classifier
        logging.info("  Phase 4: Training Classifier...")
        clf.fit_classifier(X_train_feat, y_train)
        
        joblib.dump(clf, model_path)
        logging.info(f"  Saved cluster model to {model_path}")
        
        # Generate Train Preds (Overfitted)
        logging.info("  Generating train preds...")
        probs = clf.predict_proba(X_train_feat)[:, 1]
        
        train_rep_ids = [rep_ids[i] for i in valid_indices]
        df_preds = pd.DataFrame({
            "repertoire_id": train_rep_ids,
            "label": y_train,
            "p_cluster": probs
        })
        out_csv = PREDS_DIR / f"{ds_name}_train_cluster_preds.csv"
        df_preds.to_csv(out_csv, index=False)
        
        # Predict on Test Sets
        test_ds_names = [k for k in TEST_DATASETS.keys() if k.startswith(ds_name.split("_")[0])] 
        # Better logic:
        # ds7 -> ds7, ds7_1, ds7_2
        # ds1 -> ds1
        
        current_base = ds_name.split("_")[0]
        targets = []
        for k in TEST_DATASETS.keys():
            if k == ds_name or k == current_base:
                targets.append(k)
            elif k.startswith(current_base + "_"):
                targets.append(k)
        
        targets = sorted(list(set(targets)))
            
        for test_ds in targets:
            out_test_csv = PREDS_DIR / f"{test_ds}_test_cluster_preds.csv"
            if out_test_csv.exists():
                logging.info(f"    ✅ Test preds exist for {test_ds}. Skipping.")
                continue
 
            logging.info(f"  Predicting on {test_ds}...")
            # Load Pickle
            candidates = [
                PROCESSED_DIR / f"{test_ds}_test.pkl",
                PROCESSED_DIR / f"{test_ds}_1_test.pkl",
            ]
            test_reps = None
            for p in candidates:
                if p.exists():
                    test_reps = load_repertoires_pickle(p)
                    break
            
            if not test_reps: continue
            
            test_feats = []
            test_ids = []
            
            for r in tqdm(test_reps, desc=f"    Featurizing {test_ds}"):
                emb = load_embedding_single(test_ds, r.rep_id)
                if emb is not None:
                    feat = clf.transform_repertoire(emb)
                    test_feats.append(feat)
                    test_ids.append(r.rep_id)
                    del emb
            
            if test_feats:
                X_test_feat = np.vstack(test_feats)
                probs_test = clf.predict_proba(X_test_feat)[:, 1]
                
                df_test = pd.DataFrame({
                    "repertoire_id": test_ids,
                    "p_cluster": probs_test
                })
                df_test.to_csv(out_test_csv, index=False)
                logging.info(f"    Saved test preds to {out_test_csv}")
            else:
                logging.warning("    No test features generated.")

if __name__ == "__main__":
    run_clustering_all()
