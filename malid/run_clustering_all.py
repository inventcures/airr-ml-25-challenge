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
    Scanning multiple probable paths.
    """
    base_ds = dataset_name.split("_")[0]
    
    candidates = [
        # Base matches
        EMBEDDINGS_DIR / dataset_name / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / f"{rep_id}.npy",
        # Nested Matches
        EMBEDDINGS_DIR / base_ds / "train" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "test" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / dataset_name / "train" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / dataset_name / "test" / f"{rep_id}.npy",
        # Multipart Test Matches
        EMBEDDINGS_DIR / base_ds / "test" / "1_test" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "test" / "2_test" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "test" / "3_test" / f"{rep_id}.npy",
    ]
    
    # Just iterate and try to load first one found
    for p in candidates:
        if p.exists():
            try:
                # Use mmap_mode='r' to avoid loading entire file into RAM.
                # This is crucial for large embedding files.
                emb = np.load(p, mmap_mode='r')
                if emb.ndim == 1:
                    if len(emb) == 0: return None
                    emb = emb.reshape(1, -1)
                return emb
            except:
                return None
    return None

def run_clustering_all():
    """
    Main execution loop for Clustering-Based MIL Model.
    
    STRATEGY: Memory-Safe Streaming
    -------------------------------
    To avoid OOM errors when processing massive embedding datasets, we do NOT load all data at once.
    Instead, we break the process into 4 distinct phases:
    
    1. Subsampling: Iterate repertoires one-by-one, load their embeddings, sample a small fraction, 
       and discard the rest. We collect ~200k sequences total across the dataset.
       
    2. Clustering: Use FAISS to build a k-NN graph of the subsampled sequences and run Louvain clustering.
       This defines our "visual words" or sequence motifs.
       
    3. Featurization: Iterate repertoires one-by-one again. For each repertoire, map all its sequences 
       to the nearest cluster centroids found in Phase 2. This converts a repertoire of N sequences 
       into a single fixed-size vector (histogram of cluster counts).
       
    4. Classification: Train a standard Logistic Regression on these fixed-size feature vectors.
    """
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
        
        # Check if already done and valid
        train_preds_csv = PREDS_DIR / f"{ds_name}_train_cluster_preds.csv"
        skip_training = False
        
        if train_preds_csv.exists() and model_path.exists():
            try:
                ClusterClassifier.load(model_path)
                logging.info(f"  ✅ Clustering artifacts (Model + Train Preds) exist and are valid for {ds_name}. Skipping training.")
                skip_training = True
            except Exception as e:
                logging.warning(f"  ⚠️ Found existing cluster model for {ds_name} but it is invalid/old ({e}). Retraining...")
        
        clf = None
        if not skip_training:
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
                    pass
            
            # Filter y to match X_features
            y_train = np.array([y_labels[i] for i in valid_indices])
            X_train_feat = np.vstack(X_features)
            
            del X_features
            gc.collect()
            
            # Phase 4: Train Classifier
            logging.info("  Phase 4: Training Classifier...")
            clf.fit_classifier(X_train_feat, y_train)
            
            clf.save(model_path)
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
            df_preds.to_csv(train_preds_csv, index=False)
        else:
           # Load if skipped
           clf = ClusterClassifier.load(model_path)
        
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
