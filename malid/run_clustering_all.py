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
    Main execution loop for Clustering-Based MIL Model (Bag-of-Motifs).
    
    SYSTEM ARCHITECTURE
    -------------------
    This script implements the "Stream 3" component of our solution: Structural Clustering.
    It treats the repertoire as a distribution of common sequence motifs ("Visual Words").
    
    PIPELINE STAGES (Phases):
    -------------------------
    1. Subsampling (Phase 1):
       - Goal: Discover common motifs across the population without loading all data.
       - Logic: We stream each repertoire (using mmap) and sample a small fraction of sequences.
       - Output: A pool of ~200k representative sequences (X_sub).
       - Checkpoint: `dsX_phase1_subsampled.npz`.
       
    2. Clustering (Phase 2):
       - Goal: Group similar sequences into distinct "motifs" or clusters.
       - Logic:
         a. FAISS Indexing: Build an HNSW or FlatL2 index (GPU transparently used if available).
         b. Graph Construction: Find k=10 nearest neighbors for every sequence.
         c. Community Detection: Use Louvain algorithm to discover dense clusters.
       - Output: A trained ClusterClassifier with centroids.
       
    3. Featurization (Phase 3):
       - Goal: Convert every patient's repertoire into a fixed-size histogram of motifs.
       - Logic: We re-scan every repertoire. For each sequence, we find the nearest cluster centroid.
       - Optimization: Uses `BATCH_SIZE=10000` chunking to ensure large repertoires never exceed RAM.
       - Output: A matrix X_features (N_patients x N_clusters).
       - Checkpoint: `dsX_phase3_features.npz`.
       
    4. Classification (Phase 4):
       - Goal: Predict disease status from the motif histogram.
       - Logic: Standard Logistic Regression on the features.
       - Output: Probability scores for Stream 3.
       
    MEMORY SAFETY FEATURES:
    - uses `mmap_mode='r'` for all embedding loads.
    - Explicit `gc.collect()` between phases.
    - Granular checkpointing allows resumability after preemption.
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
            # Checkpoint Paths
            ckpt_phase1 = MODELS_DIR / f"{ds_name}_phase1_subsampled.npz"
            ckpt_phase3 = MODELS_DIR / f"{ds_name}_phase3_features.npz"
            
            # --- PHASE 1: SUBSAMPLING ---
            X_sub = None
            y_origin = None
            
            if ckpt_phase1.exists():
                logging.info(f"  ✅ Phase 1 checkpoint found at {ckpt_phase1}. Loading...")
                with np.load(ckpt_phase1) as data:
                    X_sub = data['X_sub']
                    y_origin = data['y_origin']
            else:
                logging.info("  Phase 1: Subsampling sequences for clustering...")
                
                # Target: 200k sequences
                TOTAL_TARGET = 200000
                n_per_rep = max(1, TOTAL_TARGET // len(labeled_reps))
                
                X_subsampled = []
                origin_labels_accum = []
                
                for r in tqdm(labeled_reps, desc="  Subsampling"):
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
                        origin_labels_accum.extend([r.label] * len(sampled))
                        
                        del emb
                        
                if not X_subsampled:
                    logging.warning("  No embeddings found. Skipping.")
                    continue
                    
                X_sub = np.vstack(X_subsampled).astype('float32')
                y_origin = np.array(origin_labels_accum)
                
                # Save Checkpoint
                np.savez_compressed(ckpt_phase1, X_sub=X_sub, y_origin=y_origin)
                logging.info(f"  Saved Phase 1 checkpoint to {ckpt_phase1}")
                
                del X_subsampled, origin_labels_accum
                gc.collect()
            
            # --- PHASE 2: CLUSTERING ---
            # Clustering depends on Phase 1 data.
            # We don't checkpoint the model loop internally, but we save the model at the end.
            logging.info("  Phase 2: Running Clustering (FAISS + Louvain)...")
            clf = ClusterClassifier(k_neighbors=10, resolution=1.0, n_clusters_to_keep=50)
            clf.fit_clustering(X_sub, y_origin)
            
            del X_sub, y_origin
            gc.collect()
            
            # --- PHASE 3: FEATURIZATION ---
            X_train_feat = None
            y_train = None
            
            if ckpt_phase3.exists():
                logging.info(f"  ✅ Phase 3 checkpoint found at {ckpt_phase3}. Loading...")
                with np.load(ckpt_phase3) as data:
                    X_train_feat = data['X_train_feat']
                    y_train = data['y_train']
                    
                # We need valid_indices to match rep_ids for predictions later.
                # If loading from checkpoint, we assume y_train is aligned with SOMETHING.
                # Use rep_ids from labeled_reps filtering validation?
                # Actually, filtering might be tricky if we don't save valid_indices.
                # Let's save rep_ids in checkpoint too? Or just assume 1:1 if we are careful.
                # Current code filters `valid_indices`.
                # Let's verify valid_indices is saved or re-derived.
                # Re-deriving is fast (check file existence).
                # Better: Save 'valid_rep_ids' in checkpoint.
                if 'valid_rep_ids' in data:
                   saved_ids = data['valid_rep_ids'] # Array of strings
                   # Reconstruct valid_indices? No need if we have the IDs for the dataframe.
                   train_rep_ids_for_pred = saved_ids
                else:
                    # Fallback or error?
                    pass
            else:
                logging.info("  Phase 3: Featurizing repertoires...")
                X_features = []
                valid_ids_accum = []
                y_labels_accum = []
                
                for i, r in enumerate(tqdm(labeled_reps, desc="  Featurizing")):
                    emb = load_embedding_single(ds_name, r.rep_id)
                    if emb is not None:
                        # NEW: Explicit batch_size for memory safety
                        feat = clf.transform_repertoire(emb, batch_size=10000)
                        X_features.append(feat)
                        valid_ids_accum.append(r.rep_id)
                        y_labels_accum.append(r.label)
                        del emb
                    else:
                        pass
                
                y_train = np.array(y_labels_accum)
                X_train_feat = np.vstack(X_features)
                train_rep_ids_for_pred = np.array(valid_ids_accum)
                
                # Save Checkpoint
                np.savez_compressed(ckpt_phase3, 
                                    X_train_feat=X_train_feat, 
                                    y_train=y_train,
                                    valid_rep_ids=train_rep_ids_for_pred)
                logging.info(f"  Saved Phase 3 checkpoint to {ckpt_phase3}")
                
                del X_features, valid_ids_accum, y_labels_accum
                gc.collect()
            
            # --- PHASE 4: CLASSIFICATION ---
            logging.info("  Phase 4: Training Classifier...")
            clf.fit_classifier(X_train_feat, y_train)
            
            clf.save(model_path)
            logging.info(f"  Saved cluster model to {model_path}")
            
            # Generate Train Preds (Overfitted)
            # Need rep_ids aligned with X_train_feat
            # If we loaded from checkpoint, we used 'valid_rep_ids'
            
            # Predict
            probs = clf.predict_proba(X_train_feat)[:, 1]
            
            df_preds = pd.DataFrame({
                "repertoire_id": train_rep_ids_for_pred,
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
