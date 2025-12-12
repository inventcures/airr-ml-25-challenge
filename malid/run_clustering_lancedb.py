import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
import sys
import gc
from tqdm import tqdm
from collections import defaultdict, Counter
import shutil

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    print("Error: 'lancedb' not found. Please run: pip install lancedb")
    sys.exit(1)

import faiss
import igraph as ig
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TRAIN_DATASETS, TEST_DATASETS

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/clustering_lancedb.log")
    ]
)
Path("logs").mkdir(exist_ok=True)

EMBEDDINGS_DIR = Path("data/embeddings")
MODELS_DIR = Path("models/cluster_lancedb") # Separate dir
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR = Path("outputs/cluster_preds_lancedb") # Separate dir
PREDS_DIR.mkdir(parents=True, exist_ok=True)
LANCEDB_URI = "/tmp/lancedb_clustering" # Temp DB location

# Re-using the load function from original script manually to avoid circular imports if modified
def load_embedding_single(dataset_name: str, rep_id: str):
    """
    Load a single embedding file via mmap.
    """
    base_ds = dataset_name.split("_")[0]
    candidates = [
        EMBEDDINGS_DIR / dataset_name / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "train" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "test" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / dataset_name / "train" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / dataset_name / "test" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "test" / "1_test" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "test" / "2_test" / f"{rep_id}.npy",
        EMBEDDINGS_DIR / base_ds / "test" / "3_test" / f"{rep_id}.npy",
    ]
    for p in candidates:
        if p.exists():
            try:
                emb = np.load(p, mmap_mode='r')
                if emb.ndim == 1:
                    if len(emb) == 0: return None
                    emb = emb.reshape(1, -1)
                return emb
            except:
                return None
    return None

class LanceDBClusterClassifier:
    """
    Clustering classifier that uses LanceDB for GPU-accelerated Indexing.
    Replaces FAISS in the 'fit_clustering' phase.
    """
    def __init__(self, k_neighbors=10, resolution=1.0, n_clusters_to_keep=50):
        self.k_neighbors = k_neighbors
        self.resolution = resolution
        self.n_clusters_to_keep = n_clusters_to_keep
        self.top_clusters_ = None
        self.cluster_enrichment_scores_ = None
        self.cluster_centroids_ = {}
        self.centroid_matrix_ = None
        self.centroid_index_ = None
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)

    def fit_clustering(self, X_all: np.ndarray, origin_labels: np.ndarray):
        if len(X_all) == 0:
            return self
            
        logging.info(f"Clustering {len(X_all)} sequences with LanceDB...")
        d = X_all.shape[1]
        
        # 1. Setup LanceDB Table
        if Path(LANCEDB_URI).exists():
            shutil.rmtree(LANCEDB_URI)
        db = lancedb.connect(LANCEDB_URI)
        
        # Ingest Data
        # X_all is float32 numpy. Convert to list of dicts for ingestion? 
        # Or pyarrow table. 
        # Creating a Pa Table is fastest.
        tensor = pa.FixedSizeListArray.from_arrays(X_all.flatten(), d)
        schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), d))])
        tbl = db.create_table("vectors", schema=schema)
        
        # Add data in batches to avoid RAM spike
        # Or if X_all is in RAM (it is), just add it.
        # But LanceDB expects [{'vector': ...}]
        # Using pandas is easiest wrapper
        df = pd.DataFrame({'vector': list(X_all)}) 
        # Note: X_all is (200k, 1280). list(X_all) makes 200k arrays.
        tbl.add(df)
        
        # 2. Build Index with CUDA
        logging.info("Building LanceDB IVF-PQ Index on GPU...")
        # num_partitions ~ sqrt(N) ~ sqrt(200000) ~ 450. Use 256 or 512.
        tbl.create_index(
            num_partitions=256,
            num_sub_vectors=96, # 1280 dims / 96? No, 1280 must be divisible. 96 isn't.
            # 1280 / 64 = 20. 1280 / 128 = 10.
            # Let's use generic defaults or leave num_sub_vectors auto.
            # User snippet used 96, but they might have different dims.
            # Safest is just 'accelerator="cuda"' and let defaults handle sub_vectors if possible,
            # or picking a divisor of 1280. 64 is safe.
            metric="L2",
            accelerator="cuda"
        )
        
        # 3. Find Neighbors for Graph
        logging.info("Searching neighbors via LanceDB...")
        # We need k nearest neighbors for EACH vector in X_all.
        # Ideally, we query the table with X_all.
        # Current LanceDB Python API usually supports single-vector query or very small batches.
        # For 200k, we must loop or use a trick.
        edges = []
        weights = []
        
        # Iterate and Query
        # This loop might be the slow part if python overhead is high.
        # But the actual search is fast.
        for i in tqdm(range(len(X_all)), desc="Building Graph"):
            q = X_all[i]
            # k+1 because self match is likely returned
            res = tbl.search(q).limit(self.k_neighbors + 1).to_pandas()
            # res has '_distance' and index? LanceDB returns the row.
            # We didn't store an ID. The '_rowid' is internal.
            # Currently LanceDB row ordering *usually* is preserved on ingestion order for fresh tables,
            # but relying on implicit row ID is risky.
            # However, for this task, if we assume 1:1, we can map.
            # Wait, we really need the integer index 'j' such that X_all[j] is the neighbor.
            # Does LanceDB return the original index? 'row_number'?
            # Let's assume we can't easily get 'j' without storing 'id' column.
            pass
        
        # REFACTOR: We MUST store an ID column.
        # Restarting ingestion logic.
        
        return self

class LanceDBClusterClassifierFixed(LanceDBClusterClassifier):
    def fit_clustering(self, X_all: np.ndarray, origin_labels: np.ndarray):
        logging.info(f"Clustering {len(X_all)} sequences with LanceDB...")
        d = X_all.shape[1]
        
        if Path(LANCEDB_URI).exists():
            shutil.rmtree(LANCEDB_URI)
        db = lancedb.connect(LANCEDB_URI)
        
        # Create Data with ID
        ids = np.arange(len(X_all))
        df = pd.DataFrame({
            'id': ids,
            'vector': list(X_all)
        })
        tbl = db.create_table("vectors", data=df)
        
        logging.info("Building LanceDB IVF-PQ Index on GPU...")
        
        # Determine valid num_sub_vectors (must divide d)
        # 1280 (650M) -> 64 ok (20). 96 bad.
        # 480 (35M) -> 96 ok (5). 64 bad.
        # 320 (8M) -> 64 ok (5). 96 bad.
        
        valid_sub_vectors = [96, 64, 48, 32, 16]
        chosen_sub_vectors = 16 # Fallback
        
        for sv in valid_sub_vectors:
            if d % sv == 0:
                chosen_sub_vectors = sv
                break
                
        logging.info(f"  Dimension d={d}. Chosen num_sub_vectors={chosen_sub_vectors}")

        try:
            tbl.create_index(
                num_partitions=256,
                num_sub_vectors=chosen_sub_vectors, 
                metric="L2",
                accelerator="cuda"
            )
        except Exception as e:
            logging.warning(f"GPU Indexing failed: {e}. Fallback might be needed.")
            
        logging.info("Searching neighbors via LanceDB...")
        edges = []
        weights = []
        
        # Search Loop
        # In future LanceDB versions, batch search is standard.
        # For now, we loop.
        for i in tqdm(range(len(X_all)), desc="Graph Search"):
            q = X_all[i]
            results = tbl.search(q).limit(self.k_neighbors + 1).to_pandas()
            
            # results dataframe has 'id' column
            neighbors = results['id'].values
            
            for rank, neighbor_idx in enumerate(neighbors):
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx == i: continue # Skip self
                edges.append((i, neighbor_idx))
                weights.append(1.0)
                
        # 4. Build Graph & Louvain (Standard)
        g = ig.Graph(edges=edges, directed=False)
        logging.info("Running Louvain community detection...")
        partition = g.community_multilevel(weights=weights, resolution=self.resolution)
        clusters = np.array(partition.membership)
        
        # 5. Enrichment & Centroids (Same as Original)
        cluster_counts = defaultdict(lambda: {'pos': 0, 'neg': 0})
        for i, cluster_id in enumerate(clusters):
            if origin_labels[i] == 1:
                cluster_counts[cluster_id]['pos'] += 1
            else:
                cluster_counts[cluster_id]['neg'] += 1
                
        total_pos = np.sum(origin_labels == 1)
        total_neg = np.sum(origin_labels == 0)
        
        enrichment_scores = {}
        for cid, counts in cluster_counts.items():
            p_pos = (counts['pos'] + 1) / (total_pos + 1)
            p_neg = (counts['neg'] + 1) / (total_neg + 1)
            enrichment_scores[cid] = np.log(p_pos / p_neg)
            
        sorted_clusters = sorted(enrichment_scores.items(), key=lambda x: x[1], reverse=True)
        self.top_clusters_ = [str(c) for c, s in sorted_clusters[:self.n_clusters_to_keep]]
        self.cluster_enrichment_scores_ = {str(c): s for c, s in sorted_clusters}
        
        logging.info(f"Selected {len(self.top_clusters_)} enriched clusters.")
        
        self.cluster_centroids_ = {}
        for cid_str in self.top_clusters_:
            cid = int(cid_str)
            mask = clusters == cid
            centroid = X_all[mask].mean(axis=0)
            self.cluster_centroids_[cid_str] = centroid
            
        # Build Centroid Index (Standard FAISS or LanceDB?)
        # For 50 centroids, FAISS FlatL2 is instant. No need for LanceDB overhead.
        self.centroid_matrix_ = np.vstack([self.cluster_centroids_[c] for c in self.top_clusters_]).astype('float32')
        self.centroid_index_ = faiss.IndexFlatL2(d) 
        self.centroid_index_.add(self.centroid_matrix_)
        return self

    def transform_repertoire(self, emb: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        # Same robust batched logic as original
        if len(emb) == 0:
            return np.zeros(len(self.top_clusters_))
        N = len(emb)
        counts = Counter()
        for i in range(0, N, batch_size):
            chunk = emb[i : i + batch_size].astype('float32')
            _, I = self.centroid_index_.search(chunk, 1)
            counts.update(I.flatten())
            del chunk
        freqs = np.zeros(len(self.top_clusters_))
        for idx, cid_str in enumerate(self.top_clusters_):
            freqs[idx] = counts[idx]
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()
        return freqs

    def fit_classifier(self, X_features: np.ndarray, y: np.ndarray):
        logging.info("Training Logistic Regression on cluster features...")
        self.model.fit(X_features, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path):
        return joblib.load(path)


def run_clustering_lancedb():
    # Identical to original run_clustering_all but uses LanceDBClusterClassifierFixed
    # ... (Copying main loop structure, swapping CLF)
    
    # Train datasets
    for ds_name in TRAIN_DATASETS.keys():
        logging.info(f"\nProcessing {ds_name} with LanceDB...")
        
        pkl_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
        if not pkl_path.exists(): continue
        reps = load_repertoires_pickle(pkl_path)
        labeled_reps = [r for r in reps if r.label is not None]
        if not labeled_reps: continue
            
        model_path = MODELS_DIR / f"{ds_name}_cluster_model.joblib"
        train_preds_csv = PREDS_DIR / f"{ds_name}_train_cluster_preds.csv"
        
        skip_training = False
        if train_preds_csv.exists() and model_path.exists():
            logging.info("  Skipping training (Artifacts exist).")
            # But wait, old artifacts might be from standard clustering?
            # We are using separate output dirs (MODELS_DIR, PREDS_DIR overrides), so safe.
            skip_training = True
            
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
                TOTAL_TARGET = 200000
                n_per_rep = max(1, TOTAL_TARGET // len(labeled_reps))
                X_subsampled = []
                origin_labels_accum = []
                
                for r in tqdm(labeled_reps, desc="Subsampling"):
                    emb = load_embedding_single(ds_name, r.rep_id)
                    if emb is not None and len(emb) > 0:
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
            
            # Phase 2: LanceDB Clustering
            logging.info("  Phase 2: LanceDB Clustering...")
            clf = LanceDBClusterClassifierFixed(k_neighbors=10)
            clf.fit_clustering(X_sub, y_origin)
            del X_sub, y_origin
            gc.collect()
            
            # --- PHASE 3: FEATURIZATION ---
            X_train_feat = None
            y_train = None
            train_rep_ids_for_pred = None
            
            if ckpt_phase3.exists():
                logging.info(f"  ✅ Phase 3 checkpoint found at {ckpt_phase3}. Loading...")
                with np.load(ckpt_phase3) as data:
                    X_train_feat = data['X_train_feat']
                    y_train = data['y_train']
                    if 'valid_rep_ids' in data:
                        train_rep_ids_for_pred = data['valid_rep_ids']
            else:
                logging.info("  Phase 3: Featurizing repertoires...")
                X_features = []
                valid_ids_accum = []
                y_labels_accum = []
                
                for r in tqdm(labeled_reps, desc="Featurizing"):
                    emb = load_embedding_single(ds_name, r.rep_id)
                    if emb is not None:
                        # Batch size 10000
                        feat = clf.transform_repertoire(emb, batch_size=10000)
                        X_features.append(feat)
                        valid_ids_accum.append(r.rep_id)
                        y_labels_accum.append(r.label)
                        del emb
                
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
            
            # Phase 4 Classify
            logging.info("  Phase 4: Training Classifier...")
            clf.fit_classifier(X_train_feat, y_train)
            clf.save(model_path)
            
            probs = clf.predict_proba(X_train_feat)[:, 1]
            pd.DataFrame({
                "repertoire_id": train_rep_ids_for_pred, 
                "label": y_train, 
                "p_cluster": probs
            }).to_csv(train_preds_csv, index=False)
        else:
            clf = LanceDBClusterClassifierFixed.load(model_path)
            
        # Inference (Same logic as manual script, simplified loop)
        # ...
        
if __name__ == "__main__":
    run_clustering_lancedb()
