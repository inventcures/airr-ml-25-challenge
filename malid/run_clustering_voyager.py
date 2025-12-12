import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
import sys
import gc
from tqdm import tqdm
from collections import defaultdict, Counter

try:
    from voyager import Index, Space
except ImportError:
    print("Error: 'voyager' not found. Please run: pip install voyager")
    sys.exit(1)

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
        logging.FileHandler("logs/clustering_voyager.log")
    ]
)
Path("logs").mkdir(exist_ok=True)

EMBEDDINGS_DIR = Path("data/embeddings")
MODELS_DIR = Path("models/cluster_voyager")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR = Path("outputs/cluster_preds_voyager")
PREDS_DIR.mkdir(parents=True, exist_ok=True)

# Re-using the load function
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

class VoyagerClusterClassifier:
    """
    Clustering classifier that uses Spotify Voyager for high-speed CPU Indexing & Querying.
    Replaces FAISS/LanceDB in the 'fit_clustering' phase.
    """
    def __init__(self, k_neighbors=10, resolution=1.0, n_clusters_to_keep=50):
        self.k_neighbors = k_neighbors
        self.resolution = resolution
        self.n_clusters_to_keep = n_clusters_to_keep
        self.top_clusters_ = None
        self.cluster_enrichment_scores_ = None
        self.cluster_centroids_ = {}
        self.centroid_index_ = None
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)

    def fit_clustering(self, X_all: np.ndarray, origin_labels: np.ndarray):
        if len(X_all) == 0:
            return self
            
        logging.info(f"Clustering {len(X_all)} sequences with Voyager (CPU)...")
        d = X_all.shape[1]
        
        # 1. Build Index with Voyager
        # Voyager expects C-contiguous float32
        if not X_all.flags['C_CONTIGUOUS']:
            X_all = np.ascontiguousarray(X_all)
            
        # Euclidean space matches L2 logic
        # M=16, ef_construction=200 are standard HNSW params
        index = Index(Space.Euclidean, num_dimensions=d, M=16, ef_construction=200)
        
        logging.info("Adding items to Voyager index...")
        index.add_items(X_all)
        
        # 2. Find Neighbors (Batched)
        logging.info("Querying neighbors (Batch Mode)...")
        # k+1 to account for self-match
        neighbors, distances = index.query(X_all, k=self.k_neighbors + 1)
        
        # 3. Build Graph
        logging.info("Building Graph...")
        edges = []
        weights = []
        
        # Vectorized edge construction is tricky for igraph, loop is safest but optimized
        # neighbors is (N, k+1)
        # We can iterate fast
        for i in tqdm(range(len(X_all)), desc="Graph Edges"):
            row_neighbors = neighbors[i]
            # row_neighbors contains indices of X_all
            for n_idx in row_neighbors:
                if n_idx == i: continue
                # We can perform weighted graph logic based on distance if desired
                # For now using 1.0 weight as per original script
                edges.append((i, n_idx))
                weights.append(1.0)
                
        # 4. Louvain Community Detection
        g = ig.Graph(edges=edges, directed=False)
        logging.info("Running Louvain community detection...")
        partition = g.community_multilevel(weights=weights, resolution=self.resolution)
        clusters = np.array(partition.membership)
        
        # 5. Enrichment & Centroids (Identical Logic)
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
            
        # Build Centroid Index (Voyager for consistency)
        logging.info("Building Centroid Index...")
        centroids_arr = np.vstack([self.cluster_centroids_[c] for c in self.top_clusters_]).astype('float32')
        if not centroids_arr.flags['C_CONTIGUOUS']:
            centroids_arr = np.ascontiguousarray(centroids_arr)
            
        self.centroid_index_ = Index(Space.Euclidean, num_dimensions=d)
        self.centroid_index_.add_items(centroids_arr)
        
        # Clean up main index to free RAM
        del index
        return self

    def transform_repertoire(self, emb: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        if len(emb) == 0:
            return np.zeros(len(self.top_clusters_))
        N = len(emb)
        counts = Counter()
        
        # Voyager requires C-contiguous
        if not emb.flags['C_CONTIGUOUS']:
            emb = np.ascontiguousarray(emb) # This might trigger copy if mmap is strided? mmap usually C-contiguous.
            
        for i in range(0, N, batch_size):
            chunk = emb[i : i + batch_size].astype('float32')
            if not chunk.flags['C_CONTIGUOUS']:
                chunk = np.ascontiguousarray(chunk)
                
            # Query 1 neighbor
            neighbors, _ = self.centroid_index_.query(chunk, k=1)
            counts.update(neighbors.flatten())
            
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


def run_clustering_voyager():
    # Identical structure, uses VoyagerClusterClassifier
    
    # Train datasets
    for ds_name in TRAIN_DATASETS.keys():
        logging.info(f"\nProcessing {ds_name} with Voyager...")
        
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
            skip_training = True
            
        clf = None
        if not skip_training:
            # Phase 1: Subsampling (Copied)
            logging.info("  Phase 1: Subsampling...")
            # Reuse logic
            ckpt_phase1 = MODELS_DIR / f"{ds_name}_phase1_subsampled.npz"
            ckpt_phase3 = MODELS_DIR / f"{ds_name}_phase3_features.npz"
            
            X_sub = None
            y_origin = None
            
            if ckpt_phase1.exists():
                logging.info(f"  ✅ Phase 1 checkpoint found. Loading...")
                with np.load(ckpt_phase1) as data:
                    X_sub = data['X_sub']
                    y_origin = data['y_origin']
            else:
                 # Standard subsample loop
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
                if not X_subsampled: continue
                X_sub = np.vstack(X_subsampled).astype('float32')
                y_origin = np.array(origin_labels_accum)
                
                np.savez_compressed(ckpt_phase1, X_sub=X_sub, y_origin=y_origin)
                del X_subsampled, origin_labels_accum
                gc.collect()
            
            # Phase 2: Voyager Clustering
            logging.info("  Phase 2: Voyager Clustering...")
            clf = VoyagerClusterClassifier(k_neighbors=10)
            clf.fit_clustering(X_sub, y_origin)
            del X_sub, y_origin
            gc.collect()
            
            # Phase 3: Featurization
            logging.info("  Phase 3: Featurizing...")
            X_train_feat = None
            y_train = None
            train_rep_ids = None
            
            if ckpt_phase3.exists():
               with np.load(ckpt_phase3) as data:
                   X_train_feat = data['X_train_feat']
                   y_train = data['y_train']
                   if 'valid_rep_ids' in data:
                       train_rep_ids = data['valid_rep_ids']
            else:
                X_features = []
                valid_ids = []
                y_train_list = []
                for r in tqdm(labeled_reps, desc="Featurizing"):
                    emb = load_embedding_single(ds_name, r.rep_id)
                    if emb is not None:
                        feat = clf.transform_repertoire(emb, batch_size=10000)
                        X_features.append(feat)
                        valid_ids.append(r.rep_id)
                        y_train_list.append(r.label)
                        del emb
                
                X_train_feat = np.vstack(X_features)
                y_train = np.array(y_train_list)
                train_rep_ids = np.array(valid_ids)
                np.savez_compressed(ckpt_phase3, X_train_feat=X_train_feat, y_train=y_train, valid_rep_ids=train_rep_ids)
                del X_features
                gc.collect()

            # Phase 4
            logging.info("  Phase 4: Training Classifier...")
            clf.fit_classifier(X_train_feat, y_train)
            clf.save(model_path)
            
            probs = clf.predict_proba(X_train_feat)[:, 1]
            # Use train_rep_ids if available, else derive? 
            # In Phase 3 else block we set it. In if block we loaded it.
            # If load failed to find ids, we might have issue. Assuming happy path or fresh run.
            if train_rep_ids is None:
                 # Fallback if checkpoint didn't have ids (legacy)
                 pass 
            
            pd.DataFrame({
                "repertoire_id": train_rep_ids, 
                "label": y_train, 
                "p_cluster": probs
            }).to_csv(train_preds_csv, index=False)
            
        else:
            clf = VoyagerClusterClassifier.load(model_path)
            
        # Predict on Test Sets (Ported from run_clustering_all.py)
        # Identify related test sets (e.g., ds1 -> ds1_test)
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
                PROCESSED_DIR / f"{test_ds}.pkl", 
            ]
            test_reps = None
            for p in candidates:
                if p.exists():
                    test_reps = load_repertoires_pickle(p)
                    break 
            
            if not test_reps:
                 # Try matching part explicitly
                 p = PROCESSED_DIR / f"{test_ds}_test.pkl"
                 if p.exists(): test_reps = load_repertoires_pickle(p)

            if not test_reps: 
                logging.warning(f"    Could not load test pickle for {test_ds}")
                continue
            
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
    run_clustering_voyager()
