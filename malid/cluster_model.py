import numpy as np
import pandas as pd
import faiss
import igraph as ig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.utils.validation import check_is_fitted
import joblib
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from collections import Counter, defaultdict
import logging

from data.load_data import Repertoire

class ClusterClassifier:
    def __init__(
        self, 
        k_neighbors: int = 10, 
        resolution: float = 1.0, 
        n_clusters_to_keep: int = 50,
        random_state: int = 42
    ):
        self.k_neighbors = k_neighbors
        self.resolution = resolution
        self.n_clusters_to_keep = n_clusters_to_keep
        self.random_state = random_state
        
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=random_state))
        ])
        
        # State
        self.cluster_enrichment_scores_: Dict[str, float] = {} # cluster_id -> score (log odds or similar)
        self.top_clusters_: List[str] = [] # List of cluster IDs to use as features
        self.seq_to_cluster_: Dict[str, str] = {} # sequence -> cluster_id (for inference mapping)
        
        # We also need to store the FAISS index or representative sequences if we want to map new sequences?
        # For this competition, we might just map exact sequences if they overlap, or use nearest neighbor?
        # The prompt says: "Save a sequence-cluster map: for each sequence".
        # If we want to support new sequences (Task 1 test set), we need to map them to existing clusters.
        # Ideally, we'd use the FAISS index to find the nearest cluster for new sequences.
        # But for simplicity/speed, maybe we just use exact match first?
        # "Mal-ID Model 2" usually implies mapping new sequences to clusters.
        # Let's keep the FAISS index if possible, or just exact match for now if embeddings are pre-computed.
        # Wait, this model usually runs on Embeddings.
        # The README says: "Build a FAISS L2 index on all embeddings."
        # So this class expects EMBEDDINGS, not just raw sequences.
        # But `Repertoire` object has `junction_aa`.
        # We need the embeddings.
        # The embeddings are generated in Step 3.
        # So I cannot fully run this until I have embeddings.
        pass

    def fit_clustering(self, X_all: np.ndarray, origin_labels: np.ndarray):
        """
        Step 1: Build graph and find clusters using subsampled sequences.
        X_all: (N_subsampled, D)
        origin_labels: (N_subsampled,) 0 or 1
        """
        if len(X_all) == 0:
            logging.warning("No sequences provided for clustering.")
            return self
            
        logging.info(f"Clustering {len(X_all)} sequences...")
        d = X_all.shape[1]
        
        # 2. Build FAISS Index
        # TRY GPU FIRST if available
        try:
            res = faiss.StandardGpuResources()
            # On GPU, FlatL2 is extremely fast (brute force parallelized)
            # and saves System RAM.
            logging.info("GPU FAISS resources detected. Using GPU for clustering index.")
            index_cpu = faiss.IndexFlatL2(d)
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        except AttributeError:
            # Fallback to CPU Optimized HNSW
            logging.info("GPU FAISS not detected. Using CPU HNSW index.")
            index = faiss.IndexHNSWFlat(d, 32) 
        
        index.add(X_all)
        
        # 3. Find Neighbors for Graph
        # Search for k neighbors
        D, I = index.search(X_all, self.k_neighbors + 1) # +1 because self is included
        
        # 4. Build Graph
        edges = []
        weights = []
        for i in range(len(X_all)):
            for j_idx in range(1, self.k_neighbors + 1): # Skip self (index 0)
                neighbor = I[i, j_idx]
                # dist = D[i, j_idx]
                if neighbor != -1:
                    edges.append((i, neighbor))
                    weights.append(1.0)
        
        g = ig.Graph(edges=edges, directed=False)
        
        # 5. Louvain Clustering
        logging.info("Running Louvain community detection...")
        partition = g.community_multilevel(weights=weights, resolution=self.resolution)
        clusters = np.array(partition.membership)
        
        # 6. Calculate Enrichment
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
            # Add pseudocounts
            p_pos = (counts['pos'] + 1) / (total_pos + 1)
            p_neg = (counts['neg'] + 1) / (total_neg + 1)
            score = np.log(p_pos / p_neg)
            enrichment_scores[cid] = score
            
        # Select top clusters (most enriched for disease)
        sorted_clusters = sorted(enrichment_scores.items(), key=lambda x: x[1], reverse=True)
        self.top_clusters_ = [str(c) for c, s in sorted_clusters[:self.n_clusters_to_keep]]
        self.cluster_enrichment_scores_ = {str(c): s for c, s in sorted_clusters}
        
        logging.info(f"Selected {len(self.top_clusters_)} enriched clusters.")
        
        # Store representative centroids for inference mapping
        self.cluster_centroids_ = {}
        for cid_str in self.top_clusters_:
            cid = int(cid_str)
            mask = clusters == cid
            centroid = X_all[mask].mean(axis=0)
            self.cluster_centroids_[cid_str] = centroid
            
        # Build index for centroids for fast inference
        self.centroid_matrix_ = np.vstack([self.cluster_centroids_[c] for c in self.top_clusters_]).astype('float32')
        self.centroid_index_ = faiss.IndexFlatL2(d)
        self.centroid_index_.add(self.centroid_matrix_)
        return self

    def transform_repertoire(self, emb: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        """
        Map a single repertoire embedding to cluster feature vector.
        emb: (N_seqs, D) - Can be a memory-mapped array.
        Returns: (N_clusters,) vector (frequencies)
        """
        if len(emb) == 0:
            return np.zeros(len(self.top_clusters_))
            
        N = len(emb)
        counts = Counter()
        
        # Process in batches to avoid loading full mmap array into RAM
        for i in range(0, N, batch_size):
            # Slice: accessing mmap reads this chunk into RAM
            chunk = emb[i : i + batch_size].astype('float32')
            
            # FAISS search
            _, I = self.centroid_index_.search(chunk, 1)
            counts.update(I.flatten())
            
            # Explicit delete for safety
            del chunk
            
        # Create feature vector
        freqs = np.zeros(len(self.top_clusters_))
        for idx, cid_str in enumerate(self.top_clusters_):
            # FAISS index corresponds to the order in centroid_matrix_ 
            # which was built from self.top_clusters_ order.
            # So I[i] == k means it belongs to the k-th cluster in our list.
            freqs[idx] = counts[idx]
            
        # Normalize
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()
            
        return freqs

    def fit_classifier(self, X_features: np.ndarray, y: np.ndarray):
        """
        Train the logistic regression on the cluster features.
        """
        logging.info("Training Logistic Regression on cluster features...")
        self.model.fit(X_features, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for feature vectors X.
        X: (N_samples, N_features)
        """
        return self.model.predict_proba(X)

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> 'ClusterClassifier':
        obj = joblib.load(path)
        # Check for new attributes
        if not hasattr(obj, 'cluster_centroids_') or not hasattr(obj, 'centroid_index_'):
             raise ValueError("Incompatible ClusterClassifier model. Please retrain.")
        
        # Check if fitted
        try:
            check_is_fitted(obj.model)
        except Exception as e:
            raise ValueError(f"Model components not fitted: {e}")
            
        return obj
