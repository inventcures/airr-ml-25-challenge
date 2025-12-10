import numpy as np
import pandas as pd
import faiss
import igraph as ig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import joblib
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from collections import Counter, defaultdict

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

    def fit(self, rep_ids: List[str], y: List[int], sequence_embeddings: Dict[str, np.ndarray]):
        """
        sequence_embeddings: dict mapping rep_id -> array [N, D]
        """
        logging.info("Fitting ClusterClassifier...")
        
        # 1. Subsample sequences
        # We want a balanced set of sequences from positive and negative repertoires
        pos_reps = [rid for rid, label in zip(rep_ids, y) if label == 1]
        neg_reps = [rid for rid, label in zip(rep_ids, y) if label == 0]
        
        # Target: 200k total sequences (100k pos, 100k neg)
        # Or just proportional.
        total_seqs_target = 200000
        
        # Collect all embeddings
        X_all = []
        origin_labels = [] # 0 or 1
        
        # Helper to sample from a list of reps
        def sample_from_reps(reps, target_n):
            if not reps: return
            n_per_rep = max(1, target_n // len(reps))
            for rid in reps:
                if rid not in sequence_embeddings: continue
                emb = sequence_embeddings[rid]
                if len(emb) == 0: continue
                
                # Sample
                if len(emb) > n_per_rep:
                    indices = np.random.choice(len(emb), n_per_rep, replace=False)
                    sampled = emb[indices]
                else:
                    sampled = emb
                
                X_all.append(sampled)
                # Track label for enrichment
                label = 1 if rid in pos_reps else 0
                origin_labels.extend([label] * len(sampled))

        logging.info(f"Subsampling sequences (Target: {total_seqs_target})...")
        sample_from_reps(pos_reps, total_seqs_target // 2)
        sample_from_reps(neg_reps, total_seqs_target // 2)
        
        if not X_all:
            logging.warning("No sequences found for clustering!")
            return self
            
        X_all = np.vstack(X_all).astype('float32')
        origin_labels = np.array(origin_labels)
        logging.info(f"Clustering {len(X_all)} sequences...")
        
        # 2. Build FAISS Index
        d = X_all.shape[1]
        index = faiss.IndexFlatL2(d)
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
                dist = D[i, j_idx]
                if neighbor != -1:
                    edges.append((i, neighbor))
                    # Weight could be exp(-dist) or just 1
                    weights.append(1.0)
        
        g = ig.Graph(edges=edges, directed=False)
        
        # 5. Louvain Clustering
        logging.info("Running Louvain community detection...")
        partition = g.community_multilevel(weights=weights, resolution=self.resolution)
        clusters = np.array(partition.membership)
        
        # 6. Calculate Enrichment
        # P(Cluster | Disease) vs P(Cluster | Healthy)
        # We use log odds ratio or similar.
        # Simple enrichment: (Count_Pos / Total_Pos) / (Count_Neg / Total_Neg)
        
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
        # For each top cluster, compute mean embedding
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
        
        # 7. Featurize Training Data
        # We need to map ALL sequences (not just subsampled) to these clusters.
        # But for training speed, maybe we just use the subsampled ones?
        # No, we need one vector per repertoire.
        # We must map all sequences in the repertoires to the centroids.
        
        X_features = self._featurize_repertoires(rep_ids, sequence_embeddings)
        
        # 8. Train Classifier
        logging.info("Training Logistic Regression on cluster features...")
        self.model.fit(X_features, y)
        
        return self

    def _featurize_repertoires(self, rep_ids, sequence_embeddings):
        X_feat = np.zeros((len(rep_ids), len(self.top_clusters_)))
        
        for i, rid in enumerate(rep_ids):
            if rid not in sequence_embeddings: continue
            emb = sequence_embeddings[rid]
            if len(emb) == 0: continue
            
            emb = emb.astype('float32')
            
            # Find nearest cluster centroid for each sequence
            # D: [N_seqs, 1], I: [N_seqs, 1]
            _, I = self.centroid_index_.search(emb, 1)
            
            # Count occurrences of each cluster index (0 to K-1)
            counts = Counter(I.flatten())
            
            for idx, count in counts.items():
                if idx < len(self.top_clusters_):
                    X_feat[i, idx] = count
                    
        # Normalize (frequency instead of counts)
        # Or let StandardScaler handle it?
        # Counts are better for "burden", but frequency handles varying repertoire size.
        # Let's use frequency.
        row_sums = X_feat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_feat = X_feat / row_sums
        
        return X_feat

    def predict_proba(self, rep_ids: List[str], sequence_embeddings: Dict[str, np.ndarray]):
        X_feat = self._featurize_repertoires(rep_ids, sequence_embeddings)
        return self.model.predict_proba(X_feat)

