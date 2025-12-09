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
        # 1. Aggregate all sequences and embeddings
        # We need to keep track of which repertoire they came from to compute enrichment.
        
        # This is memory intensive. We might need to subsample.
        # README: "Subsample up to max_seqs_per_rep sequences"
        pass
