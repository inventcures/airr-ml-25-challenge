import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import joblib
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

from data.load_data import Repertoire

class ESMSequenceClassifier:
    def __init__(self, random_state: int = 42, top_k: int = 10):
        self.random_state = random_state
        self.top_k = top_k
        # Sequence-level classifier
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', random_state=random_state))
        ])

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray):
        """
        Fit the sequence-level classifier.
        X_seq: (N_total_seqs, D) embeddings
        y_seq: (N_total_seqs,) labels (inherited from repertoire)
        """
        self.model.fit(X_seq, y_seq)
        return self

    def predict_proba_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for individual sequences.
        Returns (N, 2) array.
        """
        return self.model.predict_proba(X_seq)

    def predict_repertoire(self, X_rep: np.ndarray) -> float:
        """
        Predict repertoire-level probability by aggregating top-k sequence probabilities.
        X_rep: (N_seqs, D) embeddings for one repertoire
        """
        if len(X_rep) == 0:
            return 0.0 # Or 0.5?
            
        # Get sequence-level probabilities
        p_seqs = self.model.predict_proba(X_rep)[:, 1]
        
        # Aggregate: Top-k mean
        # We want to find sequences that are most indicative of the POSITIVE class.
        # So we take the top k highest probabilities.
        if len(p_seqs) < self.top_k:
            k = len(p_seqs)
        else:
            k = self.top_k
            
        top_k_probs = np.sort(p_seqs)[-k:]
        return float(np.mean(top_k_probs))

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> 'ESMSequenceClassifier':
        return joblib.load(path)
