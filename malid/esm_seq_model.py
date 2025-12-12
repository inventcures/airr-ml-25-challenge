import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Note: Pipeline doesn't support partial_fit easily on the pipeline object itself in some versions, 
# but we can manually call partial_fit on steps.
# However, StandardScaler needs to know partial_fit.
# Let's keep the pipeline structure but manage partial_fit manually.

import joblib
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

from data.load_data import Repertoire

class ESMSequenceClassifier:
    def __init__(self, random_state: int = 42, top_k: int = 10):
        self.random_state = random_state
        self.top_k = top_k
        
        # We need independent steps for partial_fit control
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(
            loss='log_loss', # equivalent to logistic regression
            penalty='l2', 
            class_weight='balanced', 
            random_state=random_state,
            n_jobs=-1 
        )
        
        # Pipeline for easy full fit/predict if needed, but we will use components for partial_fit
        # Actually, let's just use components. 
        # But we want compatibility with existing code that calls self.model.fit
        # So we can keep self.model as a property or just wrap methods.
        pass

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray):
        """
        Fit the sequence-level classifier (Full Batch).
        """
        X_scaled = self.scaler.fit_transform(X_seq)
        self.clf.fit(X_scaled, y_seq)
        return self

    def partial_fit(self, X_seq: np.ndarray, y_seq: np.ndarray, classes: Optional[np.ndarray] = None):
        """
        Incremental fit for batch processing.
        """
        X_scaled = self.scaler.partial_fit(X_seq).transform(X_seq)
        self.clf.partial_fit(X_scaled, y_seq, classes=classes)
        return self

    def predict_proba_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for individual sequences.
        Returns (N, 2) array.
        """
        X_scaled = self.scaler.transform(X_seq)
        return self.clf.predict_proba(X_scaled)

    def predict_repertoire(self, X_rep: np.ndarray) -> float:
        """
        Predict repertoire-level probability by aggregating top-k sequence probabilities.
        X_rep: (N_seqs, D) embeddings for one repertoire
        """
        if len(X_rep) == 0:
            return 0.5 
            
        # Get sequence-level probabilities
        # Check if fitted first? 
        # SGDClassifier raises error if not fitted.
        X_scaled = self.scaler.transform(X_rep)
        p_seqs = self.clf.predict_proba(X_scaled)[:, 1]
        
        # Aggregate: Top-k mean
        if len(p_seqs) < self.top_k:
            k = len(p_seqs)
        else:
            k = self.top_k
            
        top_k_probs = np.sort(p_seqs)[-k:]
        return float(np.mean(top_k_probs))

    def save(self, path: Path):
        # Save both components
        joblib.dump({'scaler': self.scaler, 'clf': self.clf, 'top_k': self.top_k}, path)

from sklearn.utils.validation import check_is_fitted

    @staticmethod
    def load(path: Path) -> 'ESMSequenceClassifier':
        data = joblib.load(path)
        obj = ESMSequenceClassifier(top_k=data.get('top_k', 10))
        
        if isinstance(data, dict):
            obj.scaler = data['scaler']
            obj.clf = data['clf']
            
            # Validate fitted state
            try:
                check_is_fitted(obj.scaler)
                check_is_fitted(obj.clf)
            except Exception as e:
                raise ValueError(f"Model components not fitted: {e}")
        else:
            raise ValueError("Incompatible model format found. Please retrain.")
        return obj


