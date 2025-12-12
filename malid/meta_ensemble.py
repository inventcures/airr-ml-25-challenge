import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import List, Dict, Optional

"""
Meta-Ensemble Layer (Stacking Classifier)

SYSTEM ARCHITECTURE
-------------------
This component implements the final fusion layer of our solution.
It uses **Stacking**, a powerful ensemble technique.

INPUTS:
The input to this meta-learner is NOT raw sequence data.
Instead, it takes the **prediction probabilities** from the three base streams:
1. Stream 1: DeepRC OOF Probabilities (`p_deeprc`)
2. Stream 2: ESM Sequence Classifier OOF Probabilities (`p_esm`)
3. Stream 3: Clustering MIL OOF Probabilities (`p_cluster`)
(And optionally `p_stats` if available).

LOGIC:
We train a Logistic Regression on these probabilities.
This allows the system to learn which stream is trustworthy.
- If DeepRC is confident but ESM is unsure, the meta-learner learns to weigh DeepRC higher.
- If streams disagree, it finds the optimal weighted consensus.

ROBUSTNESS:
The `predict_proba` method handles missing columns (e.g., if one stream failed for a test set) by imputing neutral probability (0.5), ensuring the pipeline never crashes in production.
"""

class MetaEnsembleClassifier:
    """
    Logistic Regression Stacker for combining multi-modal OOF predictions.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Simple logistic regression to combine probabilities
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=random_state))
        ])
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X: DataFrame with columns like 'p_stats', 'p_esm', 'p_deeprc'
        y: Series of labels
        """
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Ensure columns match
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            # Fill missing with 0.5 (neutral)? Or 0?
            # Probabilities are usually 0-1. 0.5 is safe.
            for c in missing:
                X[c] = 0.5
        
        X = X[self.feature_names_]
        return self.model.predict_proba(X)

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> 'MetaEnsembleClassifier':
        return joblib.load(path)
