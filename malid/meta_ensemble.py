import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import List, Dict, Optional

class MetaEnsembleClassifier:
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
