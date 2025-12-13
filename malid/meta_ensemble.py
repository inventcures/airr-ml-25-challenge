import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

# HYBRID APPROACH: Try XGBoost (RunPod), Fallback to HistGradientBoosting (Local Mac)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier

class MetaEnsembleClassifier:
    """
    Hybrid Stacker: Uses XGBoost if available, else Scikit-Learn's HistGradientBoosting.
    Ensures maximum performance on server (RunPod) while working locally.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        if HAS_XGBOOST:
            # OPTION A: XGBoost (Gold Standard)
            print("⚡ Using XGBoost (Best Performance)")
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            # OPTION B: HistGradientBoosting (LightGBM equivalent, No Dependencies)
            print("⚠️ XGBoost not found. Using HistGradientBoosting (High Performance Fallback)")
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=random_state,
                scoring='log_loss'
            )
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
