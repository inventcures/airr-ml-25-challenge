import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import joblib
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from collections import Counter

from data.load_data import Repertoire

class RepertoireStatsClassifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=random_state))
        ])
        self.known_v_genes: List[str] = []
        self.known_j_genes: List[str] = []
        self.amino_acids = sorted("ACDEFGHIKLMNPQRSTVWY")

    def _extract_features(self, reps: List[Repertoire]) -> pd.DataFrame:
        """
        Extract global statistics features from a list of Repertoires.
        """
        rows = []
        for r in reps:
            # 1. V and J gene counts
            v_counts = Counter(r.v_call)
            j_counts = Counter(r.j_call)
            n_seqs = len(r.junction_aa)
            if n_seqs == 0:
                # Handle empty repertoire edge case
                rows.append({})
                continue

            # Normalize to frequencies
            feats = {}
            
            # We will fill in 0s for missing genes later based on self.known_v_genes if set,
            # or just collect all now and impute 0 later.
            # Actually, for training, we define the vocab. For inference, we use it.
            
            for v, count in v_counts.items():
                feats[f"v_{v}"] = count / n_seqs
            for j, count in j_counts.items():
                feats[f"j_{j}"] = count / n_seqs

            # 2. CDR3 Length stats
            lengths = [len(s) for s in r.junction_aa]
            if lengths:
                feats["len_mean"] = np.mean(lengths)
                feats["len_std"] = np.std(lengths)
                feats["len_median"] = np.median(lengths)
            else:
                feats["len_mean"] = 0
                feats["len_std"] = 0
                feats["len_median"] = 0

            # 3. Amino Acid Composition
            # Concatenate all sequences to count AA freq globally in the repertoire
            all_seqs = "".join(r.junction_aa)
            total_aa = len(all_seqs)
            aa_counts = Counter(all_seqs)
            
            for aa in self.amino_acids:
                if total_aa > 0:
                    feats[f"aa_{aa}"] = aa_counts[aa] / total_aa
                else:
                    feats[f"aa_{aa}"] = 0.0
            
            rows.append(feats)

        df = pd.DataFrame(rows).fillna(0.0)
        return df

    def fit(self, reps: List[Repertoire], y: List[int]):
        # 1. Build vocabulary from training data
        # Extract all potential V/J columns first
        # We need to do a first pass or just use the dataframe columns
        
        # To ensure consistent columns, we extract features, then freeze the columns
        df_feats = self._extract_features(reps)
        
        # Identify V and J columns
        self.known_v_genes = sorted([c for c in df_feats.columns if c.startswith("v_")])
        self.known_j_genes = sorted([c for c in df_feats.columns if c.startswith("j_")])
        
        # Filter/Reorder DataFrame to ensure specific order and only known columns + static stats
        # Actually, we should keep all columns found in training.
        self.feature_names_ = sorted(df_feats.columns.tolist())
        
        X = df_feats[self.feature_names_].values
        self.model.fit(X, y)
        return self

    def predict_proba(self, reps: List[Repertoire]) -> np.ndarray:
        df_feats = self._extract_features(reps)
        
        # Align columns to training features
        # Add missing columns as 0
        missing_cols = set(self.feature_names_) - set(df_feats.columns)
        for c in missing_cols:
            df_feats[c] = 0.0
            
        # Select only training columns in correct order
        X = df_feats[self.feature_names_].values
        return self.model.predict_proba(X)

    def cross_val_predict(self, reps: List[Repertoire], y: List[int], cv: int = 5) -> np.ndarray:
        """
        Generate cross-validated probability predictions for the training set.
        This is crucial for the meta-ensemble to avoid overfitting.
        """
        # We need to ensure consistent feature extraction across folds.
        # But wait, if we split first, each fold might have different genes.
        # Better strategy: Extract features for ALL, then align to the union of features?
        # Or just use the fit logic per fold.
        
        # Let's extract features for the whole dataset first to get the superset of columns?
        # No, strictly speaking, we should fit on train fold and transform test fold.
        # But the gene vocabulary is usually shared.
        # Let's just use the simple approach: Extract features for all, fill 0s.
        # Then use sklearn's cross_val_predict.
        
        df_feats = self._extract_features(reps)
        X = df_feats.fillna(0.0).values
        # Note: This uses whatever columns are present in the whole train set.
        # It's slightly leaky if a gene only appears in one fold, but for V/J usage it's minor.
        
        # We need to wrap the model to handle the feature alignment if we were doing it strictly,
        # but since we already extracted X, we can just pass X to cross_val_predict with the pipeline.
        
        # However, the pipeline expects dense input.
        return cross_val_predict(
            self.model, 
            X, 
            y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state), 
            method="predict_proba"
        )

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> 'RepertoireStatsClassifier':
        return joblib.load(path)
